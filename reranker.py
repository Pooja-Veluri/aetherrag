"""
Cross-encoder reranker using sentence-transformers.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - ~22M params, runs on CPU in ~5ms per pair
  - Scores query-document relevance directly (not embeddings)

Usage:
  reranker = CrossEncoderReranker()
  ranked   = reranker.rerank(query, candidates, top_n=5)
"""

import logging
import math
from functools import lru_cache
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mmr_select(
    candidates: list[dict],
    embed_fn: Callable[[list[str]], list[list[float]]],
    top_n: int,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance selection.

    Picks `top_n` chunks from `candidates` that balance:
      - relevance to the query (proxied by `rerank_score`, fallback `rrf_score`)
      - dissimilarity from chunks already picked (cosine on embeddings)

    lambda_param ∈ [0,1]:
      1.0 → pure relevance (= same as cross-encoder top-N)
      0.0 → pure diversity (ignores relevance)
      0.7 → relevance-leaning, drops near-duplicate chunks (good default)

    Best-effort: any embedding failure short-circuits to relevance ordering.
    """
    if not candidates:
        return []
    if len(candidates) <= top_n:
        return candidates

    # Pre-compute embeddings for every candidate using the snippet (the actual
    # retrieval target — `text` may be a wider parent window for hierarchical).
    texts = [c.get("snippet") or c.get("text", "") for c in candidates]
    try:
        embeddings = embed_fn(texts)
    except Exception as e:
        logger.warning(f"MMR embed failed ({e}); skipping diversity reranking.")
        return candidates[:top_n]

    if len(embeddings) != len(candidates):
        logger.warning("MMR embed returned wrong cardinality; skipping.")
        return candidates[:top_n]

    def relevance(c: dict) -> float:
        # rerank_score lives in [-10, 10] for ms-marco-MiniLM; rrf_score in [0, ~0.03].
        # We don't need the absolute scale to be correct — only the ordering.
        if c.get("rerank_score") is not None:
            return float(c["rerank_score"])
        return float(c.get("rrf_score", 0.0))

    remaining = list(range(len(candidates)))
    selected: list[int] = []

    # First pick: highest pure relevance
    first = max(remaining, key=lambda i: relevance(candidates[i]))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < top_n:
        def mmr_score(i: int) -> float:
            rel = relevance(candidates[i])
            sim_to_selected = max(
                _cosine(embeddings[i], embeddings[s]) for s in selected
            )
            return lambda_param * rel - (1 - lambda_param) * sim_to_selected

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


@lru_cache(maxsize=1)
def _load_model():
    """Load model once and cache — avoids re-loading on Streamlit reruns."""
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading cross-encoder model: {_MODEL_NAME}")
    return CrossEncoder(_MODEL_NAME, max_length=512)


class CrossEncoderReranker:
    """
    Reranks a candidate list of chunks using a cross-encoder.

    Each candidate must have a 'text' key (the passage text the LLM will see)
    and a 'snippet' key (the raw chunk for display). All other keys are passed through.
    """

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self._model = None   # lazy load on first call

    def _get_model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Score each candidate against the query and return top_n sorted by score.

        Args:
            query:           The user query string.
            candidates:      List of chunk dicts (must contain 'text').
            top_n:           Max results to return.
            score_threshold: If set, discard results with score below this value.
                             If the threshold filters out EVERY candidate, the
                             top scoring chunk is still returned so callers
                             never get an empty list with non-empty input.

        Returns:
            List of chunk dicts with 'rerank_score' added, sorted descending.
            Always falls back to RRF ordering on cross-encoder failure.
        """
        if not candidates:
            return []

        # Defensive: passages with empty text crash some sentence-transformers
        # versions. Filter them and remember their slot so unranked items
        # don't silently disappear.
        usable = [c for c in candidates if (c.get("text") or "").strip()]
        if not usable:
            return candidates[:top_n]

        try:
            model = self._get_model()
        except Exception as e:
            logger.error(f"Cross-encoder model load failed: {e}. Returning unranked.")
            return [{**c, "rerank_score": 0.0} for c in candidates[:top_n]]

        pairs = [(query, c["text"]) for c in usable]

        try:
            raw_scores = model.predict(pairs, show_progress_bar=False)
            # `predict` returns a numpy array; coerce defensively.
            scores = [float(s) for s in raw_scores]
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}. Returning unranked.")
            return [{**c, "rerank_score": 0.0} for c in candidates[:top_n]]

        scored = sorted(
            zip(scores, usable),
            key=lambda x: x[0],
            reverse=True,
        )

        if score_threshold is None:
            kept = scored[:top_n]
        else:
            kept = [(s, c) for s, c in scored if s >= score_threshold][:top_n]
            # Never return empty when we had usable candidates — keep the top
            # one so the chat path always has something to ground on.
            if not kept and scored:
                kept = scored[:1]

        return [{**c, "rerank_score": round(s, 4)} for s, c in kept]
