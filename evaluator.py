"""
RAG Evaluation with full LangSmith tracing.

Every Q&A turn is wrapped in a @traceable run that appears in LangSmith as:
  rag_turn  (parent)
  ├── retrieval_eval   — all retrieval metrics as output
  └── generation_eval  — all generation metrics as output (LLM-as-judge)

Retrieval metrics (deterministic, no Bedrock cost):
  hit_rate          — ≥1 retrieved chunk contains the answer keywords
  mrr               — mean reciprocal rank of first relevant chunk
  ndcg              — normalised discounted cumulative gain @K
  precision_at_k    — fraction of top-K chunks that are relevant
  chunk_diversity   — unique source-file count / K  (0=all same file, 1=all different)
  bm25_contribution — fraction of fused candidates from BM25
  rerank_score_top1 — cross-encoder score of the #1 chunk

Generation metrics (LLM-as-judge via Bedrock, each scored 0-1):
  faithfulness        — answer stays within provided context (no hallucination)
  answer_relevance    — answer directly addresses the question
  context_utilisation — how much of the retrieved context is used in the answer
  answer_completeness — answer covers all aspects of the question
"""

import json
import logging
import os
import re
import uuid
from math import log2
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------

def _ls_client():
    """Return a LangSmith Client if credentials are available, else None."""
    try:
        from langsmith import Client
        api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            return None
        return Client(api_key=api_key)
    except Exception:
        return None


def _is_tracing_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def _project() -> str:
    return os.getenv("LANGCHAIN_PROJECT", "aetherrag-poc")


# ---------------------------------------------------------------------------
# Relevance heuristic (shared by all retrieval metrics)
# ---------------------------------------------------------------------------

def _is_relevant(chunk_text: str, query: str, answer_keywords: list[str]) -> bool:
    """Chunk is relevant if it contains ≥50% of answer keywords, or ≥33% of query tokens."""
    chunk_lower = chunk_text.lower()
    if answer_keywords:
        hits = sum(1 for kw in answer_keywords if kw.lower() in chunk_lower)
        return hits >= len(answer_keywords) * 0.5
    tokens = [t.lower() for t in query.split() if len(t) > 3]
    if not tokens:
        return False
    return sum(1 for t in tokens if t in chunk_lower) >= max(1, len(tokens) // 3)


# ---------------------------------------------------------------------------
# Retrieval metrics (deterministic)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    contexts: list[dict],
    query: str,
    n_bm25: int = 0,
    answer_keywords: list[str] = None,
) -> dict:
    kw = answer_keywords or []
    k  = len(contexts)

    if k == 0:
        return {
            "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0,
            "precision_at_k": 0.0, "chunk_diversity": 0.0,
            "bm25_contribution": 0.0, "rerank_score_top1": 0.0,
            "num_chunks": 0,
        }

    relevances = [1 if _is_relevant(c["snippet"], query, kw) else 0 for c in contexts]

    # Hit rate
    hit_rate = 1.0 if any(relevances) else 0.0

    # MRR
    mrr = 0.0
    for i, rel in enumerate(relevances):
        if rel:
            mrr = 1.0 / (i + 1)
            break

    # NDCG
    dcg  = sum(rel / log2(i + 2) for i, rel in enumerate(relevances))
    idcg = sum(1.0 / log2(i + 2) for i in range(min(sum(relevances), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Precision@K
    precision_at_k = sum(relevances) / k

    # Chunk diversity — unique source files / K
    unique_sources = len({c.get("source", "") for c in contexts})
    chunk_diversity = round(unique_sources / k, 3)

    # BM25 contribution
    bm25_contribution = round(n_bm25 / k, 3)

    # Rerank score of top-1
    rerank_score_top1 = round(contexts[0].get("rerank_score", 0.0), 4)

    return {
        "hit_rate":          round(hit_rate,          3),
        "mrr":               round(mrr,               3),
        "ndcg":              round(ndcg,               3),
        "precision_at_k":    round(precision_at_k,    3),
        "chunk_diversity":   chunk_diversity,
        "bm25_contribution": bm25_contribution,
        "rerank_score_top1": rerank_score_top1,
        "num_chunks":        k,
    }


# ---------------------------------------------------------------------------
# Generation metrics (LLM-as-judge via Bedrock)
# ---------------------------------------------------------------------------

def _call_bedrock_judge(bedrock_client, model_id: str, prompt: str) -> float:
    """
    Send a scoring prompt to Bedrock. Expects a numeric reply (0–5).
    Normalises to [0, 1]. Returns 0.5 on any failure.
    """
    try:
        if "anthropic.claude" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            })
            resp = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            text = json.loads(resp["body"].read())["content"][0]["text"].strip()
        elif "amazon.titan" in model_id:
            body = json.dumps({
                "inputText": prompt + "\nScore:",
                "textGenerationConfig": {"maxTokenCount": 8, "temperature": 0.0},
            })
            resp = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            text = json.loads(resp["body"].read())["results"][0]["outputText"].strip()
        else:
            return 0.5

        m = re.search(r"[\d.]+", text)
        val = float(m.group()) if m else 0.5
        # Scores returned as 0-5 → normalise to 0-1
        return round(min(1.0, max(0.0, val / 5.0 if val > 1.0 else val)), 3)

    except Exception as e:
        logger.warning(f"Bedrock judge error: {e}")
        return 0.5


def compute_generation_metrics(
    query: str,
    answer: str,
    contexts: list[dict],
    bedrock_client,
    model_id: str,
) -> dict:
    context_str = "\n\n".join(c.get("snippet", c.get("text", "")) for c in contexts)
    ctx_preview  = context_str[:2500]
    ans_preview  = answer[:1200]

    faithfulness_prompt = (
        "Rate the FAITHFULNESS of the ANSWER on a scale of 0 to 5.\n"
        "5 = every claim in the answer is directly supported by the context, no hallucinations.\n"
        "0 = the answer contains fabricated information not present in the context.\n\n"
        f"CONTEXT:\n{ctx_preview}\n\n"
        f"ANSWER:\n{ans_preview}\n\n"
        "Reply with a single number 0–5."
    )

    relevance_prompt = (
        "Rate how well the ANSWER addresses the QUESTION on a scale of 0 to 5.\n"
        "5 = answer directly and completely answers every part of the question.\n"
        "0 = answer is off-topic or does not address the question at all.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{ans_preview}\n\n"
        "Reply with a single number 0–5."
    )

    completeness_prompt = (
        "Rate the COMPLETENESS of the ANSWER on a scale of 0 to 5.\n"
        "5 = the answer covers all aspects and sub-questions implied by the question.\n"
        "0 = the answer is missing most key aspects of the question.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{ans_preview}\n\n"
        "Reply with a single number 0–5."
    )

    # Context utilisation — heuristic: % of context key phrases echoed in answer
    ctx_sentences = [
        s.strip() for s in context_str.replace("\n", ". ").split(". ")
        if len(s.strip()) > 20
    ]
    answer_lower = answer.lower()
    if ctx_sentences:
        hits = sum(
            1 for s in ctx_sentences
            if any(w in answer_lower for w in s.lower().split()[:5] if len(w) > 4)
        )
        ctx_utilisation = round(min(1.0, hits / len(ctx_sentences)), 3)
    else:
        ctx_utilisation = 0.0

    faithfulness  = _call_bedrock_judge(bedrock_client, model_id, faithfulness_prompt)
    relevance     = _call_bedrock_judge(bedrock_client, model_id, relevance_prompt)
    completeness  = _call_bedrock_judge(bedrock_client, model_id, completeness_prompt)

    return {
        "faithfulness":          faithfulness,
        "answer_relevance":      relevance,
        "answer_completeness":   completeness,
        "context_utilisation":   ctx_utilisation,
    }


# ---------------------------------------------------------------------------
# LangSmith feedback poster
# ---------------------------------------------------------------------------

def post_feedback_to_langsmith(run_id: str, metrics: dict):
    """Post every numeric metric as a LangSmith feedback score on run_id."""
    if not run_id:
        return
    client = _ls_client()
    if client is None:
        return
    try:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=float(value),
                    feedback_source_type="api",
                    source_run_id=str(uuid.uuid4()),
                )
        logger.info(f"Posted {len(metrics)} feedback scores to LangSmith run {run_id}.")
    except Exception as e:
        logger.warning(f"Failed to post feedback to LangSmith: {e}")


# ---------------------------------------------------------------------------
# RAGEvaluator — main class used by app.py
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Wraps one complete RAG turn (retrieval + generation) in a LangSmith
    traced run and computes / logs all evaluation metrics.

    Usage:
        evaluator = RAGEvaluator(bedrock_client=..., model_id=...)
        metrics, run_id = evaluator.evaluate_turn(query, answer, contexts, n_bm25=n)
    """

    def __init__(self, bedrock_client=None, model_id: str = ""):
        self.bedrock_client = bedrock_client
        self.model_id       = model_id
        self.history: list[dict] = []

    def evaluate_turn(
        self,
        query: str,
        answer: str,
        contexts: list[dict],
        n_bm25: int = 0,
        answer_keywords: list[str] = None,
        run_generation_eval: bool = True,
        citation_check: dict | None = None,
    ) -> tuple[dict, Optional[str]]:
        """
        Evaluate one Q&A turn.

        citation_check: optional output of RagEngine.verify_citations — when
            provided, citation verification rate + raw extracted/unverified
            lists are added to the metrics dict and the LangSmith trace.

        Returns:
            (metrics_dict, langsmith_run_id)
            run_id is None when tracing is disabled or LangSmith is unreachable.
        """
        run_id = None

        if _is_tracing_enabled():
            run_id, all_metrics = self._evaluate_traced(
                query, answer, contexts, n_bm25, answer_keywords,
                run_generation_eval, citation_check,
            )
        else:
            all_metrics = self._compute_all(
                query, answer, contexts, n_bm25, answer_keywords,
                run_generation_eval, citation_check,
            )

        turn = {"query": query[:120], **all_metrics}
        self.history.append(turn)
        logger.info(f"Eval turn: {turn}")
        return all_metrics, run_id

    # ------------------------------------------------------------------

    def _compute_all(
        self,
        query: str,
        answer: str,
        contexts: list[dict],
        n_bm25: int,
        answer_keywords: Optional[list[str]],
        run_generation_eval: bool,
        citation_check: Optional[dict] = None,
    ) -> dict:
        r = compute_retrieval_metrics(contexts, query, n_bm25, answer_keywords)
        g = {}
        if run_generation_eval and self.bedrock_client and self.model_id:
            g = compute_generation_metrics(
                query, answer, contexts, self.bedrock_client, self.model_id
            )
        c = {}
        if citation_check:
            c = {
                "citation_verification_rate": citation_check["verification_rate"],
                "citations_extracted":        len(citation_check["extracted"]),
                "citations_unverified":       len(citation_check["unverified"]),
            }
        return {**r, **g, **c}

    def _evaluate_traced(
        self,
        query: str,
        answer: str,
        contexts: list[dict],
        n_bm25: int,
        answer_keywords: Optional[list[str]],
        run_generation_eval: bool,
        citation_check: Optional[dict] = None,
    ) -> tuple[Optional[str], dict]:
        """Run evaluation inside a LangSmith @traceable parent run."""
        try:
            from langsmith import traceable
            from langsmith.run_helpers import get_current_run_tree
        except ImportError:
            return None, self._compute_all(
                query, answer, contexts, n_bm25, answer_keywords,
                run_generation_eval, citation_check,
            )

        captured: dict = {}
        run_id_box:  list = [None]

        @traceable(
            name="rag_turn",
            project_name=_project(),
            tags=["rag", "evaluation"],
            metadata={"model_id": self.model_id},
        )
        def _run(q: str, ans: str) -> dict:
            # Capture run_id while inside the traceable context
            try:
                tree = get_current_run_tree()
                if tree:
                    run_id_box[0] = str(tree.id)
            except Exception:
                pass

            # ── Retrieval sub-span ────────────────────────────────────────
            @traceable(name="retrieval_eval", project_name=_project(), tags=["retrieval"])
            def _retrieval_eval(query: str) -> dict:
                return compute_retrieval_metrics(contexts, query, n_bm25, answer_keywords)

            r_metrics = _retrieval_eval(q)

            # ── Generation sub-span ───────────────────────────────────────
            g_metrics = {}
            if run_generation_eval and self.bedrock_client and self.model_id:
                @traceable(name="generation_eval", project_name=_project(), tags=["generation"])
                def _generation_eval(query: str, answer: str) -> dict:
                    return compute_generation_metrics(
                        query, answer, contexts, self.bedrock_client, self.model_id
                    )
                g_metrics = _generation_eval(q, ans)

            # ── Citation sub-span ─────────────────────────────────────────
            c_metrics = {}
            if citation_check:
                @traceable(name="citation_eval", project_name=_project(), tags=["citation"])
                def _citation_eval() -> dict:
                    return {
                        "citation_verification_rate": citation_check["verification_rate"],
                        "citations_extracted":        len(citation_check["extracted"]),
                        "citations_unverified":       len(citation_check["unverified"]),
                        "extracted_list":             [f"{s} p.{p}" for s, p in citation_check["extracted"]],
                        "unverified_list":            [f"{s} p.{p}" for s, p in citation_check["unverified"]],
                    }
                c_metrics = _citation_eval()

            return {**r_metrics, **g_metrics, **c_metrics}

        all_metrics = _run(query, answer)
        captured    = all_metrics
        run_id      = run_id_box[0]

        # Post all scores as LangSmith feedback on the parent run
        if run_id:
            post_feedback_to_langsmith(run_id, captured)

        return run_id, captured

    # ------------------------------------------------------------------

    def session_summary(self) -> dict:
        """Average of all numeric metrics across every turn this session."""
        if not self.history:
            return {}
        keys = [k for k in self.history[0] if k != "query" and isinstance(self.history[0][k], (int, float))]
        return {
            k: round(sum(t.get(k, 0) for t in self.history) / len(self.history), 3)
            for k in keys
        }
