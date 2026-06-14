"""
RAG Engine — Hybrid BM25 + Semantic Search with Cross-Encoder Reranking.

Retrieval pipeline:
  1. Semantic search  → top-N candidates from ChromaDB (dense vectors)
  2. BM25 search      → top-N candidates from OpenSearch BM25 index (sparse)
  3. Reciprocal Rank Fusion (RRF) → merge + deduplicate the two candidate lists
  4. Cross-encoder reranking → score all fused candidates, keep top-K

The BM25 index lives in a Dockerised OpenSearch instance (default
http://localhost:9200) — see docker-compose.yml at the repo root.
"""

import hashlib
import json
import logging
import os
import re
import uuid

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import boto3
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from chunker import DocumentChunker
from bm25_index import BM25Index
from reranker import CrossEncoderReranker, mmr_select
from semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

# RRF constant — 60 is the value used in the original paper
_RRF_K = 60

# How many candidates each retriever fetches before fusion + reranking
_CANDIDATE_FETCH = 20

# Aggregation/counting questions trigger a different retrieval path: instead of
# top-K relevance, we scan ALL chunks that contain the filter term so the count
# is exact. Top-K retrieval can't answer "how many X are there" correctly
# because it only ever shows the LLM K rows.
_COUNT_QUERY_PATTERNS = re.compile(
    r"\b(how many|count|number of|total (?:number|count)|list all|"
    r"all of the|every|enumerate)\b",
    re.IGNORECASE,
)

# Common severity / category words we want to treat as the *filter* term
# inside a counting question — e.g. "how many critical incidents" → filter on
# "critical". Order matters: longer phrases first so "p1" doesn't match before
# "priority 1".
_COMMON_FILTER_TERMS = [
    "critical", "high", "medium", "low",
    "priority 1", "priority 2", "priority 3", "priority 4",
    "p1", "p2", "p3", "p4",
    "sev1", "sev2", "sev3", "sev4",
    "open", "closed", "resolved", "in progress", "pending",
    "blocker", "major", "minor", "trivial",
]

# Concurrency for Bedrock embedding calls during ingestion. Titan v1 takes one
# text per request, so latency is API-round-trip-bound, not CPU-bound — a thread
# pool gives a near-linear speedup until Bedrock throttles.
_EMBED_CONCURRENCY = 16


class BedrockTitanEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function backed by amazon.titan-embed-text-v1.
    Raises on empty input or missing response — no silent zero-vector fallback.

    Embedding calls are parallelised with a thread pool: Titan v1 accepts a
    single text per request, so a sequential loop made ingestion O(N) round-trips.
    """

    def __init__(self, bedrock_client, max_workers: int = _EMBED_CONCURRENCY):
        self.bedrock_client = bedrock_client
        self.model_id = "amazon.titan-embed-text-v1"
        self.max_workers = max_workers

    def _embed_one(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Attempted to embed an empty string — check chunker min_chunk_size.")
        body = json.dumps({"inputText": text})
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        if not embedding:
            raise RuntimeError(f"Bedrock returned no embedding for model {self.model_id}.")
        return embedding

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        # Single item → skip pool overhead (queries hit this path).
        if len(input) == 1:
            return [self._embed_one(input[0])]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(input))) as pool:
            return list(pool.map(self._embed_one, input))


def _sanitise_messages_for_claude(history: list[dict]) -> list[dict]:
    """
    Enforce Claude's strict user/assistant alternation requirement.

    Failure modes seen in production:
      1. History starts with an assistant message (compression dropped the leading user turn).
      2. Consecutive same-role messages — can happen when:
         - LangGraph compression splits a turn pair.
         - A prior turn errored and the assistant row was empty/dropped.
         - The same user message was persisted twice (Streamlit rerun race).

    Strategy:
      - Drop empty messages (their absence creates same-role neighbours).
      - Drop leading assistant messages until the first user turn.
      - Merge consecutive same-role messages by joining their content with a
        blank line so no information is lost.

    Always call this on the COMPLETE message list (including the current
    user turn), not on prior history alone — otherwise a trailing user row in
    history collides with the appended new user prompt.
    """
    # Normalise roles + drop empties
    normalised = [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
        for m in history
        if m.get("content", "").strip()
    ]

    # Drop leading assistant messages
    while normalised and normalised[0]["role"] == "assistant":
        normalised.pop(0)

    if not normalised:
        return []

    # Merge consecutive same-role messages
    merged: list[dict] = []
    for msg in normalised:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    return merged


class RagEngine:
    def __init__(
        self,
        aws_access_key=None,
        aws_secret_key=None,
        aws_session_token=None,
        region="us-east-1",
        opensearch_url: Optional[str] = None,
    ):
        # ── AWS / Bedrock ──────────────────────────────────────────────────
        session_params = {}
        if aws_access_key:
            session_params["aws_access_key_id"]     = aws_access_key
        if aws_secret_key:
            session_params["aws_secret_access_key"] = aws_secret_key
        if aws_session_token:
            session_params["aws_session_token"]     = aws_session_token
        if region:
            session_params["region_name"]           = region

        self.session        = boto3.Session(**session_params)
        self.bedrock_client = self.session.client("bedrock-runtime")

        # ── ChromaDB (dense / semantic) ────────────────────────────────────
        self.embedding_function = BedrockTitanEmbeddingFunction(self.bedrock_client)
        self.chroma_client      = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name    = "rag_documents"
        self.collection         = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

        # ── BM25 (sparse) — backed by OpenSearch ──────────────────────────
        os_url    = opensearch_url or os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        self.bm25 = BM25Index(opensearch_url=os_url)

        # ── Cross-encoder reranker ─────────────────────────────────────────
        # score_threshold is None by default — the cross-encoder's raw scores
        # vary by query, so a global cutoff does more harm than good. Callers
        # can pass a threshold per-query if they want strict filtering.
        self.reranker = CrossEncoderReranker()

        # ── Semantic answer cache ──────────────────────────────────────────
        # Reuses Titan embeddings (already in use for ChromaDB) so we don't
        # add a second embedding model. The cache is keyed on the query
        # embedding; hits skip BM25 + Chroma + reranker + Bedrock entirely.
        self.semantic_cache = SemanticCache(
            embed_fn=lambda text: self.embedding_function([text])[0]
        )

    # ── Indexing ──────────────────────────────────────────────────────────

    def reset_collection(self):
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception as e:
            logger.warning(f"Error deleting Chroma collection: {e}")
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self.bm25.clear()
        # Resetting the corpus invalidates every cached answer — they were
        # generated against documents that no longer exist.
        try:
            self.semantic_cache.clear()
        except Exception as e:
            logger.warning(f"Failed to clear semantic cache on reset: {e}")

    def remove_document(self, file_name: str):
        """
        Remove every chunk belonging to `file_name` from both indexes.
        Idempotent: silently no-ops on stale source names.
        """
        # ── ChromaDB ──────────────────────────────────────────────────────
        try:
            self.collection.delete(where={"source": file_name})
        except Exception as e:
            logger.warning(f"Chroma delete by source='{file_name}' failed: {e}")

        # ── BM25 ──────────────────────────────────────────────────────────
        try:
            self.bm25.remove_source(file_name)
        except Exception as e:
            logger.warning(f"BM25 delete by source='{file_name}' failed: {e}")

        # ── Cache ─────────────────────────────────────────────────────────
        # Removing a document can change answers — invalidate the cache.
        try:
            self.semantic_cache.clear()
        except Exception:
            pass

    def add_document(
        self,
        file_name: str,
        pages_data: list[dict],
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        strategy: str = "semantic",
    ):
        """
        Chunk + index into both ChromaDB (dense) and BM25 (sparse).

        Re-ingest is safe: existing chunks for `file_name` are deleted before
        new ones are written. Identical content with the same filename produces
        identical IDs (content-hash-based) so no work is duplicated even on
        Streamlit reruns that re-fire the upload.
        """
        chunker = DocumentChunker(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk_document(pages_data)

        if not chunks:
            logger.warning(f"No chunks produced for '{file_name}' — document may be empty.")
            return

        # Drop any prior chunks for this filename so re-ingest doesn't
        # accumulate duplicates. Chroma/BM25 silently no-op if nothing matches.
        self.remove_document(file_name)

        # Content-hash IDs make re-ingestion of identical content idempotent
        # at the index level — same chunk text → same ID → Chroma upsert and
        # BM25 op_type=create both no-op.
        for c in chunks:
            content_hash = hashlib.sha1(c["text"].encode("utf-8")).hexdigest()[:10]
            c["id"] = f"{file_name}_p{c['page']}_c{c['chunk_index']}_{content_hash}"

        # ── ChromaDB ──────────────────────────────────────────────────────
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "source":      file_name,
                "page":        c["page"],
                "section":     c["section"],
                "chunk_index": c["chunk_index"],
                "strategy":    c["strategy"],
                "parent_text": c["parent_text"],
            } for c in chunks],
        )

        # ── BM25 ──────────────────────────────────────────────────────────
        self.bm25.add_documents(chunks, source=file_name)

        # Adding a new document changes the retrievable corpus, so the existing
        # cached answers (which embed a snapshot of the prior corpus) are no
        # longer guaranteed correct. Note: app.py also scopes the cache key on
        # the file list, so old hits already won't fire — this just reclaims
        # disk space.
        try:
            self.semantic_cache.clear()
        except Exception as e:
            logger.warning(f"Failed to clear semantic cache after ingest: {e}")

        logger.info(f"Indexed {len(chunks)} chunks from '{file_name}' (strategy={strategy}).")

    # ── Hybrid retrieval ──────────────────────────────────────────────────

    def query_context(
        self,
        query: str,
        n_results: int = 5,
        fetch_k: int = _CANDIDATE_FETCH,
        use_reranker: bool = True,
        rerank_score_threshold: Optional[float] = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        groundedness_threshold: Optional[float] = None,
    ) -> dict:
        """
        Hybrid BM25 + Semantic retrieval with cross-encoder reranking and
        MMR diversity selection.

        Pipeline: dense + sparse → RRF fuse → cross-encoder rerank (top n*3) →
        MMR (top n) → optional groundedness gate.

        Returns:
          {
            "chunks":           list[dict],
            "n_bm25":           int,
            "n_semantic":       int,
            "aggregation":      dict | None,
            "low_confidence":   bool,   # true → caller should refuse to answer
            "top_rerank_score": float,
          }
        Each chunk dict: text, snippet, source, page, section,
                         bm25_score, semantic_score, rrf_score, rerank_score
        """
        # 0. Aggregation short-circuit: counting/listing questions can't be
        #    answered correctly from top-K retrieval. Scan the BM25 index for
        #    every matching row and return the exact count alongside chunks.
        aggregation = self._maybe_aggregate(query)

        # 1. Dense semantic retrieval from ChromaDB
        semantic_results = self._semantic_search(query, fetch_k)

        # 2. Sparse BM25 retrieval
        bm25_results = self.bm25.query(query, n_results=fetch_k)

        # 3. Reciprocal Rank Fusion
        fused, n_bm25, n_semantic = self._rrf_merge(semantic_results, bm25_results)

        # 4. Cross-encoder reranking. Always passes the FULL fused list to the
        #    cross-encoder (not a pre-truncated slice) so reranking can pull a
        #    relevant chunk from rank 18 up to the top — that's the whole point
        #    of having two retrieval stages.
        #    We over-fetch (top_n * 3) before MMR so MMR has room to swap in
        #    diverse chunks; final truncation to n_results happens after MMR.
        rerank_pool = max(n_results * 3, n_results)
        if use_reranker and fused:
            reranked = self.reranker.rerank(
                query,
                fused,
                top_n=rerank_pool,
                score_threshold=rerank_score_threshold,
            )
            if not reranked:
                reranked = fused[:rerank_pool]
        else:
            reranked = fused[:rerank_pool]

        # 5. MMR diversity selection — picks the n_results chunks that balance
        #    relevance with diversity, so the LLM doesn't see five near-
        #    duplicate chunks from the same section.
        if use_mmr and len(reranked) > n_results:
            final = mmr_select(
                reranked,
                embed_fn=lambda texts: self.embedding_function(texts),
                top_n=n_results,
                lambda_param=mmr_lambda,
            )
        else:
            final = reranked[:n_results]

        # 6. Groundedness gate — refuse if retrieval is too weak. Only fires
        #    when the caller passes a threshold AND aggregation didn't catch
        #    the question (counting questions are answered from BM25 scan, not
        #    rerank score). Returns chunks=[] with a flag the caller checks.
        low_confidence = False
        if groundedness_threshold is not None and not aggregation:
            top_score = final[0].get("rerank_score", 0.0) if final else 0.0
            if top_score < groundedness_threshold:
                low_confidence = True

        # If aggregation kicked in and the relevance-ranked top-K missed some
        # of the matching rows, splice a sample of unique matches in so the LLM
        # has concrete examples to cite. The exact count comes from the
        # aggregation summary, not from these chunks.
        if aggregation and aggregation["matches"]:
            seen_fp = {f"{c['source']}|{c['page']}|{c['snippet'][:80]}" for c in final}
            for m in aggregation["matches"][:8]:
                fp = f"{m['source']}|{m['page']}|{m['snippet'][:80]}"
                if fp not in seen_fp:
                    final.append({**m, "rerank_score": 0.0, "rrf_score": 0.0,
                                  "semantic_score": 0.0})
                    seen_fp.add(fp)

        return {
            "chunks":         final,
            "n_bm25":         n_bm25,
            "n_semantic":     n_semantic,
            "aggregation":    aggregation,
            "low_confidence": low_confidence,
            "top_rerank_score": (final[0].get("rerank_score", 0.0) if final else 0.0),
        }

    # ── Aggregation path (counting / listing questions) ───────────────────

    def _maybe_aggregate(self, query: str) -> Optional[dict]:
        """
        Counting/listing path. If `query` is an aggregation question, scan BM25
        for every chunk containing the filter term and *also* count how many
        times the filter term appears inside those chunks — because the chunker
        packs many spreadsheet rows into one chunk, chunk count under-counts
        rows. Counting occurrences of the term inside chunk text recovers the
        true row count when each row labels its severity (e.g. xlsx rows
        emitted as `Severity: Critical | ...`).

        Returns:
          {
            "filter_term":          str,         # e.g. "critical"
            "matching_chunk_count": int,         # # chunks containing the term
            "occurrence_count":     int,         # # term occurrences inside those chunks
            "by_source":            {src: int},  # occurrences per source document
            "matches":              list[dict],  # ALL matching chunks (capped at 200)
          }
        """
        if not _COUNT_QUERY_PATTERNS.search(query):
            return None

        q_lower = query.lower()
        filter_term = next((t for t in _COMMON_FILTER_TERMS if t in q_lower), None)
        if not filter_term:
            return None

        matches = self.bm25.scan_matching(filter_term, max_rows=1000)
        if not matches:
            return None

        # Count word-bounded occurrences of filter_term inside each chunk.
        # `re.escape` because some terms contain spaces (e.g. "priority 1").
        pattern = re.compile(rf"\b{re.escape(filter_term)}\b", re.IGNORECASE)

        occurrence_count = 0
        by_source: dict[str, int] = {}
        for m in matches:
            n = len(pattern.findall(m.get("snippet") or m.get("text") or ""))
            occurrence_count += n
            by_source[m["source"]] = by_source.get(m["source"], 0) + n

        return {
            "filter_term":          filter_term,
            "matching_chunk_count": len(matches),
            "occurrence_count":     occurrence_count,
            "by_source":            by_source,
            "matches":              matches[:200],
        }

    def _semantic_search(self, query: str, k: int) -> list[dict]:
        """Query ChromaDB and normalise results."""
        try:
            results = self.collection.query(query_texts=[query], n_results=k)
        except Exception as e:
            logger.error(f"Chroma semantic search failed: {e}")
            return []

        out = []
        if results and results.get("documents") and results["documents"]:
            docs      = results["documents"][0]
            metas     = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0]  if results.get("distances")  else [1.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                meta = meta or {}
                parent = meta.get("parent_text", "")
                # Convert L2 distance → similarity score (0-1, higher = better)
                semantic_score = max(0.0, 1.0 - dist / 2.0)
                out.append({
                    "text":           parent if parent else doc,
                    "snippet":        doc,
                    "source":         meta.get("source",  "Unknown"),
                    "page":           meta.get("page",    1),
                    "section":        meta.get("section", ""),
                    "semantic_score": round(semantic_score, 4),
                    "bm25_score":     0.0,
                    "rerank_score":   0.0,
                })
        return out

    @staticmethod
    def _rrf_merge(
        semantic: list[dict],
        bm25: list[dict],
        k: int = _RRF_K,
    ) -> tuple[list[dict], int, int]:
        """
        Reciprocal Rank Fusion.
        Deduplicates by (source, page, snippet[:80]) fingerprint.
        Returns (merged_list, n_from_bm25, n_from_semantic).
        """

        def _fp(c: dict) -> str:
            return f"{c['source']}|{c['page']}|{c['snippet'][:80]}"

        scores: dict[str, float]  = {}
        chunks: dict[str, dict]   = {}
        origins: dict[str, set]   = {}

        for rank, c in enumerate(semantic):
            fp = _fp(c)
            scores[fp]  = scores.get(fp, 0.0) + 1.0 / (k + rank + 1)
            chunks[fp]  = c
            origins.setdefault(fp, set()).add("semantic")

        for rank, c in enumerate(bm25):
            fp = _fp(c)
            scores[fp]  = scores.get(fp, 0.0) + 1.0 / (k + rank + 1)
            if fp not in chunks:
                chunks[fp] = c
            else:
                # If BM25 found same chunk, keep higher bm25_score
                chunks[fp]["bm25_score"] = max(
                    chunks[fp].get("bm25_score", 0.0), c.get("bm25_score", 0.0)
                )
            origins.setdefault(fp, set()).add("bm25")

        # Attach RRF score, sort
        merged = []
        for fp, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            c = {**chunks[fp], "rrf_score": round(score, 6)}
            merged.append(c)

        n_bm25     = sum(1 for fp in origins if "bm25"     in origins[fp])
        n_semantic = sum(1 for fp in origins if "semantic" in origins[fp])

        return merged, n_bm25, n_semantic

    # ── Generation ────────────────────────────────────────────────────────

    def generate_response_stream(
        self,
        model_id: str,
        query: str,
        contexts: list[dict],
        memory_context: dict | None = None,
        aggregation: dict | None = None,
    ):
        """
        Streams response from Bedrock using retrieved chunks + compressed memory.
        contexts: list of chunk dicts from query_context()["chunks"]
        aggregation: optional {"filter_term", "occurrence_count", "by_source", ...}
                     dict produced by query_context() for counting/listing
                     questions. When present, the LLM is told the exact count
                     so it doesn't guess from chunked context.
        """
        context_str = ""
        for ctx in contexts:
            section_label = f" [{ctx['section']}]" if ctx.get("section") else ""
            context_str += (
                f"\n--- Source: {ctx['source']}{section_label} (Page {ctx['page']}) ---\n"
                f"{ctx['text']}\n"
            )

        # Inject the aggregation summary as an authoritative count the LLM
        # MUST use, since chunked retrieval can't reliably show every row.
        if aggregation:
            by_source_str = ", ".join(
                f"{src}: {n}" for src, n in aggregation["by_source"].items()
            )
            context_str = (
                f"\n--- AGGREGATION SUMMARY (authoritative count) ---\n"
                f"Filter term: '{aggregation['filter_term']}'\n"
                f"Total occurrences across all indexed documents: {aggregation['occurrence_count']}\n"
                f"Breakdown by document: {by_source_str}\n"
                f"Distinct chunks containing the term: {aggregation['matching_chunk_count']}\n"
                f"--- END AGGREGATION SUMMARY ---\n"
            ) + context_str

        base_system = (
            "You are a helpful AI assistant answering questions based on the provided document context.\n"
            "Instructions:\n"
            "- Answer the user's question using ONLY the provided context.\n"
            "- If the context does not contain the answer, state that you cannot find the answer in the provided documents.\n"
            "- Be concise, direct, and factually accurate. Do not hallucinate details.\n"
            "- Cite your sources (document name and page number) when answering.\n"
            "- For counting/listing questions, when an AGGREGATION SUMMARY block is present, use its "
            "totals as the authoritative answer. Do NOT recount from the individual chunks below — "
            "they are only a sample, not the complete set."
        )
        summary = (memory_context or {}).get("summary", "")
        system_prompt = base_system + (f"\n\nConversation summary so far:\n{summary}" if summary else "")

        recent_messages = ((memory_context or {}).get("recent_messages") or [])[-5:]
        user_prompt = f"Context details:\n{context_str}\n\nUser Question: {query}"

        # Optional Bedrock Guardrail config — set BEDROCK_GUARDRAIL_ID and
        # BEDROCK_GUARDRAIL_VERSION env vars to enable. Applied at the
        # InvokeModel call so input AND output are filtered by the guardrail.
        guardrail_id      = os.getenv("BEDROCK_GUARDRAIL_ID", "").strip()
        guardrail_version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip()
        guardrail_kwargs  = {}
        if guardrail_id:
            guardrail_kwargs["guardrailIdentifier"] = guardrail_id
            guardrail_kwargs["guardrailVersion"]    = guardrail_version

        try:
            if "anthropic.claude" in model_id:
                messages = _sanitise_messages_for_claude(
                    list(recent_messages) + [{"role": "user", "content": user_prompt}]
                )
                if not messages or messages[-1]["role"] != "user":
                    messages.append({"role": "user", "content": user_prompt})

                # Anthropic prompt caching: split the system prompt into a
                # cacheable static part (instructions, ~500 tokens — same on
                # every request) and a dynamic part (conversation summary).
                # The static block is marked with cache_control so Bedrock
                # serves it from cache on subsequent turns. Saves cost +
                # latency on repeated chat without changing semantics.
                system_blocks = [
                    {
                        "type": "text",
                        "text": base_system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if summary:
                    system_blocks.append({
                        "type": "text",
                        "text": f"Conversation summary so far:\n{summary}",
                    })

                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "system": system_blocks,
                    "messages": messages,
                    "temperature": 0.2,
                })
                response = self.bedrock_client.invoke_model_with_response_stream(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                    **guardrail_kwargs,
                )
                for event in response.get("body"):
                    chunk = json.loads(event.get("chunk").get("bytes").decode("utf-8"))
                    if chunk.get("type") == "content_block_delta":
                        yield chunk.get("delta", {}).get("text", "")

            elif "amazon.titan" in model_id:
                prompt = f"{system_prompt}\n\n"
                for m in recent_messages:
                    prompt += f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content']}\n"
                prompt += f"User: {user_prompt}\nBot:"

                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {"maxTokenCount": 512, "temperature": 0.2, "topP": 0.9},
                })
                response = self.bedrock_client.invoke_model_with_response_stream(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                    **guardrail_kwargs,
                )
                for event in response.get("body"):
                    chunk = json.loads(event.get("chunk").get("bytes").decode("utf-8"))
                    yield chunk.get("outputText", "")

            else:
                yield f"Error: Unsupported Bedrock model ID: {model_id}"

        except Exception as e:
            err_str = str(e)
            logger.error(f"Bedrock stream error: {err_str}")
            # If prompt caching isn't supported on the user's Bedrock account,
            # the API returns a validation error mentioning cache_control.
            # Auto-retry once without it so we degrade gracefully.
            if "cache_control" in err_str and "anthropic.claude" in model_id:
                logger.warning("Retrying without prompt caching.")
                try:
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "system": system_prompt,
                        "messages": messages,
                        "temperature": 0.2,
                    })
                    response = self.bedrock_client.invoke_model_with_response_stream(
                        modelId=model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                        **guardrail_kwargs,
                    )
                    for event in response.get("body"):
                        chunk = json.loads(event.get("chunk").get("bytes").decode("utf-8"))
                        if chunk.get("type") == "content_block_delta":
                            yield chunk.get("delta", {}).get("text", "")
                    return
                except Exception as retry_err:
                    err_str = str(retry_err)

            yield (
                f"An error occurred while calling the AWS Bedrock LLM: {err_str}\n\n"
                f"Ensure credentials are valid and model '{model_id}' is enabled in AWS Bedrock."
            )

    # ── Citation verification ─────────────────────────────────────────────

    @staticmethod
    def verify_citations(answer: str, contexts: list[dict]) -> dict:
        """
        Extract `(source, page)` citations from `answer` and check each one
        was actually retrieved. The model is asked to cite, but doesn't always
        cite a real chunk — this catches that.

        Patterns matched (case-insensitive):
          - "filename.pdf, page 3"
          - "filename.pdf (page 3)"
          - "filename.pdf p. 3"
          - "page 3 of filename.pdf"
          - "[filename.pdf:3]"

        Returns:
          {
            "extracted":     [(source, page), ...],
            "verified":      [(source, page), ...],
            "unverified":    [(source, page), ...],
            "verification_rate": float in [0,1],
          }
        """
        retrieved = {(c.get("source", "").lower(), int(c.get("page", 0)))
                     for c in (contexts or [])}

        # Filename pattern: a single token that ends with one of our supported
        # extensions. We disallow whitespace inside the filename so prose
        # leading up to the citation isn't swallowed into the match.
        ext  = r"(?:pdf|png|jpe?g|webp|tiff|bmp|xlsx|xlsm|txt)"
        fn   = rf"[A-Za-z0-9_\-./]+\.{ext}"
        patterns = [
            rf"({fn})\s*[\(,]\s*page[s]?\s*(\d+)",   # foo.pdf, page 3  /  foo.pdf (page 3)
            rf"({fn})\s*p\.?\s*(\d+)",                # foo.pdf p.3  /  foo.pdf p3
            rf"page[s]?\s*(\d+)\s*of\s*({fn})",       # page 3 of foo.pdf
            rf"\[({fn})\s*:\s*(\d+)\]",               # [foo.pdf:3]
        ]

        extracted: list[tuple[str, int]] = []
        for pat in patterns:
            for m in re.finditer(pat, answer, flags=re.IGNORECASE):
                groups = m.groups()
                if groups[0].isdigit():
                    page, source = int(groups[0]), groups[1]
                else:
                    source, page = groups[0], int(groups[1])
                extracted.append((source.strip().lower(), page))

        # Dedup
        seen = set()
        deduped: list[tuple[str, int]] = []
        for c in extracted:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        verified   = [c for c in deduped if c in retrieved]
        unverified = [c for c in deduped if c not in retrieved]
        rate = (len(verified) / len(deduped)) if deduped else 1.0

        return {
            "extracted":         deduped,
            "verified":          verified,
            "unverified":        unverified,
            "verification_rate": round(rate, 3),
        }
