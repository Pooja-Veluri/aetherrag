"""
BM25 sparse retrieval index backed by OpenSearch.

The index is named `aetherrag_bm25` and uses the English analyzer for
tokenisation, stemming, and stop-word removal. OpenSearch's default
similarity is BM25, so the `_score` returned on a `match` query is the
BM25 score directly.

Document mapping:
  text         (text, english analyzer) — primary search field
  source       (keyword)                — for delete-by-source
  page         (integer)
  section      (keyword)
  parent_text  (text, not indexed)      — returned to LLM context
  chunk_index  (integer)
"""

import logging
from typing import Optional

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

logger = logging.getLogger(__name__)

INDEX_NAME = "aetherrag_bm25"

_INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "english",
                }
            }
        },
        "similarity": {
            "default": {"type": "BM25"},
        },
    },
    "mappings": {
        "properties": {
            "text":        {"type": "text",    "analyzer": "english"},
            "source":      {"type": "keyword"},
            "page":        {"type": "integer"},
            "section":     {"type": "keyword"},
            "parent_text": {"type": "text",    "index": False},
            "chunk_index": {"type": "integer"},
        }
    },
}


class BM25Index:
    """
    Persistent BM25 index backed by OpenSearch.

    Public API matches the previous SQLite-backed implementation so callers
    in `rag_engine.py` need no changes:
      - add_documents(chunks, source)
      - remove_source(source)
      - clear()
      - query(query_text, n_results)
      - doc_count()
    """

    def __init__(self, opensearch_url: str = "http://localhost:9200"):
        self.client = OpenSearch(
            hosts=[opensearch_url],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        self._ensure_index()

    # ── Schema ────────────────────────────────────────────────────────────

    def _ensure_index(self):
        if not self.client.indices.exists(index=INDEX_NAME):
            self.client.indices.create(index=INDEX_NAME, body=_INDEX_BODY)
            logger.info(f"Created OpenSearch index '{INDEX_NAME}'.")

    # ── Write ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict], source: str):
        """
        Bulk-index chunks. `op_type=create` makes the call idempotent: an
        existing _id raises a per-doc 409 which `bulk` silently skips when
        `raise_on_error=False`.
        """
        if not chunks:
            return

        actions = [
            {
                "_op_type": "create",
                "_index":   INDEX_NAME,
                "_id":      c["id"],
                "_source": {
                    "text":        c["text"],
                    "source":      source,
                    "page":        c.get("page", 1),
                    "section":     c.get("section", ""),
                    "parent_text": c.get("parent_text", ""),
                    "chunk_index": c.get("chunk_index", 0),
                },
            }
            for c in chunks
        ]

        success, errors = bulk(self.client, actions, raise_on_error=False, refresh=True)
        # `errors` is a list of per-doc failure dicts. Filter version-conflict
        # (409) errors which mean "already indexed" — those are expected.
        real_failures = [
            e for e in errors
            if not (isinstance(e, dict)
                    and (e.get("create") or {}).get("status") == 409)
        ]
        if real_failures:
            logger.warning(f"BM25 bulk: {len(real_failures)} docs failed: {real_failures[:3]}")
        logger.info(f"BM25: indexed {success} chunks for '{source}'.")

    def remove_source(self, source: str):
        self.client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"term": {"source": source}}},
            refresh=True,
        )

    def clear(self):
        if self.client.indices.exists(index=INDEX_NAME):
            self.client.indices.delete(index=INDEX_NAME)
        self._ensure_index()

    # ── Query ─────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 20) -> list[dict]:
        """
        Return top-n chunks ranked by BM25 score (`_score`).
        Result shape matches the prior SQLite implementation:
          {text, snippet, source, page, section, bm25_score}
        """
        if not query_text.strip():
            return []

        try:
            response = self.client.search(
                index=INDEX_NAME,
                body={
                    "size":   n_results,
                    "query":  {"match": {"text": query_text}},
                    "_source": ["text", "source", "page", "section", "parent_text"],
                },
            )
        except Exception as e:
            logger.error(f"OpenSearch query failed: {e}")
            return []

        results = []
        for hit in response.get("hits", {}).get("hits", []):
            src    = hit["_source"]
            parent = src.get("parent_text", "")
            results.append({
                "text":       parent if parent else src["text"],
                "snippet":    src["text"],
                "source":     src["source"],
                "page":       src["page"],
                "section":    src.get("section", ""),
                "bm25_score": float(hit["_score"]),
            })
        return results

    def scan_matching(
        self,
        phrase: str,
        max_rows: int = 1000,
        source: Optional[str] = None,
    ) -> list[dict]:
        """
        Return EVERY chunk whose text contains `phrase` (case-insensitive,
        analyzer-stemmed) — not the top-K. Used by the aggregation path so
        counting/listing questions see the full matching set instead of a
        relevance-ranked slice. Optionally filter to a single source.
        """
        if not phrase.strip():
            return []

        must = [{"match_phrase": {"text": phrase}}]
        filt = [{"term": {"source": source}}] if source else []

        try:
            response = self.client.search(
                index=INDEX_NAME,
                body={
                    "size":   max_rows,
                    "query":  {"bool": {"must": must, "filter": filt}},
                    "_source": ["text", "source", "page", "section", "parent_text"],
                },
            )
        except Exception as e:
            logger.error(f"OpenSearch scan_matching failed: {e}")
            return []

        results = []
        for hit in response.get("hits", {}).get("hits", []):
            src    = hit["_source"]
            parent = src.get("parent_text", "")
            results.append({
                "text":       parent if parent else src["text"],
                "snippet":    src["text"],
                "source":     src["source"],
                "page":       src["page"],
                "section":    src.get("section", ""),
                "bm25_score": float(hit["_score"]),
            })
        return results

    # ── Introspection ─────────────────────────────────────────────────────

    def doc_count(self) -> int:
        try:
            return int(self.client.count(index=INDEX_NAME)["count"])
        except Exception:
            return 0
