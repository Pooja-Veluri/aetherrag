"""
Semantic answer cache.

Stores past Q&A turns keyed by query embedding. On each new query, embeds it
once and looks up the nearest cached entry by cosine similarity. If similarity
≥ threshold, the cached answer + retrieved chunks are returned and the
expensive RAG path (BM25 + dense search + RRF + cross-encoder + Bedrock LLM)
is skipped.

Persistence: SQLite (`./semantic_cache.db`) so the cache survives app restarts.
Embeddings stored as raw float32 bytes; the table is small enough that a
linear scan is faster than maintaining an ANN index.
"""

import io
import json
import logging
import os
import sqlite3
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("SEMANTIC_CACHE_DB", "./semantic_cache.db")

# Cosine similarity at-or-above which we consider a cache hit. 0.93 is high
# enough to avoid bleed between paraphrases of *different* questions while
# still catching trivial reformulations ("how many critical?" vs "count of
# critical incidents"). Tunable via SEMANTIC_CACHE_THRESHOLD env var.
DEFAULT_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.93"))

# Hard cap on rows scanned per lookup. Cache is pruned to this size on insert.
MAX_ROWS = int(os.environ.get("SEMANTIC_CACHE_MAX_ROWS", "5000"))

# Entries older than this many seconds are ignored on lookup. 0 = never expire.
TTL_SECONDS = int(os.environ.get("SEMANTIC_CACHE_TTL_SECONDS", "0"))


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class SemanticCache:
    """
    Persistent semantic cache for RAG answers.

    Public API:
        cache = SemanticCache(embed_fn)           # embed_fn(text)->list[float]
        hit = cache.lookup(query, scope=scope)    # dict | None
        cache.put(query, answer, contexts, scope=scope, extra={...})
        cache.clear()
        stats = cache.stats()
    """

    def __init__(
        self,
        embed_fn,
        db_path: str = DB_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        max_rows: int = MAX_ROWS,
        ttl_seconds: int = TTL_SECONDS,
    ):
        self.embed_fn    = embed_fn
        self.threshold   = threshold
        self.max_rows    = max_rows
        self.ttl_seconds = ttl_seconds
        self.conn        = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

        # Counters for the current process — never persisted.
        self._hits   = 0
        self._misses = 0

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                scope     TEXT    NOT NULL,
                query     TEXT    NOT NULL,
                embedding BLOB    NOT NULL,
                answer    TEXT    NOT NULL,
                contexts  TEXT,
                extra     TEXT,
                created_at REAL   NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_scope ON semantic_cache(scope);
            """
        )
        self.conn.commit()

    # ── Lookup ────────────────────────────────────────────────────────────

    def lookup(self, query: str, scope: str = "default") -> Optional[dict]:
        """
        Return the cached entry whose stored query embedding is closest to
        `query` and exceeds the similarity threshold. None on miss.

        Returns: {
            "query":      original cached query,
            "answer":     cached answer text,
            "contexts":   list[dict] of cached retrieval chunks (may be []),
            "extra":      whatever the caller stored,
            "similarity": cosine score in [0,1],
            "age_seconds": int,
        }
        """
        if not query or not query.strip():
            return None
        try:
            q_emb = self.embed_fn(query)
        except Exception as e:
            logger.warning(f"Semantic cache embed failed; treating as miss: {e}")
            self._misses += 1
            return None

        cur = self.conn.execute(
            "SELECT id, query, embedding, answer, contexts, extra, created_at "
            "FROM semantic_cache WHERE scope = ? ORDER BY id DESC LIMIT ?",
            (scope, self.max_rows),
        )

        now = time.time()
        best = None
        best_sim = self.threshold

        for row in cur.fetchall():
            _id, q_text, emb_blob, answer, ctx_json, extra_json, created_at = row
            if self.ttl_seconds and (now - created_at) > self.ttl_seconds:
                continue
            sim = _cosine(q_emb, _unpack(emb_blob))
            if sim >= best_sim:
                best_sim = sim
                best = (q_text, answer, ctx_json, extra_json, created_at, sim)

        if best is None:
            self._misses += 1
            return None

        q_text, answer, ctx_json, extra_json, created_at, sim = best
        self._hits += 1
        return {
            "query":       q_text,
            "answer":      answer,
            "contexts":    json.loads(ctx_json) if ctx_json else [],
            "extra":       json.loads(extra_json) if extra_json else {},
            "similarity":  round(sim, 4),
            "age_seconds": int(now - created_at),
        }

    # ── Insert ────────────────────────────────────────────────────────────

    def put(
        self,
        query: str,
        answer: str,
        contexts: Optional[list[dict]] = None,
        scope: str = "default",
        extra: Optional[dict] = None,
    ):
        """Store a new entry. Best-effort — exceptions are logged, not raised."""
        if not query.strip() or not answer.strip():
            return
        try:
            q_emb = self.embed_fn(query)
        except Exception as e:
            logger.warning(f"Semantic cache put failed (embed): {e}")
            return

        try:
            self.conn.execute(
                "INSERT INTO semantic_cache (scope, query, embedding, answer, contexts, extra, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    scope,
                    query,
                    _pack(q_emb),
                    answer,
                    json.dumps(contexts or []),
                    json.dumps(extra or {}),
                    time.time(),
                ),
            )
            # Prune oldest rows in this scope to keep the table bounded.
            self.conn.execute(
                "DELETE FROM semantic_cache WHERE id IN ("
                "  SELECT id FROM semantic_cache WHERE scope = ? "
                "  ORDER BY id DESC LIMIT -1 OFFSET ?"
                ")",
                (scope, self.max_rows),
            )
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Semantic cache put failed (sqlite): {e}")

    # ── Maintenance ───────────────────────────────────────────────────────

    def clear(self, scope: Optional[str] = None):
        if scope:
            self.conn.execute("DELETE FROM semantic_cache WHERE scope = ?", (scope,))
        else:
            self.conn.execute("DELETE FROM semantic_cache")
        self.conn.commit()

    def stats(self) -> dict:
        cur = self.conn.execute("SELECT COUNT(*) FROM semantic_cache")
        total = cur.fetchone()[0]
        return {
            "rows":       total,
            "hits":       self._hits,
            "misses":     self._misses,
            "threshold":  self.threshold,
            "ttl_seconds": self.ttl_seconds,
        }
