# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Name

**AetherRAG** — Advanced Document Intelligence. Streamlit UI branding throughout `app.py`.

## Running the App

```bash
docker compose up -d        # OpenSearch on localhost:9200 (BM25 backend)
streamlit run app.py
```

Dependencies beyond pip require system packages:
- **Tesseract OCR**: `brew install tesseract` (macOS) — required by `pytesseract`
- **Poppler**: `brew install poppler` (macOS) — required by `pdf2image`
- **Docker**: required to run the OpenSearch container (`docker-compose.yml` at repo root)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Key pip packages: `streamlit`, `chromadb`, `boto3`, `langgraph`, `langgraph-checkpoint-sqlite`, `opensearch-py`, `sentence-transformers`, `langsmith`, `pdfplumber`, `pdf2image`, `pytesseract`, `python-dotenv`.

## Environment

AWS credentials are loaded from `.env` at startup via `python-dotenv`. The app also accepts credentials entered directly in the sidebar UI (which override env vars for that session). Required variables:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN   # required for temporary/session credentials (AWS Academy, SSO)
AWS_DEFAULT_REGION
```

Optional LangSmith tracing variables (can also be set via sidebar):
```
LANGCHAIN_API_KEY      # also accepted as LANGSMITH_API_KEY
LANGCHAIN_PROJECT      # defaults to "aetherrag"
LANGCHAIN_TRACING_V2   # set to "true" to enable
```

OpenSearch connection:
```
OPENSEARCH_URL         # defaults to http://localhost:9200
```

## Persistent State Files

- **`./memory.db`** — SQLite database for `MemoryManager` (threads, chat_history, LangGraph checkpoints).
- **`./chroma_db/`** — ChromaDB persistent storage for dense vector embeddings (collection name: `rag_documents`).
- **OpenSearch index `aetherrag_bm25`** — sparse BM25 index, persisted in the `opensearch-data` Docker volume.

All three persist across app restarts. Deleting them resets history, vectors, and the BM25 index respectively.

## Architecture

The app is split into three layers:

**`app.py`** — Streamlit UI only. Page title "AetherRAG - Advanced Document Intelligence". Manages session state (`processed_files`, `messages`, AWS credential fields, LangSmith config, `evaluator`), sidebar configuration (thread management, LangSmith expander, AWS credentials expander, model selection, chunking strategy, retrieval controls, file uploader), and the chat loop. Instantiates `RagEngine` directly (not via `st.cache_resource`) — credential changes call `st.cache_resource.clear()` then re-initialize. Streams LLM responses token-by-token using a generator from `rag_engine`. Available LLM models:
  - `Claude 3 Haiku` → `anthropic.claude-3-haiku-20240307-v1:0`
  - `Claude 3.5 Sonnet` → `anthropic.claude-3-5-sonnet-20240620-v1:0`
  - `Titan Text Express` → `amazon.titan-text-express-v1`

After each assistant response, the UI renders a "Retrieval Evaluation" expander with hit_rate, MRR, NDCG, BM25 mix columns, plus optional faithfulness/relevance/context_utilisation columns when LLM-as-Judge is toggled on. Source citation cards display page badge + rerank score (red) + RRF score (green) + BM25 score (yellow) badges.

**`bm25_index.py`** — `BM25Index`. Persistent BM25 sparse index backed by **OpenSearch** (index name: `aetherrag_bm25`, default URL `http://localhost:9200`, configurable via `OPENSEARCH_URL`). Mapping uses the `english` analyzer (stemming + stop-words) on the `text` field; OpenSearch's default similarity is BM25 so `_score` is the BM25 score directly. Document fields: `text` (analyzed), `source` (keyword), `page` (int), `section` (keyword), `parent_text` (text, not indexed), `chunk_index` (int). `add_documents()` uses the bulk API with `op_type=create` for idempotent re-ingest (per-doc 409s are filtered out). `remove_source(source)` issues a `delete_by_query` on the `source` term. `clear()` deletes and recreates the index. Returns `{text, snippet, source, page, section, bm25_score}` (uses `parent_text` as `text` when present, raw chunk as `snippet`).

**`reranker.py`** — `CrossEncoderReranker`. Wraps `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, CPU-only). Model loaded once via `@lru_cache(maxsize=1)`. Scores all (query, passage) pairs in a single `model.predict()` call, sorts descending, attaches `rerank_score`. Supports optional `score_threshold` to discard low-score results. Gracefully degrades to unranked on model failure.

**`evaluator.py`** — `RAGEvaluator`. Deterministic retrieval metrics (hit_rate, MRR, NDCG, bm25_contribution, rerank_score_top1) computed without Bedrock. Optional LLM-as-judge generation metrics (faithfulness, answer_relevance, context_utilisation) call Bedrock non-streaming — guarded by `run_generation_eval` flag. LLM judge scales 0–5 responses to 0–1. `session_summary()` returns averages across all turns in `self.history`. `log_metrics_to_langsmith()` posts scores as feedback on a LangSmith run. Module also exposes a `rag_trace()` context manager (unused by app.py currently). LangSmith client initialised lazily; accepts `LANGCHAIN_API_KEY` or `LANGSMITH_API_KEY`.

**`chunker.py`** — `DocumentChunker` class. Three strategies:
  - `semantic` (default) — sentence-aware splits with abbreviation-aware splitter; sentence-level overlap carry-back
  - `hierarchical` — small retrieval child chunks + sliding parent window of `parent_multiplier=3` children; child is retrieval target, parent is LLM context
  - `fixed` — recursive character split on `["\n\n", "\n", " ", ""]`

Per-strategy defaults (override with constructor args): semantic `chunk_size=800, overlap=150, min=80`; hierarchical `400/80/60`; fixed `800/150/50`. All strategies clean text (Unicode NFC, control char removal, boilerplate stripping via regex, collapse 3+ newlines), detect section headers (markdown `##`, ALL CAPS, `1.2.3` numbered, `Title:` forms), enforce `min_chunk_size`, and never cross page boundaries. Returns uniform `[{text, snippet, page, section, strategy, chunk_index, parent_text}]`.

**`rag_engine.py`** — `RagEngine`. Owns ChromaDB (dense, path `./chroma_db`, collection `rag_documents`), `BM25Index` (sparse, OpenSearch-backed, URL from `OPENSEARCH_URL` env var or constructor arg), and `CrossEncoderReranker`. `add_document()` indexes into both ChromaDB and BM25 using the same chunk IDs. `query_context(query, n_results, fetch_k=20, use_reranker)` runs the full hybrid pipeline: semantic search (ChromaDB top-20, L2 distance converted to similarity via `1.0 - dist/2.0`) → BM25 search (top-20) → RRF merge (`k=60`, dedup by `source|page|snippet[:80]`) → cross-encoder rerank → top-K. Returns `{"chunks": [...], "n_bm25": int, "n_semantic": int}`. Each chunk carries `semantic_score`, `bm25_score`, `rrf_score`, `rerank_score`. `reset_collection()` clears both ChromaDB and BM25. `generate_response_stream()` builds a system prompt from the RAG context + memory summary, passes the last 5 `recent_messages` as conversation history, uses `max_tokens=1024` / `temperature=0.2` for Claude and `maxTokenCount=512` for Titan. Calls `_sanitise_messages_for_claude()` before building the Claude messages list to enforce strict user/assistant alternation (drops leading assistant messages, merges consecutive same-role messages).

**`ocr_engine.py`** — Document text extraction routing. For PDFs: tries native extraction via `pdfplumber` first; falls back to `pdf2image` + `pytesseract` OCR if total extracted text is under 100 characters (scanned/image-based PDFs). For images (PNG/JPG/JPEG/WEBP/TIFF/BMP): runs Tesseract OCR directly. Unrecognised extensions fall back to UTF-8 decode. Returns `[{"page": int, "text": str}, ...]` from all paths. File uploader in app.py restricts to `["pdf", "png", "jpg", "jpeg", "webp"]`.

**`memory_manager.py`** — LangGraph + SQLite persistent memory. One `MemoryManager` singleton (via `@st.cache_resource` on `get_memory_manager()`) owns a single `sqlite3.Connection` to `./memory.db` shared by:
- **LangGraph `SqliteSaver` checkpointer** — stores compressed thread state (`checkpoints` + `writes` tables).
- **Custom `threads` table** — thread metadata (id, name, created_at, last_updated).
- **Custom `chat_history` table** — full uncompressed display history with `sources` JSON column (never trimmed).

LangGraph graph has two nodes: `append_node` merges `pending` into `messages` → conditional `compress_node` fires when `len(messages) > 7` (RECENT_WINDOW=5 + COMPRESS_AT=2), calls Bedrock non-streaming to summarize old messages and trims to the 5 most recent. The summary is appended to the LLM system prompt on the next turn. `messages` is a plain `List[dict]` (no `Annotated` append reducer) so `compress_node`'s output replaces rather than appends. `delete_thread()` cleans up all four tables (threads, chat_history, checkpoints, writes).

## Key Design Details

- **Embeddings model is fixed** to `amazon.titan-embed-text-v1` regardless of which LLM model the user selects. The LLM model selection only affects generation.
- **ChromaDB collection is never migrated** — if the embedding model changes, the collection must be reset (`rag_engine.reset_collection()`), or documents re-ingested.
- **Recent messages hard cap is 5** (RECENT_WINDOW). Older messages are compressed into a rolling summary rather than discarded; full history always available in `chat_history` table for display.
- **Memory context capture order**: `get_memory_context()` is called *before* `add_message(user)` so the LLM sees prior turns only — mirrors the old `chat_history[:-1]` behaviour.
- **Thread auto-rename**: first user message in a thread becomes the thread name (truncated to 40 chars with `...`) — fires inside `add_message` when `total_count` transitions to 1.
- **Chunking strategy** is per-ingest — different files in the same collection can use different strategies. The strategy is stored as metadata per chunk (`strategy` field) but ChromaDB has no schema enforcement so mixed collections are allowed.
- **Chunk IDs** include a UUID suffix (`{filename}_p{page}_c{chunk_idx}_{uuid8}`) to allow the same file to be re-ingested without ID collisions (though duplicate content will accumulate — there is no deduplication guard).
- **`st.cache_resource.clear()`** in the credentials expander also resets the `MemoryManager` singleton, but the `memory.db` file persists on disk so no data is lost.
- **BM25 storage**: lives in OpenSearch, not in `memory.db`. The OpenSearch container must be running (`docker compose up -d`) before launching the app, otherwise `RagEngine.__init__` will fail at `BM25Index._ensure_index`.
- **Claude message sanitisation**: `_sanitise_messages_for_claude()` in `rag_engine.py` enforces the strict Claude user/assistant alternation rule on compressed LangGraph history before every generation call.
