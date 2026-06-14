# AetherRAG

Document Q&A over PDFs, images, and spreadsheets. A production-shaped Retrieval-Augmented Generation pipeline built on AWS Bedrock with hybrid retrieval, cross-encoder reranking, MMR diversity, semantic answer caching, persistent multi-thread memory, groundedness-based abstention, citation verification, and full LangSmith evaluation.

## What it does

Upload a document, ask a question, get an answer grounded in the document with cited sources. Under the hood the system runs a multi-stage retrieval pipeline so answers stay accurate even when:

- the question paraphrases the document (dense retrieval handles synonyms)
- the question contains rare tokens like ticket IDs or severity labels (BM25 handles literal matches)
- the question is a follow-up to an earlier turn (conversational query rewriting)
- the question asks for a count or list (aggregation path bypasses top-K retrieval)
- nothing in the corpus is relevant (groundedness gate refuses instead of hallucinating)

## Architecture

```
                                    ┌───────────────────┐
  user query ──► query rewriter ──► │  semantic cache   │ ── hit ─► cached answer
                 (Claude Haiku)     │  (cosine ≥ 0.93)  │
                                    └─────────┬─────────┘
                                              │ miss
                                              ▼
                  ┌───────────────────┐               ┌───────────────────┐
                  │   ChromaDB        │               │   OpenSearch      │
                  │   (Titan dense)   │               │   (BM25 sparse)   │
                  └─────────┬─────────┘               └─────────┬─────────┘
                            └────────────┬────────────────────┘
                                         ▼
                              Reciprocal Rank Fusion
                                         │
                                         ▼
                          Cross-encoder rerank (top 3K)
                          ms-marco-MiniLM-L-6-v2
                                         │
                                         ▼
                              MMR diversity (top K)
                                         │
                                         ▼
                              Groundedness gate
                                         │
                                         ▼
                       ┌────────────────────────────────┐
                       │  Bedrock generation            │
                       │  (Claude with prompt caching   │
                       │   or Titan, streaming)         │
                       │  + optional Bedrock Guardrails │
                       └────────────────┬───────────────┘
                                        ▼
                        Citation verification + LangSmith eval
```

Memory is layered: a hot window of the last 5 messages plus a rolling LLM-generated summary of older turns, persisted in SQLite via LangGraph's `SqliteSaver`. Each thread is isolated, auto-named from the first user message, and survives restart.

## Stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Single-file Python UI, no front-end build step |
| Embeddings | AWS Bedrock `amazon.titan-embed-text-v1` | Reused for queries, chunks, and the semantic cache |
| Generation | AWS Bedrock — Claude 3 Haiku, Claude 3.5 Sonnet, Titan Text Express | Selectable per session; embeddings are fixed |
| Dense retrieval | ChromaDB (persistent, local) | Lightest vector store with no separate server |
| Sparse retrieval | OpenSearch (Docker, BM25 with English analyzer) | Stemming, stop-words, `match_phrase`, `delete_by_query` out of the box |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers | ~22M params, CPU-only, ~5ms per pair |
| Memory | LangGraph + SQLite | Compressed state + checkpointing, persistent across restarts |
| Tracing & eval | LangSmith | Per-turn `rag_turn` parent run with retrieval / generation / citation child spans |
| OCR | pdfplumber, Tesseract via pytesseract, openpyxl | Native PDF text first, OCR fallback for scanned pages, row-by-row spreadsheet extraction |

## Quick start

### Prerequisites

- Python 3.11+
- Docker (for OpenSearch)
- Tesseract OCR — `brew install tesseract`
- Poppler — `brew install poppler` (required by `pdf2image`)
- AWS Bedrock access with the listed Claude / Titan models enabled in your region

### Install

```bash
pip install -r requirements.txt
docker compose up -d                    # starts OpenSearch on :9200
cp .env.example .env                    # then fill in AWS credentials
streamlit run app.py
```

### Environment

All configuration lives in `.env`. See `.env.example` for the full list. The minimum to run is:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
OPENSEARCH_URL=http://localhost:9200
```

LangSmith tracing, Bedrock Guardrails, and semantic-cache tuning are all opt-in via additional env vars.

## Features

**Retrieval**
- Hybrid BM25 + dense with Reciprocal Rank Fusion (`k=60`)
- Cross-encoder reranking with score threshold and graceful fallback
- MMR diversity selection (default `λ=0.7`) to prevent near-duplicate chunks
- Aggregation short-circuit for counting / listing questions: full BM25 phrase scan with occurrence counting, exact totals injected into the LLM prompt
- Conversational query rewriting via Claude Haiku for follow-up questions
- Groundedness gate that refuses to answer when the top reranker score is below threshold

**Generation**
- Streaming Bedrock invocation
- Anthropic prompt caching on the static system prompt with auto-fallback for accounts that don't yet support it
- Strict user/assistant alternation enforced for Claude (drops empties, merges consecutive same-role messages)
- Optional Bedrock Guardrails via `BEDROCK_GUARDRAIL_ID` env var
- Post-hoc citation verification: every `(filename, page)` cited by the LLM is regex-extracted and checked against the retrieved chunks; unverified citations surfaced to user and to LangSmith

**Memory**
- Last 5 messages sent verbatim
- Rolling LLM-generated summary of older turns when the window grows past 7
- Full uncompressed display history kept for replay and audit
- Multi-thread with auto-rename from first user message
- All state persisted in `memory.db`

**Caching**
- Semantic answer cache: query embedded once with Titan, cosine-matched against past queries (default threshold `0.93`)
- Scoped on `(model, file set)` so hits can never bleed across corpora or models
- Auto-invalidated whenever the corpus changes

**Evaluation**
- Per-turn deterministic retrieval metrics: `hit_rate`, `mrr`, `ndcg`, `precision_at_k`, `chunk_diversity`, `bm25_contribution`, `rerank_score_top1`
- Per-turn LLM-as-judge generation metrics: `faithfulness`, `answer_relevance`, `answer_completeness`, `context_utilisation`
- Citation metrics: `citation_verification_rate`, extracted, unverified
- 25-question golden dataset with `exact_match` and `token_f1` batch evaluator
- Full LangSmith tracing with parent `rag_turn` and child `retrieval_eval`, `generation_eval`, `citation_eval` spans

## Design decisions

A few non-obvious choices, in case you're reading this for an interview:

- **Dense + sparse, not just dense.** Embeddings smooth out exact tokens like ticket IDs and severity labels. BM25 is the opposite. Hybrid + RRF combines them without needing a unified score scale, since RRF only uses ranks.
- **Embedding model is fixed; LLM is selectable.** Changing the embedding model would require re-embedding the whole corpus. Selecting between Haiku / Sonnet / Titan affects only generation, which is stateless.
- **Counting questions bypass top-K retrieval.** *How many critical incidents?* can never be answered from the top-5 chunks. Aggregation intent is detected with a regex, then a full BM25 phrase scan with occurrence counting feeds an authoritative `AGGREGATION SUMMARY` block into the prompt.
- **Refuse over hallucinate.** When the top reranker score falls below threshold and the question isn't an aggregation query, the system returns *I couldn't find anything in the indexed documents…* rather than guess.
- **Content-hash chunk IDs.** Ingesting the same file twice produces identical IDs, so re-ingest is idempotent at the index level.
- **Two layers of caching.** The cross-encoder model is `@lru_cache`'d so Streamlit reruns don't reload it. Answer caching is a separate concern keyed on query embedding.

## Tradeoffs

Worth noting:

- **Single-vector embeddings.** No ColBERT-style late interaction. Simpler, but loses some token-level precision.
- **Single-pass retrieval.** No agentic loop, no query decomposition for multi-hop questions.
- **Filter detection for aggregation is keyword-list-based.** Works for severity / status / priority words but not free-form filters like *incidents about login outages*.
- **No structured output for tables.** Tabular answers are free-text rather than tool-call schemas.
- **No deep prompt-injection mitigation.** A document containing *Ignore previous instructions…* is still in scope. Bedrock Guardrails covers content filtering and topic denial when configured.

## Project layout

```
app.py                  Streamlit UI and chat loop
rag_engine.py           Retrieval pipeline, generation, citation verification
chunker.py              Three chunking strategies (semantic / hierarchical / fixed)
ocr_engine.py           PDF / image / xlsx text extraction
bm25_index.py           OpenSearch-backed BM25 index
reranker.py             Cross-encoder rerank + MMR diversity selection
semantic_cache.py       SQLite-backed semantic answer cache
query_rewriter.py       Conversational query rewriter (Claude Haiku)
memory_manager.py       LangGraph + SQLite memory with rolling summary
evaluator.py            Per-turn retrieval / generation / citation metrics
golden_evaluator.py     Batch evaluator over the golden dataset
golden_dataset.json     25 ground-truth Q&A pairs
docker-compose.yml      OpenSearch container
requirements.txt        Python dependencies
```

## Persistent state

These files / directories are written at runtime and intentionally gitignored:

- `chroma_db/` — dense vector store
- `memory.db` — chat history, threads, LangGraph checkpoints
- `semantic_cache.db` — semantic answer cache
- OpenSearch index `aetherrag_bm25` — sparse BM25 index, in the `opensearch-data` Docker volume

Deleting any of them resets that layer. The "Clear Database & Chat" button in the sidebar resets all three at once.
