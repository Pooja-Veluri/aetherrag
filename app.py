import streamlit as st
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from ocr_engine import extract_text
from rag_engine import RagEngine
from memory_manager import get_memory_manager
from evaluator import RAGEvaluator
from query_rewriter import rewrite_query

# Refusal threshold for the groundedness gate. The cross-encoder
# (ms-marco-MiniLM-L-6-v2) returns logits, not probabilities — typical
# in-domain matches score 4–10, off-topic matches score below 0. -2.0 is
# conservative: anything that low is almost certainly a hallucination risk.
GROUNDEDNESS_THRESHOLD = -2.0

st.set_page_config(
    page_title="AetherRAG - Advanced Document Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');

.stApp {
    background: radial-gradient(circle at 10% 20%, #0F0E26 0%, #06050F 90%);
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

.header-container {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    background: linear-gradient(180deg, rgba(99,102,241,0.05) 0%, rgba(0,0,0,0) 100%);
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.glow-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #A855F7 0%, #6366F1 50%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(99, 102, 241, 0.2);
    margin: 0;
    letter-spacing: -0.03em;
}

.subtitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    color: #94A3B8;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

[data-testid="stSidebar"] {
    background-color: rgba(10, 8, 28, 0.92) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5);
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Outfit', sans-serif;
    color: #C7D2FE !important;
    font-weight: 600;
}

.status-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

.status-card h4 {
    margin: 0 0 0.5rem 0;
    color: #A855F7;
    font-family: 'Outfit', sans-serif;
}

.source-card {
    background: rgba(15, 23, 42, 0.65);
    border-left: 4px solid #6366F1;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 0 12px 12px 0;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.source-card:hover {
    border-left-color: #A855F7;
    background: rgba(30, 41, 59, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.08);
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.source-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #C7D2FE;
    font-size: 0.95rem;
}

.source-badges {
    display: flex;
    gap: 0.5rem;
}

.badge-page {
    background-color: rgba(99, 102, 241, 0.2);
    color: #818CF8;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.badge-score {
    background-color: rgba(244, 63, 94, 0.15);
    color: #FB7185;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(244, 63, 94, 0.25);
}

.source-text {
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.5;
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.6rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.02);
}

.stChatMessage {
    background-color: rgba(17, 24, 39, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}

div[data-testid="stChatMessageUser"] {
    background-color: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(168, 85, 247, 0.4); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1 class="glow-title">AetherRAG</h1>
    <div class="subtitle">PREMIUM DOCUMENT COGNITION • AWS BEDROCK & CHROMADB</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

memory_manager = get_memory_manager()

# ---------------------------------------------------------------------------
# Session state — reload-resilient thread init
# ---------------------------------------------------------------------------

# Credentials and tracing config are loaded silently from .env on every rerun
# (no user-facing inputs — keys never appear in the UI).
st.session_state.aws_access_key    = os.getenv("AWS_ACCESS_KEY_ID", "")
st.session_state.aws_secret_key    = os.getenv("AWS_SECRET_ACCESS_KEY", "")
st.session_state.aws_session_token = os.getenv("AWS_SESSION_TOKEN", "")
st.session_state.aws_region        = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# LangSmith picks these up directly from the env, so just propagate.
st.session_state.langsmith_project = os.getenv("LANGCHAIN_PROJECT", "aetherrag")
st.session_state.langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
os.environ["LANGCHAIN_PROJECT"]    = st.session_state.langsmith_project

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "thread_id" not in st.session_state:
    recent = memory_manager.get_most_recent_thread()
    if recent:
        st.session_state.thread_id = recent["thread_id"]
    else:
        st.session_state.thread_id = memory_manager.create_thread()

if "messages" not in st.session_state:
    st.session_state.messages = memory_manager.get_display_history(st.session_state.thread_id)

# Per-session evaluator (holds turn history for live metrics panel)
if "evaluator" not in st.session_state:
    st.session_state.evaluator = None  # initialised after rag_engine is available
if "golden_report" not in st.session_state:
    st.session_state.golden_report = None
if "golden_running" not in st.session_state:
    st.session_state.golden_running = False

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:

    # ── Thread management ──────────────────────────────────────────────────
    st.markdown("## 💬 Conversations")

    if st.button("➕ New Session", use_container_width=True):
        new_tid = memory_manager.create_thread()
        st.session_state.thread_id = new_tid
        st.session_state.messages = []
        st.rerun()

    threads = memory_manager.list_threads()
    if threads:
        st.markdown("#### Past Sessions")
        for t in threads:
            is_active = t["thread_id"] == st.session_state.thread_id
            label = ("▶ " if is_active else "") + t["thread_name"]
            col_name, col_del = st.columns([4, 1])
            with col_name:
                if st.button(label, key=f"thread_{t['thread_id']}", use_container_width=True):
                    if not is_active:
                        st.session_state.thread_id = t["thread_id"]
                        st.session_state.messages = memory_manager.get_display_history(t["thread_id"])
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{t['thread_id']}"):
                    memory_manager.delete_thread(t["thread_id"])
                    if t["thread_id"] == st.session_state.thread_id:
                        new_tid = memory_manager.create_thread()
                        st.session_state.thread_id = new_tid
                        st.session_state.messages = []
                    st.rerun()

    st.divider()

    # Per-turn evaluation always runs (LangSmith captures the trace + scores).
    # The UI no longer renders the metrics panel — view them in LangSmith.
    gen_eval = True
    # Force tracing on so every turn shows up in LangSmith. Falls through to a
    # local-only run cleanly if no LANGCHAIN_API_KEY is present in env.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # ── Model & vector store ───────────────────────────────────────────────
    st.markdown("## ⚙️ Configuration")
    st.markdown("### 🤖 LLM & Vector Store")
    model_options = {
        "Claude 3 Haiku":    "anthropic.claude-3-haiku-20240307-v1:0",
        "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Titan Text Express": "amazon.titan-text-express-v1",
    }
    selected_model_label = st.selectbox("Select Model", options=list(model_options.keys()), index=0)
    selected_model = model_options[selected_model_label]

    # ── Advanced options (collapsed by default) ───────────────────────────
    with st.expander("Advanced settings"):
        st.markdown("**✂️ Chunking**")
        strategy_info = {
            "semantic":     "Sentence-aware splits with section detection. Best for most documents.",
            "hierarchical": "Small retrieval chunks + wider parent context sent to LLM. Best for dense/technical docs.",
            "fixed":        "Recursive character splitting. Fastest; no NLP.",
        }
        chunk_strategy = st.selectbox(
            "Strategy",
            options=list(strategy_info.keys()),
            index=0,
            format_func=lambda s: s.capitalize(),
        )
        st.caption(strategy_info[chunk_strategy])

        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=800, step=100)
        with col2:
            chunk_overlap = st.slider("Overlap", min_value=0, max_value=500, value=150, step=50)

        st.markdown("**🔎 Retrieval**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            use_reranker = st.toggle("Cross-Encoder Rerank", value=True)
        with col_r2:
            n_results = st.slider("Top-K final", min_value=2, max_value=10, value=5)

    # ── Initialize RAG engine ──────────────────────────────────────────────
    try:
        rag_engine = RagEngine(
            aws_access_key=st.session_state.aws_access_key,
            aws_secret_key=st.session_state.aws_secret_key,
            aws_session_token=st.session_state.aws_session_token,
            region=st.session_state.aws_region,
        )
    except Exception as init_err:
        st.error(f"Failed to initialize RAG Engine: {init_err}")
        st.stop()

    # Reinitialise evaluator on every rerun so session state never holds a
    # stale instance from before a code change (preserves turn history).
    existing_history = (st.session_state.evaluator.history
                        if st.session_state.evaluator else [])
    st.session_state.evaluator = RAGEvaluator(
        bedrock_client=rag_engine.bedrock_client,
        model_id=selected_model,
    )
    st.session_state.evaluator.history = existing_history

    # ── Document upload ────────────────────────────────────────────────────
    st.markdown("### 📁 Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload files for OCR / Text extraction",
        type=["pdf", "png", "jpg", "jpeg", "webp", "xlsx", "xlsm"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        pages_data = extract_text(file_bytes, uploaded_file.name)
                        rag_engine.add_document(
                            file_name=uploaded_file.name,
                            pages_data=pages_data,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            strategy=chunk_strategy,
                        )
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.sidebar.success(f"✅ Ingested {uploaded_file.name} successfully!")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

    if st.session_state.processed_files:
        st.markdown("#### 📄 Loaded Documents")
        for f in list(st.session_state.processed_files):
            row_name, row_del = st.columns([5, 1])
            row_name.markdown(f"`{f}`")
            if row_del.button("🗑", key=f"del_doc_{f}", help=f"Remove {f} from indexes"):
                with st.spinner(f"Removing {f}..."):
                    rag_engine.remove_document(f)
                    st.session_state.processed_files = [
                        x for x in st.session_state.processed_files if x != f
                    ]
                    st.success(f"Removed {f}")
                    st.rerun()

        if st.button("🧹 Clear Database & Chat", use_container_width=True):
            with st.spinner("Clearing Vector Database..."):
                rag_engine.reset_collection()
                rag_engine.semantic_cache.clear()
                st.session_state.processed_files = []
                new_tid = memory_manager.create_thread()
                st.session_state.thread_id = new_tid
                st.session_state.messages = []
                st.success("Database, semantic cache, and chat history cleared!")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

    # ── Golden dataset evaluation ──────────────────────────────────────────
    st.markdown("### 🏅 Golden Dataset Evaluation")
    st.caption("Runs all 25 ground-truth questions through the live RAG pipeline.")
    if st.button("▶ Run Golden Evaluation", use_container_width=True,
                 disabled=not st.session_state.processed_files):
        st.session_state.golden_report = None
        st.session_state.golden_running = True
        st.rerun()

    if st.session_state.get("golden_running"):
        from golden_evaluator import run_golden_evaluation
        progress_bar  = st.progress(0, text="Starting evaluation…")
        status_text   = st.empty()

        def _progress(current, total, qid):
            progress_bar.progress(current / total, text=f"Evaluating {qid} ({current}/{total})")
            status_text.caption(f"Running question {qid}…")

        with st.spinner("Running golden evaluation — this will take ~2–3 minutes…"):
            try:
                report = run_golden_evaluation(
                    rag_engine=rag_engine,
                    model_id=selected_model,
                    n_results=n_results,
                    use_reranker=use_reranker,
                    progress_callback=_progress,
                )
                st.session_state.golden_report  = report
                st.session_state.golden_running = False
                progress_bar.progress(1.0, text="Complete!")
                st.success(f"Evaluation complete — {report['evaluated']}/{report['total_questions']} questions evaluated.")
            except Exception as e:
                st.session_state.golden_running = False
                st.error(f"Evaluation failed: {e}")
                st.stop()

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown(
        """
        <div style="background-color: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.15);
                    border-radius: 12px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
            <h3 style="margin-top: 0; color: #C7D2FE; font-family: 'Outfit', sans-serif;">🔮 Welcome to your Document Brain</h3>
            <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto;">
                AetherRAG uses AWS Bedrock and ChromaDB to search and answer questions based on your uploads.
                Drag-and-drop your PDFs or images in the sidebar, then start chatting.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div>⚡ <b>OCR Enabled</b><br><span style="font-size: 0.85rem; color: #94A3B8;">Automatic scanned PDF & Image OCR</span></div>
                <div>🔍 <b>Vector Retrieval</b><br><span style="font-size: 0.85rem; color: #94A3B8;">ChromaDB Semantic Search</span></div>
                <div>🛡️ <b>Secure API</b><br><span style="font-size: 0.85rem; color: #94A3B8;">Powered by AWS Bedrock Claude/Titan</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Golden evaluation results panel
# ---------------------------------------------------------------------------

if st.session_state.get("golden_report"):
    report = st.session_state.golden_report
    avgs   = report.get("averages", {})
    by_type = report.get("by_question_type", {})

    with st.expander("🏅 Golden Dataset Evaluation Results", expanded=True):

        st.caption(
            f"Model: `{report['model_id']}` · "
            f"{report['evaluated']}/{report['total_questions']} questions · "
            f"LangSmith project: `{report.get('langsmith_project', '—')}`"
        )

        # ── Aggregate metrics ──────────────────────────────────────────────
        st.markdown("#### Aggregate Metrics")
        col_groups = [
            ("Retrieval", ["hit_rate", "mrr", "ndcg", "precision_at_k", "chunk_diversity", "bm25_contribution", "rerank_score_top1"]),
            ("Answer Quality", ["exact_match", "token_f1"]),
            ("Generation (LLM Judge)", ["faithfulness", "answer_relevance", "answer_completeness", "context_utilisation"]),
        ]
        for group_name, keys in col_groups:
            present = [k for k in keys if k in avgs]
            if not present:
                continue
            st.markdown(f"**{group_name}**")
            cols = st.columns(len(present))
            for col, k in zip(cols, present):
                label = k.replace("_", " ").title()
                col.metric(label, f"{avgs[k]:.3f}")

        # ── By question type ───────────────────────────────────────────────
        if by_type:
            st.markdown("#### By Question Type")
            type_rows = []
            for qt, data in sorted(by_type.items()):
                type_rows.append({
                    "Type":        qt,
                    "Count":       data["count"],
                    "Hit Rate":    f"{data.get('hit_rate', 0):.2f}",
                    "Token F1":    f"{data.get('token_f1', 0):.2f}",
                    "Faithfulness":f"{data.get('faithfulness', 0):.2f}",
                    "Relevance":   f"{data.get('answer_relevance', 0):.2f}",
                    "Completeness":f"{data.get('answer_completeness', 0):.2f}",
                })
            st.dataframe(type_rows, use_container_width=True, hide_index=True)

        # ── Per-question table ─────────────────────────────────────────────
        st.markdown("#### Per-Question Results")
        rows = []
        for r in report.get("results", []):
            if "error" in r:
                rows.append({"ID": r["id"], "Question": r["question"][:60], "Error": r["error"]})
                continue
            m = r.get("metrics", {})
            rows.append({
                "ID":          r["id"],
                "Type":        r["question_type"],
                "Question":    r["question"][:55] + "…",
                "Ground Truth": r["ground_truth"][:40] + "…",
                "Answer":      r.get("answer", "")[:40] + "…",
                "EM":          f"{m.get('exact_match', 0):.0f}",
                "F1":          f"{m.get('token_f1', 0):.2f}",
                "Hit":         f"{m.get('hit_rate', 0):.2f}",
                "Faith.":      f"{m.get('faithfulness', 0):.2f}",
                "Relevance":   f"{m.get('answer_relevance', 0):.2f}",
                "Complete.":   f"{m.get('answer_completeness', 0):.2f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # ── Download ───────────────────────────────────────────────────────
        st.download_button(
            "⬇ Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="evaluation_report.json",
            mime="application/json",
        )

# Render full display history (loaded from SQLite)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Retrieved Context & Citations"):
                for idx, src in enumerate(msg["sources"]):
                    rerank_badge = (
                        f'<span class="badge-score">Rerank: {src["rerank_score"]:.3f}</span>'
                        if src.get("rerank_score") else ""
                    )
                    section_label = f" — {src['section']}" if src.get("section") else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <div class="source-name">Chunk {idx+1}: {src['source']}{section_label}</div>
                            <div class="source-badges">
                                <span class="badge-page">Page {src['page']}</span>
                                {rerank_badge}
                            </div>
                        </div>
                        <div class="source-text">{src.get('snippet', src.get('text', ''))}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    thread_id = st.session_state.thread_id

    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Conversational query rewriting. Follow-ups like "what about Feb?"
    #    are useless to the retriever standalone — rewrite them into a
    #    self-contained search query using prior turns. Falls back to the
    #    original query on any failure.
    memory_context = memory_manager.get_memory_context(thread_id)
    prior_history  = memory_context.get("recent_messages") or []
    search_query   = rewrite_query(
        user_query,
        history=prior_history,
        bedrock_client=rag_engine.bedrock_client,
    )

    # 3. Semantic cache lookup keyed on the REWRITTEN query so paraphrased
    #    follow-ups hit cache too. Scope binds to (model, file set) so a hit
    #    can't bleed across models or corpuses.
    cache_scope = f"{selected_model}::{','.join(sorted(st.session_state.processed_files))}"
    cache_hit   = rag_engine.semantic_cache.lookup(search_query, scope=cache_scope)

    contexts:    list[dict] = []
    aggregation             = None
    n_bm25                  = 0
    full_response           = ""
    low_confidence          = False
    citation_check          = None

    # 4. Persist user message (must happen even on cache hit so chat history
    #    stays consistent and thread auto-rename fires for the first turn).
    memory_manager.add_message(
        thread_id=thread_id,
        role="user",
        content=user_query,
        sources=None,
        bedrock_client=rag_engine.bedrock_client,
        model_id=selected_model,
    )

    if cache_hit:
        contexts      = cache_hit["contexts"] or []
        full_response = cache_hit["answer"]
        with st.chat_message("assistant"):
            st.markdown(full_response)
            st.caption(
                f"⚡ Served from semantic cache "
                f"(similarity {cache_hit['similarity']:.3f}, "
                f"age {cache_hit['age_seconds']}s)"
            )
    else:
        # Full RAG path: hybrid retrieval → rerank → MMR → groundedness gate
        with st.spinner("Searching with hybrid BM25 + semantic retrieval..."):
            retrieval_result = rag_engine.query_context(
                search_query,
                n_results=n_results,
                use_reranker=use_reranker,
                groundedness_threshold=GROUNDEDNESS_THRESHOLD,
            )
        contexts       = retrieval_result["chunks"]
        n_bm25         = retrieval_result["n_bm25"]
        aggregation    = retrieval_result.get("aggregation")
        low_confidence = retrieval_result.get("low_confidence", False)

        with st.chat_message("assistant"):
            if low_confidence:
                # Groundedness abstention: retrieval is too weak to ground
                # an answer. Refuse instead of hallucinating.
                full_response = (
                    "I couldn't find anything in the indexed documents "
                    "that confidently answers this question. Try rephrasing, "
                    "uploading a more relevant document, or asking about a "
                    "specific section/page."
                )
                st.markdown(full_response)
                st.caption(
                    f"⚠️ Low retrieval confidence "
                    f"(top rerank score {retrieval_result.get('top_rerank_score', 0):.2f})"
                )
            else:
                # Show the rewritten query if it actually changed — helps
                # debug retrieval misses.
                if search_query.strip() != user_query.strip():
                    st.caption(f"🔎 Searching for: _{search_query}_")

                response_placeholder = st.empty()
                stream = rag_engine.generate_response_stream(
                    model_id=selected_model,
                    query=user_query,
                    contexts=contexts,
                    memory_context=memory_context,
                    aggregation=aggregation,
                )
                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                # Citation verification: does every (source, page) the model
                # cited actually appear in the retrieved chunks? Surface
                # unverified ones to the user; full report goes to LangSmith
                # via the evaluator below.
                citation_check = RagEngine.verify_citations(full_response, contexts)
                if citation_check["unverified"]:
                    bad = ", ".join(
                        f"`{src}` p.{pg}" for src, pg in citation_check["unverified"]
                    )
                    st.caption(f"⚠️ Unverified citations: {bad}")

        # Cache fresh answers (skip refusals — they'd lock in low confidence)
        if not low_confidence:
            rag_engine.semantic_cache.put(
                query=search_query,
                answer=full_response,
                contexts=contexts,
                scope=cache_scope,
            )

    # 4. Citations (rendered for both cache hit and miss)
    if contexts:
        with st.expander("🔍 Retrieved Context & Citations"):
            for idx, src in enumerate(contexts):
                rerank_badge = (
                    f'<span class="badge-score">Rerank: {src["rerank_score"]:.3f}</span>'
                    if src.get("rerank_score") else ""
                )
                rrf_badge = (
                    f'<span class="badge-score" style="background:rgba(16,185,129,0.15);color:#34d399;border-color:rgba(16,185,129,0.3)">RRF: {src["rrf_score"]:.4f}</span>'
                    if src.get("rrf_score") else ""
                )
                bm25_badge = (
                    f'<span class="badge-score" style="background:rgba(251,191,36,0.15);color:#fbbf24;border-color:rgba(251,191,36,0.3)">BM25: {src["bm25_score"]:.2f}</span>'
                    if src.get("bm25_score", 0) > 0 else ""
                )
                section_label = f" — {src['section']}" if src.get("section") else ""
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <div class="source-name">Chunk {idx+1}: {src['source']}{section_label}</div>
                        <div class="source-badges">
                            <span class="badge-page">Page {src['page']}</span>
                            {rerank_badge}{rrf_badge}{bm25_badge}
                        </div>
                    </div>
                    <div class="source-text">{src.get('snippet', src.get('text', ''))}</div>
                </div>
                """, unsafe_allow_html=True)

    # 5. Persist assistant message
    memory_manager.add_message(
        thread_id=thread_id,
        role="assistant",
        content=full_response,
        sources=contexts,
        bedrock_client=rag_engine.bedrock_client,
        model_id=selected_model,
    )

    # 6. Per-turn evaluation — silent. Retrieval + generation metrics computed
    #    and posted as feedback to LangSmith. No UI panel. Skipped on cache
    #    hits because there's no fresh retrieval/generation to evaluate.
    evaluator: RAGEvaluator = st.session_state.evaluator
    if evaluator and not cache_hit and not low_confidence:
        try:
            evaluator.evaluate_turn(
                query=user_query,
                answer=full_response,
                contexts=contexts,
                n_bm25=n_bm25,
                run_generation_eval=gen_eval,
                citation_check=citation_check,
            )
        except Exception as eval_err:
            # Evaluation must never break the chat. Log to server console.
            print(f"[evaluator] turn eval failed: {eval_err}")

    # 7. Sync session state from DB
    st.session_state.messages = memory_manager.get_display_history(thread_id)
