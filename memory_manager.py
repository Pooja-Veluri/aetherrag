import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Iterator, List

import streamlit as st
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

DB_PATH = "./memory.db"
RECENT_WINDOW = 5   # messages kept in hot state after compression
COMPRESS_AT = 2     # compress when len(messages) > RECENT_WINDOW + COMPRESS_AT


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ThreadState(TypedDict):
    messages: List[dict]    # plain list — node outputs overwrite (no append reducer)
    summary: str            # accumulated summary of compressed turns
    total_count: int        # monotonically increasing, never decremented
    pending: List[dict]     # incoming buffer; cleared by append_node


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _append_node(state: ThreadState) -> dict:
    existing = state.get("messages") or []
    pending  = state.get("pending")  or []
    return {
        "messages":    existing + pending,
        "pending":     [],
        "total_count": (state.get("total_count") or 0) + len(pending),
    }


def _should_compress(state: ThreadState) -> str:
    if len(state.get("messages") or []) > RECENT_WINDOW + COMPRESS_AT:
        return "compress"
    return END


def _compress_node(state: ThreadState, config: RunnableConfig) -> dict:
    msgs             = state["messages"]
    old_msgs         = msgs[:-RECENT_WINDOW]
    recent           = msgs[-RECENT_WINDOW:]
    existing_summary = state.get("summary") or ""
    bedrock_client   = config["configurable"].get("bedrock_client")
    model_id         = config["configurable"].get("model_id", "")
    new_summary      = _generate_summary(bedrock_client, model_id, old_msgs, existing_summary)
    return {"messages": recent, "summary": new_summary}


def _generate_summary(bedrock_client, model_id: str, messages: list, existing_summary: str) -> str:
    if not bedrock_client or not messages:
        return existing_summary

    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    if existing_summary:
        user_prompt = (
            f"Prior summary:\n{existing_summary}\n\n"
            f"New conversation turns:\n{conv_text}\n\n"
            "Create an updated concise summary that captures all important context from both."
        )
    else:
        user_prompt = (
            f"Conversation:\n{conv_text}\n\n"
            "Create a concise summary capturing the key topics and context."
        )

    try:
        if "anthropic.claude" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "system": "You are a helpful assistant that summarizes conversations concisely.",
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.0,
            })
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"].strip()

        elif "amazon.titan" in model_id:
            prompt = (
                "You summarize conversations concisely.\n\n"
                f"{user_prompt}\n\nSummary:"
            )
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {"maxTokenCount": 256, "temperature": 0.0},
            })
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["results"][0]["outputText"].strip()

        else:
            return existing_summary

    except Exception as e:
        logger.warning(f"_generate_summary failed: {e}. Keeping existing summary.")
        return existing_summary


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(checkpointer):
    g = StateGraph(ThreadState)
    g.add_node("append_node", _append_node)
    g.add_node("compress_node", _compress_node)
    g.set_entry_point("append_node")
    g.add_conditional_edges(
        "append_node",
        _should_compress,
        {"compress": "compress_node", END: END},
    )
    g.add_edge("compress_node", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_custom_tables()
        self.checkpointer = SqliteSaver(self.conn)
        self.checkpointer.setup()
        self.graph = _build_graph(self.checkpointer)

    def _create_custom_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id    TEXT PRIMARY KEY,
                thread_name  TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                last_updated TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                sources    TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_thread ON chat_history(thread_id);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def create_thread(self, thread_name: str | None = None) -> str:
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        name = thread_name or "Session " + datetime.now().strftime("%b %d %H:%M")
        self.conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?)",
            (thread_id, name, now, now),
        )
        self.conn.commit()
        return thread_id

    def list_threads(self) -> list:
        cur = self.conn.execute(
            "SELECT thread_id, thread_name, created_at, last_updated "
            "FROM threads ORDER BY last_updated DESC"
        )
        return [
            {"thread_id": r[0], "thread_name": r[1], "created_at": r[2], "last_updated": r[3]}
            for r in cur.fetchall()
        ]

    def delete_thread(self, thread_id: str):
        self.conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        self.conn.execute("DELETE FROM chat_history WHERE thread_id = ?", (thread_id,))
        self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        self.conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        self.conn.commit()

    def rename_thread(self, thread_id: str, name: str):
        self.conn.execute(
            "UPDATE threads SET thread_name = ? WHERE thread_id = ?", (name, thread_id)
        )
        self.conn.commit()

    def get_most_recent_thread(self) -> dict | None:
        cur = self.conn.execute(
            "SELECT thread_id, thread_name FROM threads ORDER BY last_updated DESC LIMIT 1"
        )
        row = cur.fetchone()
        return {"thread_id": row[0], "thread_name": row[1]} if row else None

    # ------------------------------------------------------------------
    # Message persistence
    # ------------------------------------------------------------------

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        sources=None,
        bedrock_client=None,
        model_id: str = "",
    ):
        now = datetime.now(timezone.utc).isoformat()

        # 1. Persist full message to display history
        self.conn.execute(
            "INSERT INTO chat_history (thread_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(sources) if sources else None, now),
        )

        # 2. Update LangGraph compressed state
        lg_msg = {"role": role, "content": content}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "bedrock_client": bedrock_client,
                "model_id": model_id,
            }
        }
        snap = self.graph.get_state(config)
        if not snap.values:
            invoke_input = {
                "pending":     [lg_msg],
                "messages":    [],
                "summary":     "",
                "total_count": 0,
            }
        else:
            invoke_input = {"pending": [lg_msg]}

        self.graph.invoke(invoke_input, config)

        # 3. Update thread timestamp
        self.conn.execute(
            "UPDATE threads SET last_updated = ? WHERE thread_id = ?", (now, thread_id)
        )

        # 4. Auto-rename thread from first user message
        if role == "user":
            snap_after = self.graph.get_state(config)
            if snap_after.values.get("total_count") == 1:
                auto_name = content.strip()[:40].rstrip()
                if len(content.strip()) > 40:
                    auto_name += "..."
                self.conn.execute(
                    "UPDATE threads SET thread_name = ? WHERE thread_id = ?",
                    (auto_name, thread_id),
                )

        self.conn.commit()

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def get_memory_context(self, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        snap = self.graph.get_state(config)
        if not snap.values:
            return {"summary": "", "recent_messages": []}
        v = snap.values
        return {
            "summary":         v.get("summary") or "",
            "recent_messages": v.get("messages") or [],
        }

    def get_display_history(self, thread_id: str) -> list:
        cur = self.conn.execute(
            "SELECT role, content, sources FROM chat_history "
            "WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        )
        result = []
        for role, content, sources_json in cur.fetchall():
            msg = {"role": role, "content": content}
            if sources_json:
                try:
                    msg["sources"] = json.loads(sources_json)
                except Exception:
                    msg["sources"] = []
            result.append(msg)
        return result


# ---------------------------------------------------------------------------
# Singleton factory — one MemoryManager per Streamlit server process
# ---------------------------------------------------------------------------

@st.cache_resource
def get_memory_manager() -> MemoryManager:
    return MemoryManager()
