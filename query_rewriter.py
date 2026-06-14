"""
Conversational query rewriter.

A user follow-up like "what about February?" is meaningless on its own —
retrieval against that string returns nothing useful. We use the previous
turns of the conversation to rewrite it into a self-contained search query
("how many critical incidents occurred in February").

The rewriter is intentionally cheap:
  - Calls Claude Haiku (the fastest Bedrock model the app already supports)
  - Runs in parallel with retrieval is NOT possible because retrieval needs
    the rewritten query — but it adds <300ms in practice and only fires when
    the user has prior context.
  - Falls back to the original query on any error so the chat path is robust.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Use the fastest Claude model regardless of what the user picked for
# generation — rewriting doesn't need Sonnet-level capability.
_REWRITE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Skip rewriting altogether for queries that are obviously self-contained.
# These are short heuristics — when in doubt we still rewrite, since false
# positives (rewriting a clear query) are harmless.
_SELF_CONTAINED_PATTERNS = re.compile(
    r"^(what|how|why|when|where|who|which|list|show|count|tell me|describe|explain)\b",
    re.IGNORECASE,
)

# Hard cap so we don't blow the rewriter prompt up on long histories.
_MAX_HISTORY_TURNS = 4


def _looks_like_followup(query: str) -> bool:
    """
    True if `query` clearly references prior conversation context. Used to
    decide whether rewriting is worth a Bedrock call.
    """
    q = query.strip().lower()
    if len(q) < 12:
        return True
    # Pronouns / discourse markers that almost always reference prior turns
    if re.search(r"\b(it|its|they|them|their|that|this|those|these|he|she)\b", q):
        return True
    if re.search(r"\b(also|then|next|after|before|same|similar|other|another|too|as well)\b", q):
        return True
    if q.startswith(("and ", "but ", "or ", "what about", "how about", "more on", "tell me more")):
        return True
    return False


def rewrite_query(
    query: str,
    history: list[dict],
    bedrock_client,
    model_id: Optional[str] = None,
) -> str:
    """
    Return a standalone search query. Falls back to `query` on any failure
    or when rewriting isn't warranted.

    history: list of {"role": "user"|"assistant", "content": str}, oldest first.
             Only user turns are used as context (assistant turns can drift the
             rewrite toward the model's own phrasing).
    """
    if not query or not query.strip():
        return query

    # No prior context to draw on
    user_turns = [m for m in (history or []) if m.get("role") == "user" and m.get("content")]
    if not user_turns:
        return query

    # Already self-contained AND not obviously a follow-up → skip the call
    if _SELF_CONTAINED_PATTERNS.match(query.strip()) and not _looks_like_followup(query):
        return query

    if not bedrock_client:
        return query

    recent = user_turns[-_MAX_HISTORY_TURNS:]
    history_text = "\n".join(f"- {m['content']}" for m in recent)

    rewrite_prompt = (
        "You rewrite conversational follow-up questions into self-contained search queries.\n"
        "Use the prior user questions as context to fill in missing subjects/entities.\n"
        "Return ONLY the rewritten query — no preamble, no explanation, no quotes.\n"
        "If the current question is already self-contained, return it verbatim.\n\n"
        f"Prior user questions:\n{history_text}\n\n"
        f"Current question: {query}\n\n"
        "Rewritten search query:"
    )

    use_model = model_id or _REWRITE_MODEL_ID

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 96,
            "system": "You rewrite questions into self-contained search queries. Output only the rewritten query.",
            "messages": [{"role": "user", "content": rewrite_prompt}],
            "temperature": 0.0,
        })
        resp = bedrock_client.invoke_model(
            modelId=use_model,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        text = json.loads(resp["body"].read())["content"][0]["text"].strip()
        # Strip surrounding quotes the model sometimes adds despite instructions
        text = text.strip().strip('"').strip("'").strip()
        if not text or len(text) > 4 * len(query) + 200:
            # Empty or absurdly long — treat as failure
            return query
        return text
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}. Using original query.")
        return query
