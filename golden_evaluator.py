"""
Golden-dataset batch evaluator for AetherRAG.

Uploads golden_dataset.json to LangSmith as a dataset (once, idempotent),
then runs every question through the live RAG pipeline and evaluates each
answer against the ground truth using:

  Retrieval  (deterministic):
    hit_rate, mrr, ndcg, precision_at_k, chunk_diversity,
    bm25_contribution, rerank_score_top1

  Answer quality  (string-match + LLM-as-judge):
    exact_match        — normalised exact string match
    token_f1           — token-overlap F1 (standard QA metric)
    faithfulness       — answer grounded in retrieved context
    answer_relevance   — answer addresses the question
    answer_completeness — answer covers all aspects
    context_utilisation — how much context was used

Results are:
  1. Posted to LangSmith as feedback on each run (visible in the UI)
  2. Written to evaluation_report.json for local inspection
"""

import json
import logging
import os
import re
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATASET_NAME  = "aetherrag-golden-dataset"
GOLDEN_PATH   = Path(__file__).parent / "golden_dataset.json"
REPORT_PATH   = Path(__file__).parent / "evaluation_report.json"


# ---------------------------------------------------------------------------
# Token-level helpers (for exact_match / token_f1)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalise(prediction) == _normalise(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalise(prediction).split()
    gt_tokens   = _normalise(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


# ---------------------------------------------------------------------------
# LangSmith dataset management
# ---------------------------------------------------------------------------

def _get_or_create_dataset(client, items: list[dict]):
    """Upload the golden dataset to LangSmith if it doesn't exist yet."""
    try:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info(f"Reusing existing LangSmith dataset '{DATASET_NAME}' (id={ds.id})")
        return ds
    except Exception:
        pass

    ds = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="AetherRAG golden QA dataset — 25 ground-truth question/answer pairs",
    )
    examples = [
        {
            "inputs":  {"question": item["question"], "id": item["id"],
                        "question_type": item["question_type"], "source_page": item["source_page"]},
            "outputs": {"ground_truth": item["ground_truth"]},
        }
        for item in items
    ]
    client.create_examples(dataset_id=ds.id, inputs=[e["inputs"] for e in examples],
                           outputs=[e["outputs"] for e in examples])
    logger.info(f"Created LangSmith dataset '{DATASET_NAME}' with {len(examples)} examples.")
    return ds


# ---------------------------------------------------------------------------
# LLM-as-judge helpers (reuse Bedrock)
# ---------------------------------------------------------------------------

def _bedrock_judge(bedrock_client, model_id: str, prompt: str) -> float:
    try:
        if "anthropic.claude" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            })
            resp = bedrock_client.invoke_model(
                modelId=model_id, contentType="application/json",
                accept="application/json", body=body,
            )
            text = json.loads(resp["body"].read())["content"][0]["text"].strip()
        elif "amazon.titan" in model_id:
            body = json.dumps({
                "inputText": prompt + "\nScore:",
                "textGenerationConfig": {"maxTokenCount": 8, "temperature": 0.0},
            })
            resp = bedrock_client.invoke_model(
                modelId=model_id, contentType="application/json",
                accept="application/json", body=body,
            )
            text = json.loads(resp["body"].read())["results"][0]["outputText"].strip()
        else:
            return 0.5
        m = re.search(r"[\d.]+", text)
        val = float(m.group()) if m else 0.5
        return round(min(1.0, max(0.0, val / 5.0 if val > 1.0 else val)), 3)
    except Exception as e:
        logger.warning(f"Bedrock judge error: {e}")
        return 0.5


def _llm_metrics(query: str, answer: str, ground_truth: str,
                 contexts: list[dict], bedrock_client, model_id: str) -> dict:
    ctx_str = "\n\n".join(c.get("snippet", c.get("text", "")) for c in contexts)[:2500]

    faithfulness_prompt = (
        "Rate FAITHFULNESS 0-5: does the ANSWER contain only information from the CONTEXT?\n"
        "5=no hallucinations. 0=completely fabricated.\n\n"
        f"CONTEXT:\n{ctx_str}\n\nANSWER:\n{answer[:1000]}\n\nReply with one number 0-5."
    )
    relevance_prompt = (
        "Rate RELEVANCE 0-5: does the ANSWER address the QUESTION?\n"
        "5=complete answer. 0=off-topic.\n\n"
        f"QUESTION:\n{query}\n\nANSWER:\n{answer[:1000]}\n\nReply with one number 0-5."
    )
    completeness_prompt = (
        "Rate COMPLETENESS 0-5: does the ANSWER cover all aspects of the QUESTION "
        "compared to the REFERENCE?\n"
        "5=fully covers all aspects. 0=misses everything.\n\n"
        f"QUESTION:\n{query}\n\nREFERENCE:\n{ground_truth}\n\nANSWER:\n{answer[:1000]}\n\n"
        "Reply with one number 0-5."
    )

    # Context utilisation heuristic
    ctx_sents = [s.strip() for s in ctx_str.replace("\n", ". ").split(". ") if len(s.strip()) > 20]
    ans_lower = answer.lower()
    ctx_util  = 0.0
    if ctx_sents:
        hits = sum(1 for s in ctx_sents
                   if any(w in ans_lower for w in s.lower().split()[:5] if len(w) > 4))
        ctx_util = round(min(1.0, hits / len(ctx_sents)), 3)

    return {
        "faithfulness":          _bedrock_judge(bedrock_client, model_id, faithfulness_prompt),
        "answer_relevance":      _bedrock_judge(bedrock_client, model_id, relevance_prompt),
        "answer_completeness":   _bedrock_judge(bedrock_client, model_id, completeness_prompt),
        "context_utilisation":   ctx_util,
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_golden_evaluation(
    rag_engine,
    model_id: str,
    n_results: int = 5,
    use_reranker: bool = True,
    progress_callback=None,
) -> dict:
    """
    Run all 25 golden questions through the RAG pipeline.

    Args:
        rag_engine:         live RagEngine instance
        model_id:           Bedrock model ID for generation + judging
        n_results:          top-K to retrieve per question
        use_reranker:       whether to apply cross-encoder reranking
        progress_callback:  optional callable(current, total, question_id)

    Returns:
        summary dict with per-question results and aggregate averages,
        also written to evaluation_report.json.
    """
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(Path(__file__).parent / ".env"), override=True)

    from evaluator import compute_retrieval_metrics

    golden_items = json.loads(GOLDEN_PATH.read_text())
    total = len(golden_items)

    # ── LangSmith setup ──────────────────────────────────────────────────────
    ls_client  = None
    ls_dataset = None
    ls_run_ids: dict[str, str] = {}   # question_id → langsmith_run_id

    api_key     = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    ls_project  = os.getenv("LANGCHAIN_PROJECT", "aetherrag-poc")
    ls_tracing  = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    if api_key:
        try:
            from langsmith import Client
            ls_client  = Client(api_key=api_key)
            ls_dataset = _get_or_create_dataset(ls_client, golden_items)
            logger.info("LangSmith dataset ready.")
        except Exception as e:
            logger.warning(f"LangSmith setup failed (will continue without it): {e}")

    # ── Per-question loop ────────────────────────────────────────────────────
    results = []

    for idx, item in enumerate(golden_items):
        qid          = item["id"]
        question     = item["question"]
        ground_truth = item["ground_truth"]
        qtype        = item["question_type"]

        if progress_callback:
            progress_callback(idx + 1, total, qid)

        logger.info(f"Evaluating {qid}/{total}: {question[:60]}...")

        # 1. RAG retrieval
        try:
            retrieval_result = rag_engine.query_context(
                question, n_results=n_results, use_reranker=use_reranker
            )
            contexts = retrieval_result["chunks"]
            n_bm25   = retrieval_result["n_bm25"]
        except Exception as e:
            logger.error(f"{qid} retrieval failed: {e}")
            results.append({"id": qid, "question": question, "error": str(e)})
            continue

        # 2. Generation
        try:
            answer = ""
            for tok in rag_engine.generate_response_stream(
                model_id=model_id, query=question, contexts=contexts
            ):
                answer += tok
        except Exception as e:
            logger.error(f"{qid} generation failed: {e}")
            results.append({"id": qid, "question": question, "error": str(e)})
            continue

        # 3. Ground-truth string metrics (no LLM cost)
        gt_keywords  = [w for w in ground_truth.lower().split() if len(w) > 3]
        r_metrics    = compute_retrieval_metrics(contexts, question, n_bm25, gt_keywords)
        exact_match  = compute_exact_match(answer, ground_truth)
        token_f1     = compute_token_f1(answer, ground_truth)

        # 4. LLM-as-judge
        g_metrics = _llm_metrics(
            question, answer, ground_truth, contexts,
            rag_engine.bedrock_client, model_id,
        )

        all_metrics = {
            **r_metrics,
            "exact_match":  exact_match,
            "token_f1":     token_f1,
            **g_metrics,
        }

        # 5. Post to LangSmith
        run_id = None
        if ls_client and ls_tracing:
            try:
                from langsmith import traceable
                from langsmith.run_helpers import get_current_run_tree

                rid_box: list = [None]

                @traceable(
                    name=f"golden_eval_{qid}",
                    project_name=ls_project,
                    tags=["golden", "batch-eval", qtype],
                    metadata={"question_id": qid, "question_type": qtype,
                               "source_page": item["source_page"]},
                )
                def _traced(q: str, a: str, gt: str) -> dict:
                    try:
                        t = get_current_run_tree()
                        if t:
                            rid_box[0] = str(t.id)
                    except Exception:
                        pass
                    return {"question": q, "answer": a, "ground_truth": gt,
                            "metrics": all_metrics}

                _traced(question, answer, ground_truth)
                run_id = rid_box[0]

                if run_id:
                    for key, val in all_metrics.items():
                        if isinstance(val, (int, float)):
                            ls_client.create_feedback(
                                run_id=run_id, key=key, score=float(val),
                                feedback_source_type="api",
                                source_run_id=str(uuid.uuid4()),
                            )
                    ls_run_ids[qid] = run_id
            except Exception as e:
                logger.warning(f"{qid} LangSmith trace failed: {e}")

        results.append({
            "id":            qid,
            "question":      question,
            "ground_truth":  ground_truth,
            "answer":        answer,
            "question_type": qtype,
            "source_page":   item["source_page"],
            "metrics":       all_metrics,
            "langsmith_run_id": run_id,
        })

    # ── Aggregate averages ───────────────────────────────────────────────────
    valid = [r for r in results if "metrics" in r]
    metric_keys = list(valid[0]["metrics"].keys()) if valid else []

    averages = {
        k: round(sum(r["metrics"].get(k, 0) for r in valid) / len(valid), 3)
        for k in metric_keys
    } if valid else {}

    # Breakdown by question_type
    qtypes = {r["question_type"] for r in valid}
    by_type = {}
    for qt in sorted(qtypes):
        group = [r for r in valid if r["question_type"] == qt]
        by_type[qt] = {
            "count": len(group),
            **{k: round(sum(r["metrics"].get(k, 0) for r in group) / len(group), 3)
               for k in metric_keys},
        }

    report = {
        "run_timestamp":  datetime.now(timezone.utc).isoformat(),
        "model_id":       model_id,
        "total_questions": total,
        "evaluated":      len(valid),
        "averages":       averages,
        "by_question_type": by_type,
        "results":        results,
        "langsmith_project": ls_project,
        "langsmith_dataset": DATASET_NAME,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info(f"Report written to {REPORT_PATH}")

    return report
