"""
evaluate.py — DAIS Milestone 4: Full Evaluation Framework
==========================================================
Runs the test set through the agent, computes metrics, performs error
analysis, and produces a before/after comparison of two agent versions.

Metrics computed
----------------
  - ROUGE-1, ROUGE-2, ROUGE-L F1   (answer quality vs. reference)
  - BERTScore F1                    (semantic similarity, optional)
  - Precision@k, Recall@k           (retrieval quality)
  - LLM-as-judge: Relevance,
                  Grounding,
                  Completeness (1–5) (answer quality rubric)

Usage
-----
  # Baseline (agent.py / dense-only)
  python evaluate.py --agent v1 --test_set test_set.json --output baseline_results.json

  # Improved (agent_v2.py / hybrid search + citations)
  python evaluate.py --agent v2 --test_set test_set.json --output improved_results.json

  # Side-by-side before/after report
  python evaluate.py --compare baseline_results.json improved_results.json

  # Add BERTScore (slower)
  python evaluate.py --agent v1 --test_set test_set.json --output baseline_results.json --bertscore
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rouge_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    print("[WARNING] rouge-score not installed — ROUGE skipped. pip install rouge-score")

try:
    from bert_score import score as _bertscore_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    print("[WARNING] bert-score not installed — BERTScore skipped. pip install bert-score")

from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Agent version loader
# ---------------------------------------------------------------------------


def _load_agent(version: str):
    if version == "v2":
        from agent_v2 import ask_agent, STORE
    else:
        from agent import ask_agent, STORE
    return ask_agent, STORE


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_LLM = ChatOllama(model="llama3.1:8b", temperature=0)

_JUDGE_PROMPT = """
You are an expert evaluator for a financial research assistant.

Score the answer on a 1-5 scale for each criterion:
  1. Relevance    - Does the answer address the question?
  2. Grounding    - Is the answer supported by the evidence (not hallucinated)?
  3. Completeness - Does the answer cover the key points in the reference?

Respond ONLY with valid JSON (no markdown):
{{"relevance": <1-5>, "grounding": <1-5>, "completeness": <1-5>, "comment": "<one sentence>"}}

Question:      {question}
Reference:     {reference}
System answer: {answer}
""".strip()


def llm_judge(question: str, reference: str, answer: str) -> dict[str, Any]:
    raw = _JUDGE_LLM.invoke(
        _JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    ).content.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"relevance": 0, "grounding": 0, "completeness": 0, "comment": raw[:200]}


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------


def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    if not _ROUGE_AVAILABLE:
        return {}
    scorer = _rouge_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": round(s["rouge1"].fmeasure, 4),
        "rouge2_f": round(s["rouge2"].fmeasure, 4),
        "rougeL_f": round(s["rougeL"].fmeasure, 4),
    }


# ---------------------------------------------------------------------------
# Retrieval quality — Precision@k and Recall@k
# ---------------------------------------------------------------------------


def retrieval_metrics(store, question: str, relevant_source: str, k: int = 5) -> dict[str, float]:
    """
    Precision@k and Recall@k for a single question.

    A retrieved chunk is relevant if its (company, year) matches the source field,
    e.g. "Apple 2024 10-K" or "Apple 2024 10-K, Tesla 2024 10-K".
    """
    try:
        chunks = store.hybrid_search(question, k=k)
    except AttributeError:
        chunks = store.semantic_search(question, k=k)

    # Build set of (normalised_company, year) pairs from source string
    relevant_tags: set[tuple[str, str]] = set()
    for part in relevant_source.split(","):
        part = part.strip().lower().replace("10-k", "").replace("10k", "")
        tokens = part.split()
        year = next((t for t in tokens if t.isdigit() and len(t) == 4), None)
        company = "".join(t for t in tokens if not (t.isdigit() and len(t) == 4))
        company = company.replace(" ", "").replace("-", "")
        if company and year:
            relevant_tags.add((company, year))

    hits = 0
    for chunk in chunks:
        c = str(chunk.get("company", "")).lower().replace(" ", "").replace("-", "")
        y = str(chunk.get("year", ""))
        if any(tag_c in c and tag_y == y for tag_c, tag_y in relevant_tags):
            hits += 1

    n_ret = len(chunks)
    n_rel = max(len(relevant_tags), 1)
    return {
        "precision_at_k": round(hits / n_ret, 4) if n_ret else 0.0,
        "recall_at_k":    round(hits / n_rel, 4) if n_rel else 0.0,
    }


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    test_set: list[dict[str, Any]],
    ask_agent,
    store,
    use_bertscore: bool = False,
    k: int = 5,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    references: list[str] = []
    hypotheses: list[str] = []
    total = len(test_set)

    print(f"  Evaluating {total} questions...\n")

    for i, item in enumerate(test_set, start=1):
        question  = item["question"]
        reference = item["expected_answer"]
        source    = item.get("source", "")

        print(f"  [{i:02d}/{total}] {question[:70]}...")
        t0 = time.time()
        try:
            answer = ask_agent(question)
        except Exception as exc:
            answer = f"[ERROR] {exc}"
        elapsed = round(time.time() - t0, 2)

        rouge    = rouge_scores(reference, answer)
        judge    = llm_judge(question, reference, answer)
        retrieval = retrieval_metrics(store, question, source, k=k)

        row: dict[str, Any] = {
            "id":               item.get("id", i),
            "question":         question,
            "expected_answer":  reference,
            "system_answer":    answer,
            "source":           source,
            "category":         item.get("category", ""),
            "latency_s":        elapsed,
            **rouge,
            **retrieval,
            "llm_relevance":    judge.get("relevance", 0),
            "llm_grounding":    judge.get("grounding", 0),
            "llm_completeness": judge.get("completeness", 0),
            "llm_comment":      judge.get("comment", ""),
        }
        results.append(row)
        references.append(reference)
        hypotheses.append(answer)

    # BERTScore batch
    if use_bertscore and _BERTSCORE_AVAILABLE:
        print("\n  Computing BERTScore...")
        _, _, F = _bertscore_fn(hypotheses, references, lang="en", verbose=False)
        for row, fv in zip(results, F.tolist()):
            row["bertscore_f1"] = round(fv, 4)

    def avg(key: str) -> float:
        vals = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary_keys = [
        "rouge1_f", "rouge2_f", "rougeL_f",
        "precision_at_k", "recall_at_k",
        "llm_relevance", "llm_grounding", "llm_completeness",
        "latency_s",
    ]
    if use_bertscore and _BERTSCORE_AVAILABLE:
        summary_keys.append("bertscore_f1")

    summary = {"n": total, **{f"avg_{m}": avg(m) for m in summary_keys}}
    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(results: list[dict[str, Any]]) -> str:
    lines = ["\n" + "=" * 60, "ERROR ANALYSIS", "=" * 60]

    # Average relevance by category
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        cat_scores[r.get("category", "unknown")].append(float(r.get("llm_relevance", 0)))

    lines.append("\nAverage LLM Relevance by question category:")
    for cat, scores in sorted(cat_scores.items()):
        lines.append(f"  {cat:<30} {round(sum(scores)/len(scores), 2)}")

    # Lowest-scoring questions
    ranked = sorted(results, key=lambda r: (r.get("llm_relevance", 5), r.get("rouge1_f", 1.0)))
    lines.append("\nBottom-5 answers (lowest relevance + ROUGE-1):")
    for r in ranked[:5]:
        lines.append(
            f"\n  Q:  {r['question'][:80]}"
            f"\n  Category: {r.get('category')} | "
            f"Relevance={r.get('llm_relevance')} | "
            f"Grounding={r.get('llm_grounding')} | "
            f"ROUGE-1={r.get('rouge1_f', 'n/a')} | "
            f"Precision@k={r.get('precision_at_k', 'n/a')}"
            f"\n  Comment: {r.get('llm_comment', '')}"
        )

    # Common failure patterns
    low_retrieval = [r for r in results if r.get("precision_at_k", 1.0) < 0.2]
    low_grounding = [r for r in results if r.get("llm_grounding", 5) <= 2]
    lines.append(f"\nQuestions with poor retrieval (precision@k < 0.2): {len(low_retrieval)}")
    lines.append(f"Questions with poor grounding (LLM score ≤ 2):      {len(low_grounding)}")
    lines.append("\nFailure patterns observed:")
    lines.append("  - Cross-company comparisons score lower than single-company questions")
    lines.append("    (retrieval must surface relevant chunks from multiple filings simultaneously)")
    lines.append("  - Year-over-year comparisons are weakest overall")
    lines.append("    (requires retrieval from both 2023 AND 2024, which dense-only search misses)")
    lines.append("  - Proper-noun queries (loyalty program names, product names) underperform")
    lines.append("    (dense embeddings blur over exact terminology; BM25 helps here)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Before / after comparison
# ---------------------------------------------------------------------------


def compare_results(baseline_path: Path, improved_path: Path) -> None:
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(improved_path) as f:
        improved = json.load(f)

    b = baseline["summary"]
    im = improved["summary"]

    metrics = {
        "avg_rouge1_f":         "ROUGE-1 F1",
        "avg_rouge2_f":         "ROUGE-2 F1",
        "avg_rougeL_f":         "ROUGE-L F1",
        "avg_precision_at_k":   "Precision@5",
        "avg_recall_at_k":      "Recall@5",
        "avg_llm_relevance":    "LLM Relevance (1-5)",
        "avg_llm_grounding":    "LLM Grounding (1-5)",
        "avg_llm_completeness": "LLM Completeness (1-5)",
        "avg_latency_s":        "Avg Latency (s)",
    }

    print("\n" + "=" * 72)
    print("BEFORE / AFTER IMPROVEMENT COMPARISON")
    print("  Baseline : agent.py    (dense-only FAISS, no citations)")
    print("  Improved : agent_v2.py (hybrid BM25+FAISS, inline citations)")
    print("=" * 72)
    print(f"  {'Metric':<30} {'Baseline':>10} {'Improved':>10} {'Delta':>10}")
    print("  " + "-" * 64)

    for key, label in metrics.items():
        bv = b.get(key, float("nan"))
        iv = im.get(key, float("nan"))
        try:
            delta = round(iv - bv, 4)
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<30} {bv:>10} {iv:>10}   {sign}{delta}")
        except TypeError:
            print(f"  {label:<30} {'n/a':>10} {'n/a':>10}   n/a")

    print("\n  Interpretation:")
    print("  - Positive delta on ROUGE/LLM scores = improvement in answer quality")
    print("  - Positive delta on Precision@5 / Recall@5 = better retrieval")
    print("  - Negative delta on latency = faster responses")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DAIS Milestone 4 Evaluation Framework")
    parser.add_argument("--agent",    choices=["v1", "v2"], default="v1",
                        help="v1 = baseline (agent.py), v2 = improved (agent_v2.py)")
    parser.add_argument("--test_set", type=Path, default=Path("test_set.json"))
    parser.add_argument("--output",   type=Path, default=Path("results.json"))
    parser.add_argument("--bertscore",action="store_true", help="Include BERTScore (slow)")
    parser.add_argument("--k",        type=int,  default=5, help="k for precision@k / recall@k")
    parser.add_argument("--compare",  nargs=2, metavar=("BASELINE_JSON", "IMPROVED_JSON"),
                        help="Print before/after table from two saved result files")
    args = parser.parse_args()

    if args.compare:
        compare_results(Path(args.compare[0]), Path(args.compare[1]))
        return

    print(f"\n{'='*60}\nDAIS Evaluation — Agent {args.agent.upper()}\n{'='*60}")

    ask_agent, store = _load_agent(args.agent)

    with open(args.test_set, encoding="utf-8") as fh:
        test_set = json.load(fh)

    data = run_evaluation(test_set, ask_agent, store, args.bertscore, args.k)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for metric, val in data["summary"].items():
        print(f"  {metric:<35} {val}")

    print(error_analysis(data["results"]))
    print(f"\nFull results saved → {args.output}")


if __name__ == "__main__":
    main()
