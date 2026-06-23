"""Build final project summary files from existing result artifacts.

The script is intentionally analysis-only. It reads committed metrics and
summary files, then writes the final machine-readable and Markdown summaries.
It does not import model implementations, load weights, train, or run inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


POSITIONING_EN = "A Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking."
POSITIONING_ZH = "基于 Jittor 的 RankRAG 风格大模型重排序轻量复现与实证分析。"
MAIN_METHODS = [
    "BM25",
    "Jittor MLP",
    "Jittor TextCNN",
    "Qwen2.5-1.5B zero-shot reranker",
    "Qwen2.5-1.5B LoRA 10k-rerun",
    "Cross-Encoder",
]
DOWNSTREAM_RUNS = [
    (
        "Qwen2.5-1.5B + Original Prompt",
        "outputs/downstream_rag_eval/downstream_rag_eval_results.json",
        "outputs/downstream_rag_eval",
    ),
    (
        "Qwen2.5-1.5B + Strict Short-Answer Prompt",
        "outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/downstream_rag_eval_results.json",
        "outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt",
    ),
    (
        "Qwen2.5-7B + Original Prompt",
        "outputs/downstream_rag_eval_qwen2_7b/downstream_rag_eval_results.json",
        "outputs/downstream_rag_eval_qwen2_7b",
    ),
    (
        "Qwen2.5-7B + Strict Short-Answer Prompt",
        "outputs/downstream_rag_eval_qwen2_7b_strict_prompt/downstream_rag_eval_results.json",
        "outputs/downstream_rag_eval_qwen2_7b_strict_prompt",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json", default="outputs/final_results_summary.json")
    parser.add_argument("--output-csv", default="outputs/final_results_summary.csv")
    parser.add_argument("--output-doc", default="docs/final_results.md")
    return parser.parse_args()


def read_json(root: Path, rel_path: str) -> Any:
    return json.loads((root / rel_path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "not recorded"


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return str(value)


def metric_row(method: str, metrics: dict[str, Any], source_path: str) -> dict[str, Any]:
    return {
        "method": method,
        "recall_at_1": metrics.get("recall_at_1", metrics.get("recall@1")),
        "recall_at_3": metrics.get("recall_at_3", metrics.get("recall@3")),
        "recall_at_5": metrics.get("recall_at_5", metrics.get("recall@5")),
        "ndcg_at_5": metrics.get("ndcg_at_5", metrics.get("ndcg@5")),
        "mrr": metrics.get("mrr"),
        "pairwise_accuracy": metrics.get("pairwise_accuracy"),
        "source_path": source_path,
    }


def markdown_table(headers: list[str], rows: list[list[Any]], align_right_from: int = 1) -> str:
    aligns = ["---"] * len(headers)
    for idx in range(align_right_from, len(headers)):
        aligns[idx] = "---:"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def candidate_pool_signature(root: Path) -> dict[str, Any]:
    msmarco_path = root / "data/processed/msmarco_medium/test.jsonl"
    lora_path = root / "data/processed/lora_qwen_1_5b_10k/test_queries.jsonl"
    query_count = 0
    pair_count = 0
    positives_per_query: set[int] = set()
    with msmarco_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            query_count += 1
            candidates = item.get("candidates", [])
            pair_count += len(candidates)
            positives_per_query.add(sum(1 for candidate in candidates if int(candidate.get("label", 0)) == 1))
    return {
        "msmarco_medium_test_path": "data/processed/msmarco_medium/test.jsonl",
        "msmarco_medium_test_sha256": sha256(msmarco_path),
        "lora_test_path": "data/processed/lora_qwen_1_5b_10k/test_queries.jsonl",
        "lora_test_sha256": sha256(lora_path),
        "query_count": query_count,
        "pair_count": pair_count,
        "positive_labels_per_query": sorted(positives_per_query),
        "candidate_pool_note": "All six main reranking methods use the same 500-query / 4044-pair candidate pool.",
    }


def load_main_results(root: Path) -> list[dict[str, Any]]:
    cost = read_json(root, "outputs/cost_effectiveness_table.json")
    rows = []
    for item in cost["methods"]:
        if item["method"] in MAIN_METHODS:
            rows.append(
                {
                    "method": item["method"],
                    "recall_at_1": item["recall_at_1"],
                    "recall_at_3": item["recall_at_3"],
                    "recall_at_5": item["recall_at_5"],
                    "ndcg_at_5": item["ndcg_at_5"],
                    "mrr": item["mrr"],
                    "pairwise_accuracy": item["pairwise_accuracy"],
                    "effectiveness_comparable": item["effectiveness_comparable"],
                    "query_count": item["query_count"],
                    "pair_count": item["pair_count"],
                    "source_path": "outputs/cost_effectiveness_table.json",
                }
            )
    return rows


def load_alignment(root: Path) -> list[dict[str, Any]]:
    rows = read_json(root, "outputs/l2_msmarco_medium_results.json")
    by_key = {(row["Method"], row["Framework"]): row for row in rows}
    output = []
    for model in ["MLP", "TextCNN"]:
        torch_row = by_key[(model, "PyTorch")]
        jittor_row = by_key[(model, "Jittor")]
        for framework, row, baseline in [("PyTorch", torch_row, None), ("Jittor", jittor_row, torch_row)]:
            output.append(
                {
                    "model": model,
                    "framework": framework,
                    "recall_at_1": row["recall@1"],
                    "recall_at_3": row["recall@3"],
                    "recall_at_5": row["recall@5"],
                    "ndcg_at_5": row["ndcg@5"],
                    "mrr": row["mrr"],
                    "delta_r1_vs_pytorch": None if baseline is None else row["recall@1"] - baseline["recall@1"],
                    "delta_ndcg5_vs_pytorch": None if baseline is None else row["ndcg@5"] - baseline["ndcg@5"],
                    "source_path": "outputs/l2_msmarco_medium_results.json",
                }
            )
    return output


def load_lora_e1(root: Path) -> list[dict[str, Any]]:
    payload = read_json(root, "outputs/lora_ablation_results.json")
    output = []
    for row in payload["runs"]:
        output.append(
            {
                "experiment_name": row["experiment_name"],
                "training_pairs": row["training_pairs"],
                "effective_epochs": row["effective_epochs"],
                "max_steps": row["max_steps"],
                "recall_at_1": row["recall_at_1"],
                "recall_at_3": row["recall_at_3"],
                "recall_at_5": row["recall_at_5"],
                "ndcg_at_5": row["ndcg_at_5"],
                "mrr": row["mrr"],
                "pairwise_accuracy": row["pairwise_accuracy"],
                "train_runtime_sec": row["train_runtime_sec"],
                "eval_runtime_sec": row["eval_runtime_sec"],
                "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
                "hardware": row["hardware"],
                "source_path": "outputs/lora_ablation_results.json",
            }
        )
    return output


def load_lora_e2(root: Path) -> list[dict[str, Any]]:
    payload = read_json(root, "outputs/lora_scoring_ablation_results.json")
    return [{**row, "source_path": "outputs/lora_scoring_ablation_results.json"} for row in payload["runs"]]


def load_downstream(root: Path) -> list[dict[str, Any]]:
    output = []
    for run_name, path, output_dir in DOWNSTREAM_RUNS:
        payload = read_json(root, path)
        for item in payload["methods"]:
            metrics = item["metrics"]
            output.append(
                {
                    "run_name": run_name,
                    "method": item["method"],
                    "questions": metrics["questions"],
                    "top_k": metrics["top_k"],
                    "gold_in_context": metrics.get("gold_in_context"),
                    "answer_hit": metrics["answer_hit"],
                    "exact_match": metrics["exact_match"],
                    "token_f1": metrics["token_f1"],
                    "successes": metrics["successes"],
                    "source_path": path,
                    "output_dir": output_dir,
                }
            )
    return output


def load_error_summary(root: Path) -> dict[str, Any]:
    payload = read_json(root, "outputs/error_taxonomy_summary.json")
    return {**payload, "source_path": "outputs/error_taxonomy_summary.json"}


def load_resource_summary(root: Path) -> dict[str, Any]:
    payload = read_json(root, "outputs/cost_effectiveness_table.json")
    return {
        "title": payload["title"],
        "strictly_comparable_method_count": payload["comparability"]["strictly_comparable_method_count"],
        "figure_type": payload["comparability"]["figure_type"],
        "methods": [
            {
                "method": row["method"],
                "recall_at_1": row["recall_at_1"],
                "ndcg_at_5": row["ndcg_at_5"],
                "local_task_training": row["local_task_training"],
                "pretrained_semantic_ability": row["pretrained_semantic_ability"],
                "training_runtime_sec": row["training_runtime_sec"],
                "evaluation_runtime_sec": row["evaluation_runtime_sec"],
                "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
                "evaluation_hardware": row["evaluation_hardware"],
                "resource_strictly_comparable": row["resource_strictly_comparable"],
            }
            for row in payload["methods"]
        ],
        "source_path": "outputs/cost_effectiveness_table.json",
    }


def build_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for section, items in [
        ("main_reranking", summary["main_reranking_results"]),
        ("pytorch_jittor_alignment", summary["pytorch_jittor_alignment"]),
        ("lora_data_size_ablation", summary["lora_data_size_ablation"]),
        ("scoring_method_ablation", summary["scoring_method_ablation"]),
        ("downstream_rag", summary["downstream_rag_results"]),
    ]:
        for item in items:
            row = {"section": section}
            row.update(item)
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_doc(summary: dict[str, Any]) -> str:
    main_rows = [
        [
            row["method"],
            fmt(row["recall_at_1"], 3),
            fmt(row["recall_at_3"], 3),
            fmt(row["recall_at_5"], 3),
            fmt(row["ndcg_at_5"], 4),
            fmt(row["mrr"], 4),
            fmt(row["pairwise_accuracy"], 4),
        ]
        for row in summary["main_reranking_results"]
    ]
    align_rows = [
        [
            row["framework"],
            row["model"],
            fmt(row["recall_at_1"], 3),
            fmt(row["recall_at_3"], 3),
            fmt(row["recall_at_5"], 3),
            fmt(row["ndcg_at_5"], 4),
            fmt(row["mrr"], 4),
            "baseline" if row["delta_r1_vs_pytorch"] is None else fmt(row["delta_r1_vs_pytorch"], 4),
            "baseline" if row["delta_ndcg5_vs_pytorch"] is None else fmt(row["delta_ndcg5_vs_pytorch"], 4),
        ]
        for row in summary["pytorch_jittor_alignment"]
    ]
    e1_rows = [
        [
            row["experiment_name"],
            row["training_pairs"],
            fmt(row["effective_epochs"], 4),
            row["max_steps"],
            fmt(row["recall_at_1"], 3),
            fmt(row["recall_at_3"], 3),
            fmt(row["recall_at_5"], 3),
            fmt(row["ndcg_at_5"], 4),
            fmt(row["mrr"], 4),
            fmt(row["pairwise_accuracy"], 4),
            fmt(row["train_runtime_sec"], 3),
            fmt(row["peak_gpu_memory_mib"], 1),
        ]
        for row in summary["lora_data_size_ablation"]
    ]
    e2_rows = [
        [
            row["scoring_method"],
            fmt(row["recall_at_1"], 3),
            fmt(row["recall_at_3"], 3),
            fmt(row["recall_at_5"], 3),
            fmt(row["ndcg_at_5"], 4),
            fmt(row["mrr"], 4),
            fmt(row["inference_runtime_sec"], 3),
            fmt(row["parse_failure_rate"], 4),
            fmt(row["query_tie_rate"], 4),
        ]
        for row in summary["scoring_method_ablation"]
    ]
    downstream_rows = [
        [
            row["run_name"],
            row["method"],
            fmt(row["gold_in_context"], 3),
            fmt(row["answer_hit"], 3),
            fmt(row["exact_match"], 3),
            fmt(row["token_f1"], 4),
            row["successes"],
        ]
        for row in summary["downstream_rag_results"]
    ]
    error = summary["error_analysis_summary"]
    error_rows = [[key, value] for key, value in error["cases_by_primary_error_type"].items()]
    resource_rows = [
        [
            row["method"],
            fmt(row["recall_at_1"], 3),
            fmt(row["ndcg_at_5"], 4),
            row["local_task_training"],
            "yes" if row["pretrained_semantic_ability"] else "no",
            fmt(row["training_runtime_sec"], 3),
            fmt(row["evaluation_runtime_sec"], 3),
            fmt(row["peak_gpu_memory_mib"], 1),
            row["evaluation_hardware"],
            str(row["resource_strictly_comparable"]).lower(),
        ]
        for row in summary["resource_effectiveness_summary"]["methods"]
    ]
    return f"""# Final Results

This file is generated from committed result artifacts by `scripts/build_final_project_summary.py`. The machine-readable source of truth is `outputs/final_results_summary.json`.

## 1. Main Reranking Results

The main reranking table uses the same MS MARCO medium subset: 500 queries, 4044 query-passage pairs, the same candidate pool, and the same gold definition. The LoRA row uses the Stage E `10k-rerun` result, not the historical 10k run.

{markdown_table(["Method", "R@1", "R@3", "R@5", "NDCG@5", "MRR", "Pairwise Acc"], main_rows)}

## 2. PyTorch/Jittor Alignment

MLP and TextCNN are lightweight alignment baselines. They are not core models from the original RankRAG paper, and exact number-by-number equality is not expected.

{markdown_table(["Framework", "Model", "R@1", "R@3", "R@5", "NDCG@5", "MRR", "Delta R@1 vs PyTorch", "Delta NDCG@5 vs PyTorch"], align_rows)}

## 3. LoRA Data-Size Ablation

E1 fixes the optimizer-step budget at 800 steps. It measures the effect of adding training-data diversity, not a fixed-epoch comparison. The 1k, 3k, and 10k-rerun subsets are nested.

{markdown_table(["Run", "Training pairs", "Effective epochs", "Max steps", "R@1", "R@3", "R@5", "NDCG@5", "MRR", "Pairwise Acc", "Train runtime sec", "Peak GPU MiB"], e1_rows)}

## 4. Scoring-Method Ablation

E2 compares scoring rules for the same 10k-rerun LoRA adapter. These are not three different models.

{markdown_table(["Scoring method", "R@1", "R@3", "R@5", "NDCG@5", "MRR", "Runtime sec", "Parse failure rate", "Query tie rate"], e2_rows)}

`generate_parse` is much slower and produces many ties. `relevant_logprob` and `logprob_margin` have nearly identical R@1 but differ slightly across the full ranking metrics.

## 5. Downstream RAG Results

Downstream RAG uses 50 questions and top-3 contexts. Higher evidence availability does not guarantee the generator will use that evidence correctly.

{markdown_table(["Generator / Prompt", "Reranker", "Gold@3", "Answer Hit", "EM", "Token F1", "Successes"], downstream_rows, align_right_from=2)}

## 6. Error Analysis Summary

Stage G uses 30 unique stratified diagnostic queries: six cases from each A/B/C/D/E selection bucket. This is not an unbiased estimate over all 500 queries.

{markdown_table(["Primary error type", "Count"], error_rows)}

- High-confidence cases: {error["cases_by_confidence"]["high"]}
- Medium-confidence cases: {error["cases_by_confidence"]["medium"]}
- Self-consistency sample size: {error["self_consistency_sample_size"]}
- Primary label agreement: {fmt(error["primary_label_agreement"], 2)}
- Secondary label agreement: {fmt(error["secondary_label_agreement"], 2)}
- Note: self-consistency is single-annotator repeated annotation, not inter-annotator agreement.

## 7. Resource and Effectiveness Profile

The effectiveness metrics are comparable across the six main methods. Resource records are heterogeneous, so they are a profile rather than a strict speed benchmark.

{markdown_table(["Method", "R@1", "NDCG@5", "Training", "Pretrained semantics", "Train runtime", "Eval runtime", "Peak VRAM", "Hardware", "Strict resource comparable"], resource_rows)}

Strictly resource-comparable method count: {summary["resource_effectiveness_summary"]["strictly_comparable_method_count"]}.

## 8. Source Files

The main source files used to build this summary are recorded in `outputs/final_results_summary.json` under `source_files`.
"""


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    source_files = {
        "main_reranking": "outputs/cost_effectiveness_table.json",
        "pytorch_jittor_alignment": "outputs/l2_msmarco_medium_results.json",
        "lora_data_size_ablation": "outputs/lora_ablation_results.json",
        "scoring_method_ablation": "outputs/lora_scoring_ablation_results.json",
        "downstream_rag": [path for _, path, _ in DOWNSTREAM_RUNS],
        "error_analysis": "outputs/error_taxonomy_summary.json",
        "resource_effectiveness": "outputs/cost_effectiveness_table.json",
    }
    signature = candidate_pool_signature(root)
    summary = {
        "project_name": "RankRAG-Jittor",
        "project_positioning": {
            "en": POSITIONING_EN,
            "zh": POSITIONING_ZH,
        },
        "git_commit": git_commit(root),
        "dataset": "MS MARCO medium subset",
        "test_query_count": signature["query_count"],
        "pair_count": signature["pair_count"],
        "candidate_pool_signature": signature,
        "main_reranking_results": load_main_results(root),
        "pytorch_jittor_alignment": load_alignment(root),
        "lora_data_size_ablation": load_lora_e1(root),
        "scoring_method_ablation": load_lora_e2(root),
        "downstream_rag_results": load_downstream(root),
        "error_analysis_summary": load_error_summary(root),
        "resource_effectiveness_summary": load_resource_summary(root),
        "limitations": [
            "This is a lightweight reproduction and empirical analysis, not a full RankRAG reproduction.",
            "The main ranking evaluation uses a 500-query MS MARCO medium subset.",
            "Downstream RAG uses 50 questions.",
            "Error analysis uses 30 stratified diagnostic cases.",
            "Resource measurements come from heterogeneous environments and are not a strict speed benchmark.",
            "Base model weights and LoRA adapters are not versioned in Git.",
        ],
        "source_files": source_files,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_json = root / args.output_json
    output_csv = root / args.output_csv
    output_doc = root / args.output_doc
    write_json(output_json, summary)
    build_csv(output_csv, summary)
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_doc.write_text(build_doc(summary), encoding="utf-8")
    print(json.dumps({"status": "ready", "output_json": args.output_json, "output_doc": args.output_doc}, indent=2))


if __name__ == "__main__":
    main()
