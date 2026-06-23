"""Aggregate formal E2 LoRA scoring-method ablation outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


RUNS = [
    ("generate_parse", Path("outputs/lora_scoring_ablation/generate_parse")),
    ("relevant_logprob", Path("outputs/lora_scoring_ablation/relevant_logprob")),
    ("logprob_margin", Path("outputs/lora_scoring_ablation/logprob_margin")),
]


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(metrics: dict[str, Any], method: str, output_dir: Path) -> dict[str, Any]:
    return {
        "scoring_method": method,
        "recall_at_1": metrics.get("recall@1"),
        "recall_at_3": metrics.get("recall@3"),
        "recall_at_5": metrics.get("recall@5"),
        "ndcg_at_5": metrics.get("ndcg@5"),
        "mrr": metrics.get("mrr"),
        "pairwise_accuracy": metrics.get("pairwise_accuracy"),
        "inference_runtime_sec": metrics.get("inference_time_sec"),
        "parse_failure_count": metrics.get("parse_failure_count"),
        "parse_failure_rate": metrics.get("parse_failure_rate"),
        "query_tie_rate": metrics.get("query_tie_rate"),
        "average_candidate_tie_rate": metrics.get("average_candidate_tie_rate"),
        "max_tie_group_size": metrics.get("max_tie_group_size"),
        "num_queries": metrics.get("num_queries"),
        "num_pairs": metrics.get("num_pairs"),
        "git_commit": metrics.get("git_commit"),
        "hardware": metrics.get("device_name"),
        "adapter_dir": metrics.get("adapter_dir"),
        "test_path": metrics.get("test_path"),
        "output_dir": str(output_dir),
        "status": "ready",
    }


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    if len(rows) != 3:
        errors.append("expected three scoring-method runs")
    commits = {row.get("git_commit") for row in rows}
    if len(commits) != 1:
        errors.append(f"E2 runs do not share one commit: {sorted(commits)}")
    adapters = {row.get("adapter_dir") for row in rows}
    if len(adapters) != 1:
        errors.append(f"E2 runs do not share one adapter: {sorted(adapters)}")
    tests = {row.get("test_path") for row in rows}
    if len(tests) != 1:
        errors.append(f"E2 runs do not share one test path: {sorted(tests)}")
    for row in rows:
        for key in ["recall_at_1", "recall_at_3", "recall_at_5", "ndcg_at_5", "mrr", "pairwise_accuracy"]:
            value = row.get(key)
            if value is None or not (0.0 <= float(value) <= 1.0):
                errors.append(f"{row['scoring_method']}: invalid {key}={value}")
    return errors


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    return "" if value is None else str(value)


def write_markdown(rows: list[dict[str, Any]], errors: list[str]) -> None:
    lines = [
        "# LoRA Scoring-Method Ablation Results",
        "",
        "Status: " + ("ready" if not errors else "failed"),
        "",
        "| Method | R@1 | R@3 | R@5 | NDCG@5 | MRR | Pairwise Acc. | Runtime s | Parse fail % | Tie rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scoring_method']} | {fmt(row.get('recall_at_1'))} | {fmt(row.get('recall_at_3'))} | "
            f"{fmt(row.get('recall_at_5'))} | {fmt(row.get('ndcg_at_5'))} | {fmt(row.get('mrr'))} | "
            f"{fmt(row.get('pairwise_accuracy'))} | {fmt(row.get('inference_runtime_sec'))} | "
            f"{fmt(row.get('parse_failure_rate'))} | {fmt(row.get('average_candidate_tie_rate'))} |"
        )
    if errors:
        lines += ["", "## Validation Errors", ""]
        lines += [f"- {error}" for error in errors]
    Path("outputs/lora_scoring_ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["scoring_method"] for row in rows]
    metrics = [("R@1", "recall_at_1"), ("MRR", "mrr"), ("NDCG@5", "ndcg_at_5")]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = range(len(labels))
    width = 0.24
    for offset, (label, key) in enumerate(metrics):
        positions = [value + (offset - 1) * width for value in x]
        ax.bar(positions, [row[key] for row in rows], width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric")
    ax.set_title("LoRA reranker scoring-method ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lora_scoring_ablation_results.png", dpi=220)
    plt.close(fig)


def write_analysis(rows: list[dict[str, Any]], tokenization: dict[str, Any] | None) -> None:
    by_method = {row["scoring_method"]: row for row in rows}
    margin = by_method.get("logprob_margin", {})
    relevant = by_method.get("relevant_logprob", {})
    generate = by_method.get("generate_parse", {})

    def delta(row_a: dict[str, Any], row_b: dict[str, Any], key: str) -> str:
        if row_a.get(key) is None or row_b.get(key) is None:
            return "n/a"
        return f"{float(row_a[key]) - float(row_b[key]):+.4f}"

    lines = [
        "# Stage E2: LoRA Scoring-Method Ablation",
        "",
        "E2 compares inference-time scoring rules for the same 10k-rerun LoRA adapter. It is not a new model-training ablation.",
        "",
        "## Methods",
        "",
        "- `generate_parse`: deterministic generation of Relevant / Irrelevant, parsed into binary scores.",
        "- `relevant_logprob`: score = log P(Relevant).",
        "- `logprob_margin`: score = log P(Relevant) - log P(Irrelevant), the backward-compatible default.",
        "",
        "## Observations",
        "",
        f"- generate_parse tie rate: {fmt(generate.get('average_candidate_tie_rate'))}; parse failure rate: {fmt(generate.get('parse_failure_rate'))}.",
        f"- relevant_logprob R@1 delta vs margin: {delta(relevant, margin, 'recall_at_1')}.",
        f"- logprob_margin R@1: {fmt(margin.get('recall_at_1'))}; MRR: {fmt(margin.get('mrr'))}; NDCG@5: {fmt(margin.get('ndcg_at_5'))}.",
        "",
        "The log-probability methods compute the full label token sequence probability rather than only the first token.",
    ]
    if tokenization:
        lines += [
            "",
            "## Label Tokenization",
            "",
            f"- Relevant: ids={tokenization.get('Relevant', {}).get('token_ids')}, length={tokenization.get('Relevant', {}).get('token_count')}",
            f"- Irrelevant: ids={tokenization.get('Irrelevant', {}).get('token_ids')}, length={tokenization.get('Irrelevant', {}).get('token_count')}",
        ]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/scoring_ablation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = []
    tokenization = None
    for method, output_dir in RUNS:
        metrics_path = output_dir / "qwen_1_5b_lora_metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Missing metrics for {method}: {metrics_path}")
        metrics = read_json(metrics_path)
        rows.append(normalize(metrics, method, output_dir))
        if tokenization is None:
            tokenization = metrics.get("label_tokenization")
    errors = validate(rows)
    payload = {"status": "ready" if not errors else "failed", "runs": rows, "validation_errors": errors, "label_tokenization": tokenization}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/lora_scoring_ablation_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, Path("outputs/lora_scoring_ablation_results.csv"))
    write_markdown(rows, errors)
    write_plot(rows)
    write_analysis(rows, tokenization)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
