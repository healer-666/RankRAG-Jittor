"""Aggregate Qwen2.5-1.5B LoRA reranker results with existing baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from utils import ensure_parent, read_json, resolve_path, write_json


METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "ndcg@5",
    "ndcg@10",
    "mrr",
    "pairwise_accuracy",
]

LORA_RUNS = [
    {
        "name": "Qwen2.5-1.5B LoRA v1",
        "framework": "PyTorch",
        "training": "LoRA, 5k pairs, lr=1e-4",
        "config": "configs/lora_qwen_1_5b_formal.yaml",
        "metrics": "outputs/lora_qwen_1_5b_formal/qwen_1_5b_lora_metrics.json",
        "summary": "outputs/lora_qwen_1_5b_formal/train_summary.json",
        "role": "tuning run",
    },
    {
        "name": "Qwen2.5-1.5B LoRA v2",
        "framework": "PyTorch",
        "training": "LoRA, 5k pairs, lr=5e-5",
        "config": "configs/lora_qwen_1_5b_lr5e5_s500.yaml",
        "metrics": "outputs/lora_qwen_1_5b_lr5e5_s500/qwen_1_5b_lora_metrics.json",
        "summary": "outputs/lora_qwen_1_5b_lr5e5_s500/train_summary.json",
        "role": "tuning run",
    },
    {
        "name": "Qwen2.5-1.5B LoRA v3",
        "framework": "PyTorch",
        "training": "LoRA, 10k pairs, lr=1e-4",
        "config": "configs/lora_qwen_1_5b_10k_lr1e4_s800.yaml",
        "metrics": "outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_metrics.json",
        "summary": "outputs/lora_qwen_1_5b_10k_lr1e4_s800/train_summary.json",
        "role": "formal result",
    },
]

BASELINE_RUNS = [
    {
        "name": "BM25",
        "framework": "rank_bm25",
        "training": "none",
        "path": "outputs/msmarco_medium_retrieval_baseline_metrics.json",
        "nested": ["bm25", "metrics"],
        "role": "lexical baseline",
    },
    {
        "name": "Jittor MLP",
        "framework": "Jittor",
        "training": "from scratch",
        "path": "outputs/msmarco_medium_jittor_metrics.json",
        "role": "lightweight Jittor reranker",
    },
    {
        "name": "Jittor TextCNN",
        "framework": "Jittor",
        "training": "from scratch",
        "path": "outputs/msmarco_medium_textcnn_jittor_metrics.json",
        "role": "lightweight Jittor reranker",
    },
    {
        "name": "Qwen2.5-0.5B Jittor zero-shot",
        "framework": "JittorLLM",
        "training": "zero-shot",
        "path": "outputs/jittorllm_qwen2_0_5b_full/metrics.json",
        "role": "LLM zero-shot reranker",
    },
    {
        "name": "Qwen2.5-1.5B Jittor zero-shot",
        "framework": "JittorLLM",
        "training": "zero-shot",
        "path": "outputs/jittorllm_qwen2_1_5b_full/metrics.json",
        "role": "LLM zero-shot reranker",
    },
    {
        "name": "Cross-Encoder reference",
        "framework": "sentence-transformers",
        "training": "external pretrained",
        "path": "outputs/msmarco_medium_cross_encoder_metrics.json",
        "role": "external semantic reference",
    },
]


def nested_get(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def load_optional_json(path: str) -> dict[str, Any] | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any] | None, metric: str) -> float | None:
    if not metrics or metric not in metrics:
        return None
    return float(metrics[metric])


def collect_baseline_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in BASELINE_RUNS:
        metrics = load_optional_json(spec["path"])
        if metrics and spec.get("nested"):
            metrics = nested_get(metrics, spec["nested"])
        row = {
            "method": spec["name"],
            "framework": spec["framework"],
            "training": spec["training"],
            "role": spec["role"],
            "status": "ready" if metrics else "pending",
            "train_pairs": None,
            "valid_pairs": None,
            "steps": None,
            "learning_rate": None,
            "lora_r": None,
            "runtime_sec": None,
            "pairs_per_second": metric_value(metrics, "pairs_per_second"),
        }
        for metric in METRICS:
            row[metric] = metric_value(metrics, metric)
        rows.append(row)
    return rows


def collect_lora_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in LORA_RUNS:
        config_path = resolve_path(spec["config"])
        metrics = read_json(spec["metrics"]) if resolve_path(spec["metrics"]).exists() else None
        summary = read_json(spec["summary"]) if resolve_path(spec["summary"]).exists() else {}
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        row = {
            "method": spec["name"],
            "framework": spec["framework"],
            "training": spec["training"],
            "role": spec["role"],
            "status": "ready" if metrics else "pending",
            "config_path": spec["config"],
            "metrics_path": spec["metrics"],
            "summary_path": spec["summary"],
            "train_pairs": summary.get("train_pairs"),
            "valid_pairs": summary.get("valid_pairs"),
            "steps": summary.get("steps", config.get("max_train_steps")),
            "learning_rate": config.get("learning_rate"),
            "lora_r": summary.get("lora_r", config.get("lora_r")),
            "loss_start": summary.get("loss_start"),
            "loss_end": summary.get("loss_end"),
            "loss_decreased": summary.get("loss_decreased"),
            "runtime_sec": summary.get("runtime_sec"),
            "pairs_per_second": metric_value(metrics, "pairs_per_second"),
        }
        for metric in METRICS:
            row[metric] = metric_value(metrics, metric)
        rows.append(row)
    return rows


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_lr(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


def write_markdown(rows: list[dict[str, Any]], path: str) -> None:
    headers = [
        "Method",
        "Framework",
        "Training",
        "R@1",
        "R@3",
        "R@5",
        "R@10",
        "NDCG@5",
        "NDCG@10",
        "MRR",
        "Pairwise Acc.",
        "Throughput",
        "Role",
    ]
    key_map = {
        "Method": "method",
        "Framework": "framework",
        "Training": "training",
        "R@1": "recall@1",
        "R@3": "recall@3",
        "R@5": "recall@5",
        "R@10": "recall@10",
        "NDCG@5": "ndcg@5",
        "NDCG@10": "ndcg@10",
        "MRR": "mrr",
        "Pairwise Acc.": "pairwise_accuracy",
        "Throughput": "pairs_per_second",
        "Role": "role",
    }
    lines = [
        "# Qwen2.5-1.5B LoRA Reranker Comparison",
        "",
        "Metrics are read from the committed JSON result files. LoRA adapter weights and base model weights are excluded from version control.",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(key_map[header])
            if header == "Throughput" and value is not None:
                values.append(f"{float(value):.2f} pairs/s")
            else:
                values.append(format_value(value))
        lines.append("| " + " | ".join(values) + " |")
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tuning_markdown(rows: list[dict[str, Any]], path: str) -> None:
    headers = [
        "Run",
        "Train pairs",
        "Valid pairs",
        "Steps",
        "LR",
        "LoRA r",
        "Loss start",
        "Loss end",
        "R@1",
        "R@5",
        "NDCG@5",
        "MRR",
    ]
    lines = [
        "# Qwen2.5-1.5B LoRA Tuning Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            row["method"],
            format_value(row.get("train_pairs")),
            format_value(row.get("valid_pairs")),
            format_value(row.get("steps")),
            format_lr(row.get("learning_rate")),
            format_value(row.get("lora_r")),
            format_value(row.get("loss_start")),
            format_value(row.get("loss_end")),
            format_value(row.get("recall@1")),
            format_value(row.get("recall@5")),
            format_value(row.get("ndcg@5")),
            format_value(row.get("mrr")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    baseline_rows = collect_baseline_rows()
    lora_rows = collect_lora_rows()
    comparison_rows = baseline_rows[:5] + [lora_rows[-1]] + baseline_rows[5:]

    write_json("outputs/lora_qwen2_1_5b_comparison.json", comparison_rows)
    write_markdown(comparison_rows, "outputs/lora_qwen2_1_5b_comparison.md")
    write_json("outputs/lora_qwen2_1_5b_tuning_summary.json", lora_rows)
    write_tuning_markdown(lora_rows, "outputs/lora_qwen2_1_5b_tuning_summary.md")
    print(resolve_path("outputs/lora_qwen2_1_5b_comparison.md").read_text(encoding="utf-8"))
    print(resolve_path("outputs/lora_qwen2_1_5b_tuning_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
