"""Aggregate L2.5 MS MARCO medium results including Cross-Encoder reference."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_parent, resolve_path, write_json


METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5"]
PLOT_METRICS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]

METHODS = [
    {
        "method": "TF-IDF",
        "framework": "sklearn",
        "training": "none",
        "path": "outputs/msmarco_medium_retrieval_baseline_metrics.json",
        "nested_key": "tfidf",
        "role": "lexical baseline",
    },
    {
        "method": "BM25",
        "framework": "rank_bm25",
        "training": "none",
        "path": "outputs/msmarco_medium_retrieval_baseline_metrics.json",
        "nested_key": "bm25",
        "role": "lexical baseline",
    },
    {
        "method": "MLP",
        "framework": "PyTorch",
        "training": "from scratch",
        "path": "outputs/msmarco_medium_torch_metrics.json",
        "role": "PyTorch lightweight reranker",
    },
    {
        "method": "MLP",
        "framework": "Jittor",
        "training": "from scratch",
        "path": "outputs/msmarco_medium_jittor_metrics.json",
        "role": "Jittor lightweight reranker",
    },
    {
        "method": "TextCNN",
        "framework": "PyTorch",
        "training": "from scratch",
        "path": "outputs/msmarco_medium_textcnn_torch_metrics.json",
        "role": "PyTorch neural reranker",
    },
    {
        "method": "TextCNN",
        "framework": "Jittor",
        "training": "from scratch",
        "path": "outputs/msmarco_medium_textcnn_jittor_metrics.json",
        "role": "Jittor neural reranker",
    },
    {
        "method": "Cross-Encoder",
        "framework": "sentence-transformers",
        "training": "external pretrained",
        "path": "outputs/msmarco_medium_cross_encoder_metrics.json",
        "role": "external pretrained semantic reranker reference",
    },
]


def load_metrics(spec: dict) -> dict | None:
    path = resolve_path(spec["path"])
    if not path.exists():
        print(f"pending: {spec['path']}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("nested_key"):
        payload = payload.get(spec["nested_key"], {})
        if payload.get("status") != "ready":
            return None
        payload = payload.get("metrics", {})
    return payload


def collect_rows() -> list[dict]:
    rows = []
    for spec in METHODS:
        metrics = load_metrics(spec)
        row = {
            "Method": spec["method"],
            "Framework": spec["framework"],
            "Training": spec["training"],
            "Role": spec["role"],
            "Status": "ready" if metrics else "pending",
            "Time / throughput": None,
        }
        for metric in METRICS:
            row[metric] = float(metrics[metric]) if metrics and metric in metrics else None
        if metrics:
            if "pairs_per_second" in metrics:
                row["Time / throughput"] = f"{metrics['pairs_per_second']:.2f} pairs/s"
            elif "total_time_sec" in metrics:
                row["Time / throughput"] = f"{metrics['total_time_sec']:.2f}s"
        rows.append(row)
    return rows


def format_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict], path: str) -> None:
    headers = ["Method", "Framework", "Training", "R@1", "R@3", "R@5", "R@10", "MRR", "NDCG@3", "NDCG@5", "Time / throughput", "Role"]
    metric_map = {
        "R@1": "recall@1",
        "R@3": "recall@3",
        "R@5": "recall@5",
        "R@10": "recall@10",
        "MRR": "mrr",
        "NDCG@3": "ndcg@3",
        "NDCG@5": "ndcg@5",
    }
    lines = [
        "# L2.5 MS MARCO Medium Results",
        "",
        "Cross-Encoder is an external pretrained semantic reranker reference, not the Jittor reproduction body.",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        values = []
        for header in headers:
            key = metric_map.get(header, header)
            values.append(format_value(row.get(key)))
        lines.append("| " + " | ".join(values) + " |")
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(rows: list[dict], path: str) -> None:
    ready_rows = [row for row in rows if row["Status"] == "ready"]
    if not ready_rows:
        return
    labels = [f"{row['Method']}\n{row['Framework']}" for row in ready_rows]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for idx, metric in enumerate(PLOT_METRICS):
        values = [row[metric] or 0.0 for row in ready_rows]
        ax.bar(x + (idx - 2) * width, values, width=width, label=metric)
    ax.set_title("L2.5 MS MARCO medium: lexical, lightweight Jittor, and pretrained semantic reranking")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=160)
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    write_json("outputs/l25_msmarco_medium_results.json", rows)
    write_markdown(rows, "outputs/l25_msmarco_medium_results.md")
    plot_results(rows, "outputs/l25_msmarco_medium_results.png")
    print(resolve_path("outputs/l25_msmarco_medium_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
