"""Aggregate L2 MS MARCO results across retrieval, MLP, and TextCNN methods."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_parent, read_json, resolve_path, write_json


METRICS = ["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr", "pairwise_accuracy"]
PLOT_METRICS = ["recall@1", "recall@3", "mrr", "ndcg@3"]


def load_metrics_file(path: str) -> dict | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        print(f"pending: {path}")
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def add_row(rows: list[dict], method: str, framework: str, metrics: dict | None) -> None:
    row = {"Method": method, "Framework": framework}
    if metrics is None:
        row["Status"] = "pending"
        for metric in METRICS:
            row[metric] = None
    else:
        row["Status"] = "ready"
        for metric in METRICS:
            row[metric] = float(metrics[metric]) if metric in metrics else None
    rows.append(row)


def collect_rows() -> list[dict]:
    rows: list[dict] = []

    retrieval = load_metrics_file("outputs/msmarco_retrieval_baseline_metrics.json")
    for method in ["tfidf", "bm25"]:
        payload = retrieval.get(method) if retrieval else None
        metrics = payload.get("metrics") if payload and payload.get("status") == "ready" else None
        add_row(rows, method.upper(), "sklearn/rank_bm25", metrics)

    add_row(rows, "MLP", "PyTorch", load_metrics_file("outputs/msmarco_torch_metrics.json"))
    add_row(rows, "MLP", "Jittor", load_metrics_file("outputs/msmarco_jittor_metrics.json"))
    add_row(rows, "TextCNN", "PyTorch", load_metrics_file("outputs/msmarco_textcnn_torch_metrics.json"))
    add_row(rows, "TextCNN", "Jittor", load_metrics_file("outputs/msmarco_textcnn_jittor_metrics.json"))
    return rows


def format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(rows: list[dict], path: str) -> None:
    headers = ["Method", "Framework", "Status", "Recall@1", "Recall@3", "Recall@5", "NDCG@1", "NDCG@3", "NDCG@5", "MRR", "Pairwise Accuracy"]
    metric_keys = ["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr", "pairwise_accuracy"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | --- | " + " | ".join(["---:"] * len(metric_keys)) + " |"]
    for row in rows:
        values = [format_value(row[key]) for key in metric_keys]
        lines.append("| " + " | ".join([row["Method"], row["Framework"], row["Status"], *values]) + " |")
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(rows: list[dict], path: str) -> None:
    ready_rows = [row for row in rows if row["Status"] == "ready"]
    if not ready_rows:
        return

    labels = [f"{row['Method']}\n{row['Framework']}" for row in ready_rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, metric in enumerate(PLOT_METRICS):
        values = [row[metric] if row[metric] is not None else 0.0 for row in ready_rows]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=metric)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("L2 MS MARCO multi-model ranking results")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=160)
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    write_json("outputs/l2_msmarco_results.json", rows)
    write_markdown(rows, "outputs/l2_msmarco_results.md")
    plot_results(rows, "outputs/l2_msmarco_results.png")
    print(resolve_path("outputs/l2_msmarco_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
