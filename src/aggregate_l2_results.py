"""Aggregate L2 MS MARCO results across retrieval, MLP, and TextCNN methods."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_parent, read_json, resolve_path, write_json


BASE_METRICS = ["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr", "pairwise_accuracy"]
MEDIUM_METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "mrr",
    "pairwise_accuracy",
]
PLOT_METRICS = ["recall@1", "recall@3", "mrr", "ndcg@3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="msmarco", choices=["msmarco", "msmarco_medium"])
    return parser.parse_args()


def load_metrics_file(path: str) -> dict | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        print(f"pending: {path}")
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def add_row(rows: list[dict], method: str, framework: str, metrics: dict | None, metric_keys: list[str]) -> None:
    row = {"Method": method, "Framework": framework}
    if metrics is None:
        row["Status"] = "pending"
        for metric in metric_keys:
            row[metric] = None
    else:
        row["Status"] = "ready"
        for metric in metric_keys:
            row[metric] = float(metrics[metric]) if metric in metrics else None
    rows.append(row)


def paths_for_run(run_name: str) -> dict[str, str]:
    if run_name == "msmarco_medium":
        return {
            "retrieval": "outputs/msmarco_medium_retrieval_baseline_metrics.json",
            "torch": "outputs/msmarco_medium_torch_metrics.json",
            "jittor": "outputs/msmarco_medium_jittor_metrics.json",
            "textcnn_torch": "outputs/msmarco_medium_textcnn_torch_metrics.json",
            "textcnn_jittor": "outputs/msmarco_medium_textcnn_jittor_metrics.json",
            "json": "outputs/l2_msmarco_medium_results.json",
            "md": "outputs/l2_msmarco_medium_results.md",
            "png": "outputs/l2_msmarco_medium_results.png",
        }
    return {
        "retrieval": "outputs/msmarco_retrieval_baseline_metrics.json",
        "torch": "outputs/msmarco_torch_metrics.json",
        "jittor": "outputs/msmarco_jittor_metrics.json",
        "textcnn_torch": "outputs/msmarco_textcnn_torch_metrics.json",
        "textcnn_jittor": "outputs/msmarco_textcnn_jittor_metrics.json",
        "json": "outputs/l2_msmarco_results.json",
        "md": "outputs/l2_msmarco_results.md",
        "png": "outputs/l2_msmarco_results.png",
    }


def collect_rows(paths: dict[str, str], metric_keys: list[str]) -> list[dict]:
    rows: list[dict] = []

    retrieval = load_metrics_file(paths["retrieval"])
    for method in ["tfidf", "bm25"]:
        payload = retrieval.get(method) if retrieval else None
        metrics = payload.get("metrics") if payload and payload.get("status") == "ready" else None
        add_row(rows, method.upper(), "sklearn/rank_bm25", metrics, metric_keys)

    add_row(rows, "MLP", "PyTorch", load_metrics_file(paths["torch"]), metric_keys)
    add_row(rows, "MLP", "Jittor", load_metrics_file(paths["jittor"]), metric_keys)
    add_row(rows, "TextCNN", "PyTorch", load_metrics_file(paths["textcnn_torch"]), metric_keys)
    add_row(rows, "TextCNN", "Jittor", load_metrics_file(paths["textcnn_jittor"]), metric_keys)
    return rows


def format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def metric_label(metric: str) -> str:
    return metric.replace("recall", "Recall").replace("ndcg", "NDCG").replace("mrr", "MRR").replace(
        "pairwise_accuracy", "Pairwise Accuracy"
    )


def write_markdown(rows: list[dict], path: str, metric_keys: list[str]) -> None:
    headers = ["Method", "Framework", "Status", *[metric_label(metric) for metric in metric_keys]]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | --- | " + " | ".join(["---:"] * len(metric_keys)) + " |"]
    for row in rows:
        values = [format_value(row[key]) for key in metric_keys]
        lines.append("| " + " | ".join([row["Method"], row["Framework"], row["Status"], *values]) + " |")
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(rows: list[dict], path: str, run_name: str) -> None:
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
    title = "L2 MS MARCO medium multi-model ranking results" if run_name == "msmarco_medium" else "L2 MS MARCO multi-model ranking results"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = paths_for_run(args.run_name)
    metric_keys = MEDIUM_METRICS if args.run_name == "msmarco_medium" else BASE_METRICS
    rows = collect_rows(paths, metric_keys)
    write_json(paths["json"], rows)
    write_markdown(rows, paths["md"], metric_keys)
    plot_results(rows, paths["png"], args.run_name)
    print(resolve_path(paths["md"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
