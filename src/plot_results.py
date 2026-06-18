"""Plot training loss and validation metric curves from backend log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from utils import resolve_path


LOG_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"valid_pairwise_acc=(?P<pairwise>[-+0-9.eE]+)\s+"
    r"valid_recall@1=(?P<recall1>[-+0-9.eE]+)\s+"
    r"valid_recall@3=(?P<recall3>[-+0-9.eE]+)\s+"
    r"valid_recall@5=(?P<recall5>[-+0-9.eE]+)\s+"
    r"valid_mrr=(?P<mrr>[-+0-9.eE]+)\s+"
    r"valid_ndcg@5=(?P<ndcg5>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="synthetic", choices=["synthetic", "msmarco"])
    return parser.parse_args()


def paths_for_run(run_name: str) -> dict[str, str]:
    if run_name == "msmarco":
        return {
            "torch_log": "logs/msmarco_torch_train.log",
            "jittor_log": "logs/msmarco_jittor_train.log",
            "loss_png": "outputs/msmarco_loss_curve.png",
            "metrics_png": "outputs/msmarco_metrics_compare.png",
        }
    return {
        "torch_log": "logs/torch_train.log",
        "jittor_log": "logs/jittor_train.log",
        "loss_png": "outputs/loss_curve.png",
        "metrics_png": "outputs/metrics_compare.png",
    }


def parse_log(path: str | Path) -> list[dict[str, float]]:
    """Extract per-epoch training loss and validation metrics from a train log."""

    resolved = resolve_path(path)
    if not resolved.exists():
        return []

    rows: list[dict[str, float]] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        # Keep the latest training block when logs contain an earlier smoke test.
        if "Training " in line and " scorer on " in line:
            rows = []
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("loss")),
                "valid_pairwise_acc": float(match.group("pairwise")),
                "valid_recall@1": float(match.group("recall1")),
                "valid_recall@3": float(match.group("recall3")),
                "valid_recall@5": float(match.group("recall5")),
                "valid_mrr": float(match.group("mrr")),
                "valid_ndcg@5": float(match.group("ndcg5")),
            }
        )
    return rows


def plot_loss(curves: dict[str, list[dict[str, float]]], output_path: Path, jittor_available: bool) -> None:
    plt.figure(figsize=(8, 5))
    for backend, rows in curves.items():
        if rows:
            plt.plot([row["epoch"] for row in rows], [row["train_loss"] for row in rows], marker="o", label=backend)
    suffix = "" if jittor_available else " (Jittor result is unavailable)"
    plt.title(f"Training Loss Curve{suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Pairwise Ranking Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_metrics(curves: dict[str, list[dict[str, float]]], output_path: Path, jittor_available: bool) -> None:
    metric_names = ["valid_pairwise_acc", "valid_recall@1", "valid_recall@3", "valid_recall@5", "valid_mrr", "valid_ndcg@5"]
    torch_rows = curves.get("PyTorch", [])
    jittor_rows = curves.get("Jittor", [])

    if not torch_rows and not jittor_rows:
        raise RuntimeError("No train log metrics found. Run training before plotting.")

    x = range(len(metric_names))
    width = 0.35
    torch_values = [torch_rows[-1][name] if torch_rows else 0.0 for name in metric_names]
    jittor_values = [jittor_rows[-1][name] if jittor_rows else 0.0 for name in metric_names]

    plt.figure(figsize=(10, 5))
    plt.bar([idx - width / 2 for idx in x], torch_values, width=width, label="PyTorch")
    if jittor_available:
        plt.bar([idx + width / 2 for idx in x], jittor_values, width=width, label="Jittor")
    suffix = "" if jittor_available else " (Jittor result is unavailable)"
    plt.title(f"Validation Metrics Compare{suffix}")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(list(x), metric_names, rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    paths = paths_for_run(args.run_name)
    output_dir = resolve_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_rows = parse_log(paths["torch_log"])
    jittor_rows = parse_log(paths["jittor_log"])
    jittor_available = bool(jittor_rows)
    curves = {"PyTorch": torch_rows, "Jittor": jittor_rows}

    if not jittor_available:
        print("Jittor result is unavailable; plotting PyTorch curves only.")

    loss_path = resolve_path(paths["loss_png"])
    metrics_path = resolve_path(paths["metrics_png"])
    plot_loss(curves, loss_path, jittor_available=jittor_available)
    plot_metrics(curves, metrics_path, jittor_available=jittor_available)

    print(f"Wrote {loss_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
