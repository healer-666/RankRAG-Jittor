"""Generate polished README figures from existing experiment artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

BLUE = "#345C8C"
BLUE_LIGHT = "#8FB3D9"
GRAY = "#6B7280"
GRAY_LIGHT = "#E5E7EB"
ORANGE = "#D97706"
ORANGE_LIGHT = "#F2C078"
GREEN = "#4F7F6F"
TEXT = "#1F2937"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#D1D5DB",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def draw_box(ax, xy, wh, title, body="", color=BLUE, fill="#F8FAFC"):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, weight="bold", color=TEXT)
    if body:
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=9.5, color=GRAY, linespacing=1.2)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#9CA3AF"))


def metric(metrics: dict, name: str) -> float:
    return float(metrics[name])


def medium_rows() -> list[dict]:
    retrieval = read_json("outputs/msmarco_medium_retrieval_baseline_metrics.json")
    rows = [
        {"method": "TF-IDF", "group": "Lexical", **retrieval["tfidf"]["metrics"]},
        {"method": "BM25", "group": "Lexical", **retrieval["bm25"]["metrics"]},
        {"method": "MLP PyTorch", "group": "PyTorch", **read_json("outputs/msmarco_medium_torch_metrics.json")},
        {"method": "MLP Jittor", "group": "Jittor", **read_json("outputs/msmarco_medium_jittor_metrics.json")},
        {
            "method": "TextCNN PyTorch",
            "group": "PyTorch",
            **read_json("outputs/msmarco_medium_textcnn_torch_metrics.json"),
        },
        {
            "method": "TextCNN Jittor",
            "group": "Jittor",
            **read_json("outputs/msmarco_medium_textcnn_jittor_metrics.json"),
        },
        {
            "method": "Cross-Encoder",
            "group": "Pretrained",
            **read_json("outputs/msmarco_medium_cross_encoder_metrics.json"),
        },
    ]
    return rows


def small_rows() -> dict[str, dict]:
    retrieval = read_json("outputs/msmarco_retrieval_baseline_metrics.json")
    return {
        "BM25": retrieval["bm25"]["metrics"],
        "MLP Jittor": read_json("outputs/msmarco_jittor_metrics.json"),
        "TextCNN Jittor": read_json("outputs/msmarco_textcnn_jittor_metrics.json"),
    }


def plot_project_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "RankRAG-Jittor: Resource-Constrained Context Ranking Reproduction", ha="center", fontsize=17, weight="bold")
    ax.text(0.5, 0.89, "Lightweight Jittor rerankers + public ranking evaluation + pretrained semantic reference", ha="center", fontsize=11, color=GRAY)

    draw_box(ax, (0.05, 0.58), (0.24, 0.19), "L0\nSynthetic Smoke Test", "pipeline validation\nhard negatives", BLUE_LIGHT)
    draw_box(ax, (0.38, 0.58), (0.24, 0.19), "L1 / L2\nPublic Ranking", "MS MARCO small & medium\nTF-IDF / BM25 / MLP / TextCNN", BLUE)
    draw_box(ax, (0.71, 0.58), (0.24, 0.19), "L2.5\nSemantic Reference", "Cross-Encoder\nexternal pretrained reranker", ORANGE)
    draw_box(ax, (0.20, 0.22), (0.22, 0.16), "Jittor Body", "MLP + TextCNN\nfrom-scratch rerankers", GREEN)
    draw_box(ax, (0.58, 0.22), (0.22, 0.16), "Analysis Artifacts", "metrics, figures,\ncase studies, reports", GRAY)
    arrow(ax, (0.29, 0.675), (0.38, 0.675))
    arrow(ax, (0.62, 0.675), (0.71, 0.675))
    arrow(ax, (0.50, 0.58), (0.31, 0.38))
    arrow(ax, (0.50, 0.58), (0.69, 0.38))
    save(fig, "project_overview.png")


def plot_method_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.92, "Method Taxonomy: from lexical matching to pretrained semantic reranking", ha="center", fontsize=16, weight="bold")
    stages = [
        ("Lexical baselines", "TF-IDF\nBM25", GRAY, 0.08),
        ("Lightweight trainable rerankers", "MLP PyTorch / Jittor\nTextCNN PyTorch / Jittor", BLUE, 0.39),
        ("Pretrained semantic reranker", "Cross-Encoder\nMS MARCO MiniLM", ORANGE, 0.70),
    ]
    for title, body, color, x in stages:
        draw_box(ax, (x, 0.37), (0.22, 0.27), title, body, color=color)
    arrow(ax, (0.30, 0.505), (0.39, 0.505))
    arrow(ax, (0.61, 0.505), (0.70, 0.505))
    ax.text(0.5, 0.18, "Surface lexical signals  →  from-scratch neural features  →  pretrained semantic judgment", ha="center", fontsize=11, color=GRAY)
    save(fig, "method_taxonomy.png")


def plot_medium_main_results() -> None:
    rows = medium_rows()
    metrics = [("recall@1", "R@1"), ("recall@3", "R@3"), ("recall@5", "R@5"), ("mrr", "MRR"), ("ndcg@5", "NDCG@5")]
    labels = [row["method"] for row in rows]
    colors = {"Lexical": GRAY, "PyTorch": BLUE_LIGHT, "Jittor": BLUE, "Pretrained": ORANGE}
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        values = [metric(row, key) for row in rows]
        ax.barh(y, values, color=[colors[row["group"]] for row in rows], height=0.62)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", alpha=0.2)
        for yi, v in zip(y, values):
            ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("MS MARCO medium main results", fontsize=17, weight="bold", y=0.98)
    fig.text(0.5, 0.02, "Cross-Encoder is an external pretrained reference, not the Jittor reproduction body.", ha="center", color=GRAY, fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, "msmarco_medium_main_results.png")


def plot_small_vs_medium() -> None:
    small = small_rows()
    medium = {row["method"]: row for row in medium_rows() if row["method"] in {"BM25", "MLP Jittor", "TextCNN Jittor"}}
    methods = ["BM25", "MLP Jittor", "TextCNN Jittor"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, key, title in [(axes[0], "recall@5", "R@5"), (axes[1], "mrr", "MRR")]:
        x = np.arange(len(methods))
        ax.bar(x - 0.18, [small[m][key] for m in methods], width=0.36, color=BLUE_LIGHT, label="small (≤5 candidates)")
        ax.bar(x + 0.18, [medium[m][key] for m in methods], width=0.36, color=BLUE, label="medium (~8-10 candidates)")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Small vs medium subset: medium is more discriminative", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "small_vs_medium_comparison.png")


def plot_framework_alignment() -> None:
    rows = {row["method"]: row for row in medium_rows()}
    pairs = [("MLP PyTorch", "MLP Jittor", "MLP"), ("TextCNN PyTorch", "TextCNN Jittor", "TextCNN")]
    metrics = [("recall@1", "R@1"), ("recall@3", "R@3"), ("recall@5", "R@5"), ("mrr", "MRR"), ("ndcg@5", "NDCG@5")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (pt, jt, title) in zip(axes, pairs):
        x = np.arange(len(metrics))
        ax.bar(x - 0.18, [rows[pt][key] for key, _ in metrics], width=0.36, color=BLUE_LIGHT, label="PyTorch")
        ax.bar(x + 0.18, [rows[jt][key] for key, _ in metrics], width=0.36, color=BLUE, label="Jittor")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("PyTorch / Jittor alignment on MS MARCO medium", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "framework_alignment.png")


def plot_cross_encoder_gain() -> None:
    rows = {row["method"]: row for row in medium_rows()}
    methods = ["BM25", "MLP Jittor", "TextCNN Jittor", "Cross-Encoder"]
    metrics = [("recall@1", "R@1"), ("recall@5", "R@5"), ("mrr", "MRR"), ("ndcg@5", "NDCG@5")]
    colors = [GRAY, BLUE, BLUE_LIGHT, ORANGE]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        values = [rows[m][key] for m in methods]
        ax.bar(np.arange(len(methods)), values, color=colors, width=0.64)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(values):
            ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("External pretrained Cross-Encoder provides a strong semantic reranking reference", fontsize=15.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "cross_encoder_gain.png")


def parse_loss_log(path: str) -> list[float]:
    resolved = ROOT / path
    if not resolved.exists():
        return []
    losses = []
    pattern = re.compile(r"epoch=(\d+)\s+train_loss=([-+0-9.eE]+)")
    for line in resolved.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            losses.append(float(match.group(2)))
    return losses[-5:]


def plot_training_curves() -> None:
    logs = [
        ("MLP PyTorch", "logs/msmarco_medium_torch_train.log", BLUE_LIGHT),
        ("MLP Jittor", "logs/msmarco_medium_jittor_train.log", BLUE),
        ("TextCNN PyTorch", "logs/msmarco_medium_textcnn_torch_train.log", ORANGE_LIGHT),
        ("TextCNN Jittor", "logs/msmarco_medium_textcnn_jittor_train.log", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, path, color in logs:
        losses = parse_loss_log(path)
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2, label=label, color=color)
    ax.set_title("Training curves on MS MARCO medium", fontsize=16, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pairwise ranking loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "training_curves_overview.png")


def plot_case_summary() -> None:
    path = ROOT / "outputs" / "msmarco_medium_cross_encoder_case_study.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = [
        ("CE succeeds\nBM25 fails", "cross_encoder_success_bm25_failure"),
        ("BM25 succeeds\nCE fails", "bm25_success_cross_encoder_failure"),
        ("All methods\nfail", "all_methods_failure"),
        ("Lightweight\nmodel gap", "lightweight_gap"),
    ]
    counts = [len(data.get(key, [])) for _, key in labels]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(np.arange(len(labels)), counts, color=[ORANGE, GRAY, GRAY_LIGHT, BLUE], width=0.58)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([label for label, _ in labels])
    ax.set_ylim(0, max(counts + [1]) + 1)
    ax.set_ylabel("Selected cases")
    ax.set_title("Cross-Encoder case study buckets", fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    save(fig, "case_study_summary.png")


def main() -> None:
    ensure_fig_dir()
    plot_project_overview()
    plot_method_taxonomy()
    plot_medium_main_results()
    plot_small_vs_medium()
    plot_framework_alignment()
    plot_cross_encoder_gain()
    plot_training_curves()
    plot_case_summary()


if __name__ == "__main__":
    main()
