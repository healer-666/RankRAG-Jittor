"""Create publication-style README figures from existing experiment outputs.

The figures intentionally stay simple: one analytical question per chart,
white background, restrained blue/gray palette, and orange only for the
external pretrained Cross-Encoder reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

INK = "#243040"
MUTED = "#667085"
GRID = "#E6EAF0"
LEXICAL = "#7A869A"
PYTORCH = "#8DB7E8"
JITTOR = "#2F6FA5"
PRETRAINED = "#D97925"
PANEL = "#F8FAFC"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
    }
)


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = FIG_DIR / name
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output}")


def setup_axis(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=18)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def metric_rows() -> list[dict]:
    retrieval = read_json("outputs/msmarco_medium_retrieval_baseline_metrics.json")
    return [
        {"method": "TF-IDF", "group": "Lexical", **retrieval["tfidf"]["metrics"]},
        {"method": "BM25", "group": "Lexical", **retrieval["bm25"]["metrics"]},
        {"method": "MLP PyTorch", "group": "PyTorch", **read_json("outputs/msmarco_medium_torch_metrics.json")},
        {"method": "MLP Jittor", "group": "Jittor", **read_json("outputs/msmarco_medium_jittor_metrics.json")},
        {"method": "TextCNN PyTorch", "group": "PyTorch", **read_json("outputs/msmarco_medium_textcnn_torch_metrics.json")},
        {"method": "TextCNN Jittor", "group": "Jittor", **read_json("outputs/msmarco_medium_textcnn_jittor_metrics.json")},
        {"method": "Cross-Encoder", "group": "Pretrained", **read_json("outputs/msmarco_medium_cross_encoder_metrics.json")},
    ]


def group_color(group: str) -> str:
    return {"Lexical": LEXICAL, "PyTorch": PYTORCH, "Jittor": JITTOR, "Pretrained": PRETRAINED}[group]


def draw_box(ax, x, y, w, h, title, body, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=PANEL,
    )
    ax.add_patch(patch)
    ax.text(x + 0.04, y + h - 0.055, title, ha="left", va="top", fontsize=12, fontweight="bold")
    ax.text(x + 0.04, y + h - 0.13, body, ha="left", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)


def draw_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#98A2B3"))


def make_project_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.94, "RankRAG-Jittor experiment map", fontsize=18, fontweight="bold")
    ax.text(0.04, 0.89, "Context ranking reproduction, public ranking evaluation, and pretrained semantic reference", fontsize=10.5, color=MUTED)

    draw_box(ax, 0.05, 0.58, 0.25, 0.20, "L0: smoke test", "Synthetic hard-negative data\nPipeline correctness check", LEXICAL)
    draw_box(ax, 0.38, 0.58, 0.25, 0.20, "L1 / L2: Jittor body", "MS MARCO small + medium\nMLP and TextCNN rerankers", JITTOR)
    draw_box(ax, 0.71, 0.58, 0.25, 0.20, "L2.5: reference", "External Cross-Encoder\nPretrained semantic reranking", PRETRAINED)
    draw_box(ax, 0.22, 0.25, 0.25, 0.18, "Baselines", "TF-IDF and BM25\nLexical matching strength", LEXICAL)
    draw_box(ax, 0.55, 0.25, 0.25, 0.18, "Artifacts", "Metrics, figures, case studies\nREADME-ready reports", JITTOR)
    draw_arrow(ax, (0.30, 0.68), (0.38, 0.68))
    draw_arrow(ax, (0.63, 0.68), (0.71, 0.68))
    draw_arrow(ax, (0.50, 0.58), (0.35, 0.43))
    draw_arrow(ax, (0.50, 0.58), (0.67, 0.43))
    save(fig, "01_project_overview.png")


def make_method_layers() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.92, "Method layers", fontsize=17, fontweight="bold")
    ax.text(0.04, 0.86, "The project compares lexical matching, lightweight trainable rerankers, and pretrained semantic reranking.", fontsize=10.5, color=MUTED)

    boxes = [
        (0.06, "Lexical baselines", "TF-IDF\nBM25", LEXICAL),
        (0.38, "Lightweight rerankers", "MLP PyTorch / Jittor\nTextCNN PyTorch / Jittor", JITTOR),
        (0.70, "Pretrained reference", "Cross-Encoder\nMS MARCO MiniLM", PRETRAINED),
    ]
    for x, title, body, color in boxes:
        draw_box(ax, x, 0.36, 0.24, 0.26, title, body, color)
    draw_arrow(ax, (0.30, 0.49), (0.38, 0.49))
    draw_arrow(ax, (0.62, 0.49), (0.70, 0.49))
    ax.text(0.50, 0.18, "surface overlap  →  from-scratch neural features  →  pretrained semantic judgment", ha="center", fontsize=10.5, color=MUTED)
    save(fig, "02_method_layers.png")


def make_main_results() -> None:
    rows = metric_rows()
    order = ["TF-IDF", "BM25", "MLP Jittor", "TextCNN Jittor", "Cross-Encoder"]
    rows = [next(row for row in rows if row["method"] == name) for name in order]
    metrics = [("recall@1", "R@1"), ("recall@5", "R@5"), ("mrr", "MRR"), ("ndcg@5", "NDCG@5")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.4), sharey=True)
    y = np.arange(len(rows))
    for ax, (key, label) in zip(axes, metrics):
        values = [row[key] for row in rows]
        colors = [group_color(row["group"]) for row in rows]
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_xlim(0, 1.0)
        ax.set_title(label, fontsize=12.5, fontweight="bold")
        ax.grid(axis="x", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        for yi, v in zip(y, values):
            ax.text(v + 0.018, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("MS MARCO medium: pretrained semantic reranking is the strongest reference", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.90, "500 test queries, up to 10 candidates per query. Cross-Encoder is external and pretrained.", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.86])
    save(fig, "03_msmarco_medium_results.png")


def make_alignment() -> None:
    rows = {row["method"]: row for row in metric_rows()}
    metrics = [("recall@1", "R@1"), ("recall@3", "R@3"), ("recall@5", "R@5"), ("mrr", "MRR"), ("ndcg@5", "NDCG@5")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, title, pt, jt in [
        (axes[0], "MLP", "MLP PyTorch", "MLP Jittor"),
        (axes[1], "TextCNN", "TextCNN PyTorch", "TextCNN Jittor"),
    ]:
        x = np.arange(len(metrics))
        ax.bar(x - 0.17, [rows[pt][m] for m, _ in metrics], width=0.34, color=PYTORCH, label="PyTorch")
        ax.bar(x + 0.17, [rows[jt][m] for m, _ in metrics], width=0.34, color=JITTOR, label="Jittor")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("PyTorch and Jittor stay close on lightweight rerankers", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.90, "MS MARCO medium metrics. Small differences are expected from initialization and framework details.", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.86])
    save(fig, "04_framework_alignment.png")


def make_medium_difficulty() -> None:
    small_retrieval = read_json("outputs/msmarco_retrieval_baseline_metrics.json")
    small = {
        "BM25": small_retrieval["bm25"]["metrics"],
        "MLP Jittor": read_json("outputs/msmarco_jittor_metrics.json"),
        "TextCNN Jittor": read_json("outputs/msmarco_textcnn_jittor_metrics.json"),
    }
    medium = {row["method"]: row for row in metric_rows()}
    methods = ["BM25", "MLP Jittor", "TextCNN Jittor"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for ax, key, label in [(axes[0], "recall@5", "R@5"), (axes[1], "mrr", "MRR")]:
        x = np.arange(len(methods))
        ax.bar(x - 0.17, [small[m][key] for m in methods], width=0.34, color="#B7CDE5", label="small, ≤5 candidates")
        ax.bar(x + 0.17, [medium[m][key] for m in methods], width=0.34, color=JITTOR, label="medium, ~8-10 candidates")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=12, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Medium subset is more discriminative than small", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.90, "Small is a quick check; medium is the main public-data evaluation used in the README.", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.86])
    save(fig, "05_small_vs_medium.png")


def parse_losses(path: str) -> list[float]:
    log = ROOT / path
    if not log.exists():
        return []
    pattern = re.compile(r"epoch=(\d+)\s+train_loss=([-+0-9.eE]+)")
    losses = []
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            losses.append(float(match.group(2)))
    return losses[-5:]


def make_training_curves() -> None:
    series = [
        ("MLP PyTorch", "logs/msmarco_medium_torch_train.log", PYTORCH),
        ("MLP Jittor", "logs/msmarco_medium_jittor_train.log", JITTOR),
        ("TextCNN PyTorch", "logs/msmarco_medium_textcnn_torch_train.log", "#E8B06D"),
        ("TextCNN Jittor", "logs/msmarco_medium_textcnn_jittor_train.log", "#5F8D7A"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for label, path, color in series:
        losses = parse_losses(path)
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2.2, markersize=5, label=label, color=color)
    setup_axis(ax, "Training curves on MS MARCO medium", "Pairwise ranking loss over the latest 5 logged epochs.")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "06_training_curves.png")


def main() -> None:
    make_project_overview()
    make_method_layers()
    make_main_results()
    make_alignment()
    make_medium_difficulty()
    make_training_curves()


if __name__ == "__main__":
    main()
