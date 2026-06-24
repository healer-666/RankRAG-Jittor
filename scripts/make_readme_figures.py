"""Generate README figures with the figures4papers publication style.

The visual style follows the installed scientific-figure-making skill from
ChenLiu-1996/figures4papers: high-contrast matplotlib figures, explicit
semantic palette, thick axis lines, print-safe hatches, wide comparison panels,
and high-DPI PNG export for README embedding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "ink": "#272727",
    "muted": "#767676",
    "grid": "#E7E7E7",
}


def apply_publication_style(font_size: int = 15, axes_linewidth: float = 2.0) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": axes_linewidth,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
        }
    )


def save_figure(fig: plt.Figure, name: str, *, dpi: int = 320) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_DIR / name, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def load_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_dict(relative_path: str, key: str | None = None) -> dict:
    data = load_json(relative_path)
    if key is not None:
        data = data[key]
    return data.get("metrics", data)


def medium_rows() -> list[dict]:
    sources = [
        ("TF-IDF", "outputs/msmarco_medium_retrieval_baseline_metrics.json", "tfidf", "Lexical"),
        ("BM25", "outputs/msmarco_medium_retrieval_baseline_metrics.json", "bm25", "Lexical"),
        ("MLP PyTorch", "outputs/msmarco_medium_torch_metrics.json", None, "PyTorch"),
        ("MLP Jittor", "outputs/msmarco_medium_jittor_metrics.json", None, "Jittor"),
        ("TextCNN PyTorch", "outputs/msmarco_medium_textcnn_torch_metrics.json", None, "PyTorch"),
        ("TextCNN Jittor", "outputs/msmarco_medium_textcnn_jittor_metrics.json", None, "Jittor"),
        ("Cross-Encoder", "outputs/msmarco_medium_cross_encoder_metrics.json", None, "External"),
    ]
    rows = []
    for method, path, key, family in sources:
        metrics = metric_dict(path, key)
        rows.append(
            {
                "method": method,
                "family": family,
                "r1": float(metrics["recall@1"]),
                "r3": float(metrics["recall@3"]),
                "r5": float(metrics["recall@5"]),
                "r10": float(metrics.get("recall@10", 1.0)),
                "mrr": float(metrics["mrr"]),
                "ndcg5": float(metrics["ndcg@5"]),
            }
        )
    return rows


def small_medium_rows() -> list[dict]:
    sources = [
        (
            "BM25",
            "outputs/msmarco_retrieval_baseline_metrics.json",
            "outputs/msmarco_medium_retrieval_baseline_metrics.json",
            "bm25",
        ),
        ("MLP Jittor", "outputs/msmarco_jittor_metrics.json", "outputs/msmarco_medium_jittor_metrics.json", None),
        (
            "TextCNN Jittor",
            "outputs/msmarco_textcnn_jittor_metrics.json",
            "outputs/msmarco_medium_textcnn_jittor_metrics.json",
            None,
        ),
    ]
    rows = []
    for method, small_path, medium_path, key in sources:
        small = metric_dict(small_path, key)
        medium = metric_dict(medium_path, key)
        rows.append(
            {
                "method": method,
                "small_r5": float(small["recall@5"]),
                "medium_r5": float(medium["recall@5"]),
                "small_mrr": float(small["mrr"]),
                "medium_mrr": float(medium["mrr"]),
            }
        )
    return rows


def parse_log(relative_path: str) -> list[dict]:
    path = ROOT / relative_path
    if not path.exists():
        return []
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+).*?train_loss=(?P<loss>[0-9.]+).*?"
        r"valid_recall@5=(?P<r5>[0-9.]+).*?valid_mrr=(?P<mrr>[0-9.]+)",
        re.IGNORECASE,
    )
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group("epoch")),
                    "loss": float(match.group("loss")),
                    "valid_r5": float(match.group("r5")),
                    "valid_mrr": float(match.group("mrr")),
                }
            )
    return rows


def draw_scope_figure() -> None:
    apply_publication_style(font_size=15, axes_linewidth=2.0)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)

    boxes = [
        ("Input", "Query + candidate\ncontexts", "#FFFFFF", PALETTE["neutral_dark"]),
        ("Selector", "MLP / TextCNN\nranking score", PALETTE["green_1"], PALETTE["blue_main"]),
        ("Evaluation", "Recall@K, NDCG@K\nMRR, pairwise acc.", "#FFFFFF", PALETTE["neutral_dark"]),
        ("Out of scope", "LLM instruction tuning\nanswer generation", PALETTE["red_1"], PALETTE["red_strong"]),
    ]
    xs = [0.6, 3.5, 6.4, 9.3]
    for idx, ((title, body, face, edge), x) in enumerate(zip(boxes, xs)):
        box = FancyBboxPatch(
            (x, 1.55),
            2.1,
            1.45,
            boxstyle="round,pad=0.08,rounding_size=0.05",
            facecolor=face,
            edgecolor="black",
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 1.05, 2.65, title, ha="center", va="center", color=edge, fontweight="bold", fontsize=13)
        ax.text(x + 1.05, 2.05, body, ha="center", va="center", color=PALETTE["ink"], fontsize=10)
        if idx < 2:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 2.22, 2.25),
                    (xs[idx + 1] - 0.14, 2.25),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=2.0,
                    color=PALETTE["ink"],
                )
            )
        if idx == 2:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 2.22, 2.25),
                    (xs[idx + 1] - 0.14, 2.25),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=2.0,
                    linestyle="--",
                    color=PALETTE["red_strong"],
                )
            )

    ax.text(
        0.6,
        4.25,
        "RankRAG-style context ranking reproduction boundary",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.6,
        3.82,
        "This project reproduces the selector / evidence-ranking module in PyTorch and Jittor, not the full RankRAG LLM generation stack.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    save_figure(fig, "01_reproduction_scope.png")


def draw_medium_metric_panels(rows: list[dict]) -> None:
    apply_publication_style(font_size=16, axes_linewidth=2.2)
    selected = ["BM25", "MLP Jittor", "TextCNN Jittor", "Cross-Encoder"]
    metric_keys = [("r1", "R@1"), ("r5", "R@5"), ("mrr", "MRR"), ("ndcg5", "NDCG@5")]
    colors = {
        "BM25": PALETTE["neutral"],
        "MLP Jittor": PALETTE["blue_main"],
        "TextCNN Jittor": PALETTE["blue_secondary"],
        "Cross-Encoder": PALETTE["red_strong"],
    }
    hatches = {"BM25": "//", "MLP Jittor": "", "TextCNN Jittor": "..", "Cross-Encoder": "\\\\"}

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]},
    )
    for ax, (key, label) in zip(axes[:4], metric_keys):
        vals = [next(row for row in rows if row["method"] == method)[key] for method in selected]
        bars = ax.bar(
            np.arange(len(selected)),
            vals,
            color=[colors[method] for method in selected],
            edgecolor="black",
            linewidth=1.6,
        )
        for bar, method, value in zip(bars, selected, vals):
            bar.set_hatch(hatches[method])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ymin = max(0.0, min(vals) - 0.08)
        ymax = min(1.05, max(vals) + 0.08)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_title(label, fontweight="bold", pad=10)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    axes[0].set_ylabel("Metric value")

    axes[4].set_axis_off()
    handles = [Patch(facecolor=colors[method], edgecolor="black", hatch=hatches[method], label=method) for method in selected]
    axes[4].legend(handles=handles, loc="center left", fontsize=12)
    fig.suptitle(
        "MS MARCO medium: lightweight Jittor rerankers vs lexical and semantic references",
        fontsize=18,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        0.965,
        "500 test queries, up to 10 candidates/query. Cross-Encoder is external pretrained reference, not the Jittor reproduction body.",
        ha="center",
        fontsize=11,
        color=PALETTE["muted"],
    )
    save_figure(fig, "02_msmarco_medium_metrics.png")


def draw_ranked_mrr(rows: list[dict]) -> None:
    apply_publication_style(font_size=15, axes_linewidth=2.0)
    ordered = sorted(rows, key=lambda row: row["mrr"])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    bar_colors = []
    hatches = []
    for row in ordered:
        if row["method"] == "Cross-Encoder":
            bar_colors.append(PALETTE["red_strong"])
            hatches.append("\\\\")
        elif "Jittor" in row["method"]:
            bar_colors.append(PALETTE["blue_main"])
            hatches.append("")
        elif "PyTorch" in row["method"]:
            bar_colors.append(PALETTE["blue_secondary"])
            hatches.append("..")
        else:
            bar_colors.append(PALETTE["neutral"])
            hatches.append("//")
    bars = ax.barh(
        [row["method"] for row in ordered],
        [row["mrr"] for row in ordered],
        color=bar_colors,
        edgecolor="black",
        linewidth=1.5,
    )
    for bar, hatch, row in zip(bars, hatches, ordered):
        bar.set_hatch(hatch)
        ax.text(row["mrr"] + 0.012, bar.get_y() + bar.get_height() / 2, f"{row['mrr']:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Mean reciprocal rank")
    ax.set_xlim(0, 0.70)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.set_title("MRR ranking on MS MARCO medium", fontweight="bold", pad=14)
    ax.text(
        0,
        1.02,
        "Ranked comparison across lexical baselines, lightweight trainable rerankers, and the external semantic reference.",
        transform=ax.transAxes,
        fontsize=10,
        color=PALETTE["muted"],
    )
    save_figure(fig, "03_msmarco_medium_mrr_ranking.png")


def draw_subset_comparison(rows: list[dict]) -> None:
    apply_publication_style(font_size=15, axes_linewidth=2.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    methods = [row["method"] for row in rows]
    y = np.arange(len(methods))
    panels = [
        (axes[0], "Recall@5", "small_r5", "medium_r5", (0.55, 1.05)),
        (axes[1], "MRR", "small_mrr", "medium_mrr", (0.35, 0.58)),
    ]
    for ax, title, small_key, medium_key, xlim in panels:
        small_vals = [row[small_key] for row in rows]
        medium_vals = [row[medium_key] for row in rows]
        ax.barh(y + 0.17, small_vals, height=0.28, color=PALETTE["neutral"], edgecolor="black", linewidth=1.3, hatch="//", label="Small")
        ax.barh(y - 0.17, medium_vals, height=0.28, color=PALETTE["blue_main"], edgecolor="black", linewidth=1.3, label="Medium")
        for yi, sv, mv in zip(y, small_vals, medium_vals):
            ax.text(sv + 0.008, yi + 0.17, f"{sv:.3f}", va="center", fontsize=9)
            ax.text(mv + 0.008, yi - 0.17, f"{mv:.3f}", va="center", fontsize=9)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    axes[0].set_yticks(y, methods)
    axes[1].tick_params(axis="y", length=0)
    axes[0].legend(loc="lower right", fontsize=10)
    fig.suptitle("Medium subset makes ranking metrics less saturated", fontsize=17, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.965,
        "Small uses 200 test queries and up to 5 candidates; medium uses 500 test queries and up to 10 candidates.",
        ha="center",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    save_figure(fig, "04_small_vs_medium.png")


def draw_training_curves() -> None:
    apply_publication_style(font_size=15, axes_linewidth=2.0)
    logs = [
        ("MLP PyTorch", "logs/msmarco_medium_torch_train.log", PALETTE["blue_secondary"], "--"),
        ("MLP Jittor", "logs/msmarco_medium_jittor_train.log", PALETTE["blue_main"], "-"),
        ("TextCNN PyTorch", "logs/msmarco_medium_textcnn_torch_train.log", PALETTE["teal"], "--"),
        ("TextCNN Jittor", "logs/msmarco_medium_textcnn_jittor_train.log", PALETTE["green_3"], "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, path, color, linestyle in logs:
        rows = parse_log(path)
        if not rows:
            continue
        epochs = [row["epoch"] for row in rows]
        losses = [row["loss"] for row in rows]
        mrr = [row["valid_mrr"] for row in rows]
        axes[0].plot(epochs, losses, label=label, color=color, linestyle=linestyle, linewidth=2.2, marker="o", markersize=4)
        axes[1].plot(epochs, mrr, label=label, color=color, linestyle=linestyle, linewidth=2.2, marker="o", markersize=4)
    axes[0].set_yscale("log")
    axes[0].set_title("Train loss", fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss, log scale")
    axes[1].set_title("Validation MRR", fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MRR")
    axes[1].set_ylim(0.34, 0.44)
    for ax in axes:
        ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    legend_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.8, marker="o", markersize=5, label=label)
        for label, _path, color, linestyle in logs
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        frameon=False,
        handlelength=3.4,
        borderaxespad=0.8,
    )
    fig.suptitle("Training behavior on MS MARCO medium", fontsize=17, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.965,
        "Pairwise ranking loss decreases while validation MRR remains in the expected lightweight-reranker range. Dashed = PyTorch, solid = Jittor.",
        ha="center",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    save_figure(fig, "05_training_curves.png")


def main() -> None:
    rows = medium_rows()
    draw_scope_figure()
    draw_medium_metric_panels(rows)
    draw_ranked_mrr(rows)
    draw_subset_comparison(small_medium_rows())
    draw_training_curves()
    print(f"Generated README figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
