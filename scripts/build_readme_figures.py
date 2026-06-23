"""Build README figures from existing result artifacts.

This script is documentation-only. It reads committed JSON summaries and writes
static figures used by the README. It does not load models, train, or run
inference.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIG_DIR = Path("docs/figures")
MAIN_RESULTS = "outputs/final_results_summary.json"
COST_PROFILE = "outputs/cost_effectiveness_table.json"
LORA_ABLATION = "outputs/lora_ablation_results.json"
ERROR_SUMMARY = "outputs/error_taxonomy_summary.json"

PUBLIC_LABELS = {
    "BM25": "BM25",
    "Jittor MLP": "Jittor MLP",
    "Jittor TextCNN": "Jittor TextCNN",
    "Qwen2.5-1.5B zero-shot reranker": "Qwen zero-shot",
    "Qwen2.5-1.5B LoRA 10k-rerun": "Qwen LoRA",
    "Cross-Encoder": "Cross-Encoder",
}

ROLE_COLORS = {
    "BM25": "#6b7280",
    "Jittor MLP": "#2563eb",
    "Jittor TextCNN": "#38bdf8",
    "Qwen2.5-1.5B zero-shot reranker": "#7c3aed",
    "Qwen2.5-1.5B LoRA 10k-rerun": "#a855f7",
    "Cross-Encoder": "#f97316",
    "PyTorch": "#94a3b8",
    "Jittor": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def read_json(root: Path, rel_path: str) -> Any:
    return json.loads((root / rel_path).read_text(encoding="utf-8"))


def ensure_dir(root: Path) -> Path:
    path = root / FIG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(cleaned, encoding="utf-8")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=10)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fc: str,
    ec: str,
    text_color: str = "#0f172a",
    fontsize: int = 10,
    linestyle: str = "-",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="semibold",
        wrap=True,
    )
    return box


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#64748b",
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def add_elbow_arrow(ax: plt.Axes, points: list[tuple[float, float]], color: str = "#64748b") -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.4)
    add_arrow(ax, points[-2], points[-1], color)


def build_overview_figure(root: Path) -> list[str]:
    out_dir = ensure_dir(root)
    svg_path = out_dir / "rankrag_jittor_overview.svg"
    png_path = out_dir / "rankrag_jittor_overview.png"
    excalidraw_path = out_dir / "rankrag_jittor_overview.excalidraw"

    fig, ax = plt.subplots(figsize=(16, 8.2), dpi=200)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    panels = [
        (0.35, 0.45, 3.2, 7.2, "Input and candidate pool"),
        (4.0, 0.45, 7.4, 7.2, "Reranking body"),
        (11.85, 0.45, 3.8, 7.2, "Results and analysis"),
    ]
    for x, y, w, h, title in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.2,
                edgecolor="#dbe4ef",
                facecolor="#ffffff",
            )
        )
        ax.text(x + 0.22, y + h - 0.35, title, fontsize=10.5, color="#475569", fontweight="semibold")

    add_box(ax, 0.75, 6.15, 2.4, 0.62, "Query", "#eff6ff", "#2563eb", fontsize=11)
    add_box(ax, 0.75, 4.85, 2.4, 0.78, "Candidate\npassages", "#eff6ff", "#2563eb", fontsize=11)
    add_arrow(ax, (1.95, 6.13), (1.95, 5.65), "#2563eb")

    add_box(ax, 4.45, 6.15, 2.35, 0.64, "Framework alignment\nbaselines", "#dbeafe", "#2563eb", fontsize=9.5)
    add_box(ax, 4.35, 5.12, 1.35, 0.55, "PyTorch MLP", "#eff6ff", "#60a5fa", fontsize=8.7)
    add_box(ax, 5.95, 5.12, 1.35, 0.55, "Jittor MLP", "#dbeafe", "#2563eb", fontsize=8.7)
    add_box(ax, 4.35, 4.28, 1.35, 0.55, "PyTorch\nTextCNN", "#eff6ff", "#60a5fa", fontsize=8.4)
    add_box(ax, 5.95, 4.28, 1.35, 0.55, "Jittor\nTextCNN", "#dbeafe", "#2563eb", fontsize=8.4)

    add_box(ax, 7.65, 5.95, 2.35, 0.62, "Qwen zero-shot", "#f3e8ff", "#7c3aed", fontsize=10)
    add_box(ax, 7.65, 5.02, 2.35, 0.62, "Qwen LoRA", "#f3e8ff", "#a855f7", fontsize=10)
    add_box(ax, 7.65, 4.10, 2.35, 0.62, "LLM relevance\njudgment", "#faf5ff", "#a855f7", fontsize=9.2)
    add_box(ax, 7.65, 2.95, 2.35, 0.62, "Cross-Encoder\nreference", "#fff7ed", "#f97316", fontsize=9.2, linestyle="--")
    add_box(ax, 6.02, 2.0, 2.1, 0.68, "Reranking", "#f8fafc", "#64748b", fontsize=11)

    add_arrow(ax, (3.15, 5.28), (4.35, 5.42), "#2563eb")
    add_arrow(ax, (3.15, 4.98), (7.65, 6.22), "#7c3aed", rad=-0.18)
    add_arrow(ax, (7.30, 5.39), (7.00, 2.68), "#2563eb", rad=0.16)
    add_arrow(ax, (8.82, 4.10), (7.92, 2.68), "#a855f7")
    add_arrow(ax, (8.82, 2.95), (8.12, 2.42), "#f97316")

    add_box(ax, 12.25, 6.18, 2.8, 0.58, "Ranked passages", "#f8fafc", "#64748b", fontsize=10)
    add_box(ax, 12.25, 5.28, 2.8, 0.58, "Top-3 evidence", "#f8fafc", "#64748b", fontsize=10)
    add_box(ax, 12.25, 4.38, 2.8, 0.58, "Qwen generator", "#f3e8ff", "#7c3aed", fontsize=10)
    add_box(ax, 12.25, 3.48, 2.8, 0.58, "Answer", "#ecfdf5", "#10b981", fontsize=10)
    add_elbow_arrow(ax, [(8.12, 2.34), (11.55, 2.34), (11.55, 6.47), (12.25, 6.47)], "#64748b")
    add_arrow(ax, (13.65, 6.18), (13.65, 5.86), "#64748b")
    add_arrow(ax, (13.65, 5.28), (13.65, 4.96), "#64748b")
    add_arrow(ax, (13.65, 4.38), (13.65, 4.06), "#64748b")

    ax.text(12.15, 2.72, "Evaluation and diagnosis", fontsize=10.5, color="#475569", fontweight="semibold")
    chips = [
        ("Ranking metrics", 12.15, 2.20),
        ("Data-size ablation", 13.68, 2.20),
        ("Scoring ablation", 12.15, 1.65),
        ("Downstream RAG", 13.68, 1.65),
        ("Error taxonomy", 12.15, 1.10),
        ("Resource profile", 13.68, 1.10),
    ]
    for label, x, y in chips:
        add_box(ax, x, y, 1.36, 0.36, label, "#fff7ed", "#f59e0b", fontsize=7.7)

    ax.text(
        7.75,
        0.82,
        "Better ranking improves evidence access; answer generation can still fail.",
        ha="center",
        va="center",
        color="#475569",
        fontsize=9.4,
    )

    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    strip_trailing_whitespace(svg_path)

    write_excalidraw_overview(excalidraw_path)
    return [svg_path.as_posix(), png_path.as_posix(), excalidraw_path.as_posix()]


def excalidraw_element(element_type: str, x: float, y: float, w: float, h: float, **kwargs: Any) -> dict[str, Any]:
    return {
        "id": kwargs.pop("id", uuid.uuid4().hex[:20]),
        "type": element_type,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": kwargs.pop("strokeColor", "#1e293b"),
        "backgroundColor": kwargs.pop("backgroundColor", "transparent"),
        "fillStyle": "solid",
        "strokeWidth": kwargs.pop("strokeWidth", 2),
        "strokeStyle": kwargs.pop("strokeStyle", "solid"),
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": kwargs.pop("seed", 1),
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
        **kwargs,
    }


def text_element(text: str, x: float, y: float, font_size: int = 18, color: str = "#0f172a") -> dict[str, Any]:
    lines = text.split("\n")
    width = max(len(line) for line in lines) * font_size * 0.58
    height = len(lines) * font_size * 1.25
    return excalidraw_element(
        "text",
        x,
        y,
        width,
        height,
        strokeColor=color,
        text=text,
        fontSize=font_size,
        fontFamily=1,
        textAlign="center",
        verticalAlign="middle",
        containerId=None,
        originalText=text,
        lineHeight=1.25,
        roundness=None,
    )


def arrow_element(x1: float, y1: float, x2: float, y2: float, color: str) -> dict[str, Any]:
    return excalidraw_element(
        "arrow",
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        strokeColor=color,
        backgroundColor="transparent",
        points=[[0, 0], [x2 - x1, y2 - y1]],
        startBinding=None,
        endBinding=None,
        startArrowhead=None,
        endArrowhead="arrow",
        roundness={"type": 2},
    )


def write_excalidraw_overview(path: Path) -> None:
    elements: list[dict[str, Any]] = []
    panels = [
        (20, 20, 260, 620, "Input and candidate pool"),
        (320, 20, 600, 620, "Reranking body"),
        (960, 20, 320, 620, "Results and analysis"),
    ]
    for x, y, w, h, title in panels:
        elements.append(excalidraw_element("rectangle", x, y, w, h, strokeColor="#dbe4ef", backgroundColor="#ffffff"))
        elements.append(text_element(title, x + 18, y + 20, 16, "#475569"))
    boxes = [
        (55, 135, 190, 50, "Query", "#eff6ff", "#2563eb"),
        (55, 245, 190, 64, "Candidate\npassages", "#eff6ff", "#2563eb"),
        (360, 130, 190, 58, "Framework alignment\nbaselines", "#dbeafe", "#2563eb"),
        (350, 225, 110, 45, "PyTorch MLP", "#eff6ff", "#60a5fa"),
        (485, 225, 110, 45, "Jittor MLP", "#dbeafe", "#2563eb"),
        (350, 300, 110, 52, "PyTorch\nTextCNN", "#eff6ff", "#60a5fa"),
        (485, 300, 110, 52, "Jittor\nTextCNN", "#dbeafe", "#2563eb"),
        (635, 125, 190, 50, "Qwen zero-shot", "#f3e8ff", "#7c3aed"),
        (635, 205, 190, 50, "Qwen LoRA", "#f3e8ff", "#a855f7"),
        (635, 285, 190, 52, "LLM relevance\njudgment", "#faf5ff", "#a855f7"),
        (635, 385, 190, 52, "Cross-Encoder\nreference", "#fff7ed", "#f97316"),
        (500, 485, 170, 55, "Reranking", "#f8fafc", "#64748b"),
        (990, 132, 230, 46, "Ranked passages", "#f8fafc", "#64748b"),
        (990, 215, 230, 46, "Top-3 evidence", "#f8fafc", "#64748b"),
        (990, 298, 230, 46, "Qwen generator", "#f3e8ff", "#7c3aed"),
        (990, 381, 230, 46, "Answer", "#ecfdf5", "#10b981"),
    ]
    for x, y, w, h, label, fill, stroke in boxes:
        style = "dashed" if "Cross-Encoder" in label else "solid"
        elements.append(
            excalidraw_element(
                "rectangle",
                x,
                y,
                w,
                h,
                strokeColor=stroke,
                backgroundColor=fill,
                strokeStyle=style,
            )
        )
        elements.append(text_element(label, x + 8, y + 10, 15 if "\n" not in label else 13, "#0f172a"))
    for x1, y1, x2, y2, color in [
        (150, 185, 150, 245, "#2563eb"),
        (245, 275, 350, 247, "#2563eb"),
        (245, 305, 635, 150, "#7c3aed"),
        (595, 247, 585, 485, "#2563eb"),
        (730, 337, 660, 485, "#a855f7"),
        (730, 385, 670, 510, "#f97316"),
        (670, 510, 990, 155, "#64748b"),
        (1105, 178, 1105, 215, "#64748b"),
        (1105, 261, 1105, 298, "#64748b"),
        (1105, 344, 1105, 381, "#64748b"),
    ]:
        elements.append(arrow_element(x1, y1, x2, y2, color))
    elements.append(text_element("Evaluation and diagnosis", 990, 468, 16, "#475569"))
    for idx, label in enumerate(["Ranking metrics", "Data-size ablation", "Scoring ablation", "Downstream RAG", "Error taxonomy", "Resource profile"]):
        col = idx % 2
        row = idx // 2
        x = 990 + col * 125
        y = 505 + row * 42
        elements.append(excalidraw_element("rectangle", x, y, 112, 30, strokeColor="#f59e0b", backgroundColor="#fff7ed"))
        elements.append(text_element(label, x + 4, y + 7, 10, "#0f172a"))
    elements.append(text_element("Better ranking improves evidence access; answer generation can still fail.", 385, 590, 16, "#475569"))
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#f8fafc", "gridSize": None},
        "files": {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_main_results(root: Path) -> list[str]:
    out_dir = ensure_dir(root)
    final = read_json(root, MAIN_RESULTS)
    read_json(root, COST_PROFILE)
    read_json(root, LORA_ABLATION)
    read_json(root, ERROR_SUMMARY)
    rows = sorted(final["main_reranking_results"], key=lambda item: item["ndcg_at_5"])
    y_pos = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for idx, row in enumerate(rows):
        method = row["method"]
        color = ROLE_COLORS[method]
        r1 = row["recall_at_1"]
        ndcg = row["ndcg_at_5"]
        ax.plot([r1, ndcg], [idx, idx], color=color, linewidth=2.8, alpha=0.55)
        ax.scatter(r1, idx, s=72, color="#ffffff", edgecolor=color, linewidth=2.2, marker="o", zorder=3)
        ax.scatter(ndcg, idx, s=78, color=color, edgecolor="#ffffff", linewidth=1.0, marker="D", zorder=4)
        ax.text(r1 - 0.012, idx + 0.18, f"{r1:.3f}", ha="right", va="center", fontsize=9, color="#475569")
        ax.text(ndcg + 0.012, idx + 0.18, f"{ndcg:.3f}", ha="left", va="center", fontsize=9, color="#475569")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([PUBLIC_LABELS[row["method"]] for row in rows], fontsize=10.5)
    ax.set_xlim(0.0, 0.78)
    ax.set_xlabel("Metric value", fontsize=11, color="#334155")
    ax.set_title("Reranking Effectiveness on MS MARCO Medium", loc="left", fontsize=16, fontweight="bold", color="#0f172a", pad=20)
    ax.text(0, len(rows) - 0.15, "500 queries · 4,044 query-passage pairs", fontsize=10, color="#64748b")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#475569", label="Recall@1", markerfacecolor="#ffffff", markeredgewidth=2, linewidth=0),
            Line2D([0], [0], marker="D", color="#475569", label="NDCG@5", markerfacecolor="#475569", linewidth=0),
        ],
        loc="lower right",
        frameon=False,
        fontsize=10,
    )
    style_axis(ax)
    fig.tight_layout()
    svg_path = out_dir / "main_reranking_results.svg"
    png_path = out_dir / "main_reranking_results.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    strip_trailing_whitespace(svg_path)
    return [svg_path.as_posix(), png_path.as_posix()]


def build_alignment(root: Path) -> list[str]:
    out_dir = ensure_dir(root)
    final = read_json(root, MAIN_RESULTS)
    rows = final.get("pytorch_jittor_alignment", [])
    if not rows:
        return []
    metrics = ["recall_at_1", "ndcg_at_5"]
    labels = {"recall_at_1": "Recall@1", "ndcg_at_5": "NDCG@5"}
    models = ["MLP", "TextCNN"]
    by_key = {(row["model"], row["framework"]): row for row in rows}
    if not all((model, fw) in by_key for model in models for fw in ["PyTorch", "Jittor"]):
        return []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=220, sharex=True)
    fig.patch.set_facecolor("#ffffff")
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#ffffff")
        for idx, model in enumerate(models):
            torch_value = by_key[(model, "PyTorch")][metric]
            jittor_value = by_key[(model, "Jittor")][metric]
            y = len(models) - 1 - idx
            ax.plot([torch_value, jittor_value], [y, y], color="#94a3b8", linewidth=2.4, alpha=0.7)
            ax.scatter(torch_value, y, s=86, color="#ffffff", edgecolor=ROLE_COLORS["PyTorch"], linewidth=2.2, marker="o", zorder=3)
            ax.scatter(jittor_value, y, s=86, color=ROLE_COLORS["Jittor"], edgecolor="#ffffff", linewidth=1.1, marker="D", zorder=4)
            ax.text(torch_value - 0.010, y + 0.16, f"{torch_value:.3f}", ha="right", va="center", fontsize=9, color="#64748b")
            ax.text(jittor_value + 0.010, y + 0.16, f"{jittor_value:.3f}", ha="left", va="center", fontsize=9, color="#2563eb")
        ax.set_yticks([1, 0])
        ax.set_yticklabels(models, fontsize=10.5)
        ax.set_xlim(0.15, 0.50)
        ax.set_ylim(-0.35, 1.35)
        ax.set_title(labels[metric], fontsize=13, fontweight="bold", color="#0f172a")
        style_axis(ax)
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="#94a3b8", label="PyTorch", markerfacecolor="#ffffff", markeredgewidth=2, linewidth=0),
            Line2D([0], [0], marker="D", color="#2563eb", label="Jittor", markerfacecolor="#2563eb", linewidth=0),
        ],
        loc="upper right",
        frameon=False,
        fontsize=10,
    )
    fig.suptitle("PyTorch/Jittor Alignment Baselines", x=0.02, y=1.03, ha="left", fontsize=16, fontweight="bold", color="#0f172a")
    fig.text(0.02, 0.94, "Same data and metrics; trend alignment matters more than exact equality.", fontsize=10, color="#64748b")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    svg_path = out_dir / "pytorch_jittor_alignment.svg"
    png_path = out_dir / "pytorch_jittor_alignment.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    strip_trailing_whitespace(svg_path)
    return [svg_path.as_posix(), png_path.as_posix()]


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    generated = []
    generated.extend(build_overview_figure(root))
    generated.extend(build_main_results(root))
    generated.extend(build_alignment(root))
    print(json.dumps({"status": "ready", "generated": generated}, indent=2))


if __name__ == "__main__":
    main()
