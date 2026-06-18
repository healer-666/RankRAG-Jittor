"""Create PyTorch/Jittor metric alignment tables from evaluation JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import read_json, resolve_path, write_json


METRIC_ORDER = ["recall@1", "ndcg@1", "recall@3", "ndcg@3", "recall@5", "ndcg@5", "mrr", "pairwise_accuracy"]


def maybe_read_json(path: str | Path) -> dict[str, Any] | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    return read_json(resolved)


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_markdown(rows: list[dict[str, Any]], jittor_available: bool) -> str:
    lines = [
        "# PyTorch / Jittor Metrics Compare",
        "",
        "| Metric | PyTorch | Jittor |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {format_value(row['torch'])} | {format_value(row['jittor'])} |")

    if not jittor_available:
        lines.extend(["", "Jittor environment is not available yet."])
    return "\n".join(lines) + "\n"


def main() -> None:
    torch_metrics = maybe_read_json("outputs/torch_metrics.json")
    if torch_metrics is None:
        raise FileNotFoundError("outputs/torch_metrics.json not found. Run PyTorch eval first.")

    jittor_metrics = maybe_read_json("outputs/jittor_metrics.json")
    jittor_available = jittor_metrics is not None

    rows = []
    for metric in METRIC_ORDER:
        rows.append(
            {
                "metric": metric,
                "torch": torch_metrics.get(metric),
                "jittor": jittor_metrics.get(metric) if jittor_metrics else None,
            }
        )

    payload = {
        "status": "ready" if jittor_available else "partial",
        "note": "" if jittor_available else "Jittor environment is not available yet.",
        "rows": rows,
    }
    write_json("outputs/metrics_compare.json", payload)

    md_path = resolve_path("outputs/metrics_compare.md")
    md_path.write_text(build_markdown(rows, jittor_available), encoding="utf-8")

    if not jittor_available:
        print("Jittor environment is not available yet; wrote N/A for Jittor metrics.")
    print(f"Wrote {resolve_path('outputs/metrics_compare.json')}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
