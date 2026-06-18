"""Create PyTorch/Jittor metric alignment tables from evaluation JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import read_json, resolve_path, write_json


METRIC_ORDER = ["recall@1", "ndcg@1", "recall@3", "ndcg@3", "recall@5", "ndcg@5", "mrr", "pairwise_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="synthetic", choices=["synthetic", "msmarco"])
    return parser.parse_args()


def paths_for_run(run_name: str) -> dict[str, str]:
    if run_name == "msmarco":
        return {
            "torch_metrics": "outputs/msmarco_torch_metrics.json",
            "jittor_metrics": "outputs/msmarco_jittor_metrics.json",
            "compare_json": "outputs/msmarco_metrics_compare.json",
            "compare_md": "outputs/msmarco_metrics_compare.md",
        }
    return {
        "torch_metrics": "outputs/torch_metrics.json",
        "jittor_metrics": "outputs/jittor_metrics.json",
        "compare_json": "outputs/metrics_compare.json",
        "compare_md": "outputs/metrics_compare.md",
    }


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
        "| Metric | PyTorch | Jittor | Diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {format_value(row['torch'])} | {format_value(row['jittor'])} | {format_value(row['diff'])} |"
        )

    if not jittor_available:
        lines.extend(["", "Jittor environment is not available yet."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = paths_for_run(args.run_name)

    torch_metrics = maybe_read_json(paths["torch_metrics"])
    if torch_metrics is None:
        raise FileNotFoundError(f"{paths['torch_metrics']} not found. Run PyTorch eval first.")

    jittor_metrics = maybe_read_json(paths["jittor_metrics"])
    jittor_available = jittor_metrics is not None

    rows = []
    for metric in METRIC_ORDER:
        torch_value = torch_metrics.get(metric)
        jittor_value = jittor_metrics.get(metric) if jittor_metrics else None
        diff = None if torch_value is None or jittor_value is None else float(jittor_value) - float(torch_value)
        rows.append(
            {
                "metric": metric,
                "torch": torch_value,
                "jittor": jittor_value,
                "diff": diff,
            }
        )

    payload = {
        "status": "ready" if jittor_available else "partial",
        "note": "" if jittor_available else "Jittor environment is not available yet.",
        "rows": rows,
    }
    write_json(paths["compare_json"], payload)

    md_path = resolve_path(paths["compare_md"])
    md_path.write_text(build_markdown(rows, jittor_available), encoding="utf-8")

    if not jittor_available:
        print("Jittor environment is not available yet; wrote N/A for Jittor metrics.")
    print(f"Wrote {resolve_path(paths['compare_json'])}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
