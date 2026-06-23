"""Validate Stage F resource-effectiveness outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METHODS = [
    "BM25",
    "Jittor MLP",
    "Jittor TextCNN",
    "Qwen2.5-1.5B zero-shot reranker",
    "Qwen2.5-1.5B LoRA 10k-rerun",
    "Cross-Encoder",
]
RESOURCE_FIELDS = [
    "parameter_count",
    "training_hardware",
    "evaluation_hardware",
    "training_runtime_sec",
    "evaluation_runtime_sec",
    "peak_gpu_memory_mib",
    "peak_cpu_memory_mib",
]
METRIC_FIELDS = ["recall_at_1", "recall_at_3", "recall_at_5", "ndcg_at_5", "mrr", "pairwise_accuracy"]
ALLOWED_RUNTIME_SCOPES = {"training_only", "scoring_only", "end_to_end_evaluation", "index_building", "unknown"}
ALLOWED_MEASUREMENT_STATUS = {"measured", "derived_from_measured", "metadata_only", "not_recorded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--table-json", default="outputs/cost_effectiveness_table.json")
    parser.add_argument("--table-csv", default="outputs/cost_effectiveness_table.csv")
    parser.add_argument("--table-md", default="outputs/cost_effectiveness_table.md")
    parser.add_argument("--audit-json", default="outputs/cost_effectiveness_input_audit.json")
    parser.add_argument("--comparability-json", default="outputs/cost_effectiveness_comparability.json")
    parser.add_argument("--doc", default="docs/cost_effectiveness_analysis.md")
    parser.add_argument("--figure", default="docs/figures/cost_effectiveness_tradeoff.png")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def assert_path(root: Path, errors: list[str], path_text: str) -> None:
    if path_text == "not recorded":
        return
    if not (root / path_text).exists():
        fail(errors, f"evidence path missing: {path_text}")


def parse_md_metrics(md_text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in md_text.splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "Method" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        method = parts[0]
        if method in METHODS:
            rows[method] = {
                "recall_at_1": float(parts[1]),
                "mrr": float(parts[2]),
                "ndcg_at_5": float(parts[3]),
            }
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    errors: list[str] = []

    table_path = root / args.table_json
    csv_path = root / args.table_csv
    md_path = root / args.table_md
    audit_path = root / args.audit_json
    comparability_path = root / args.comparability_json
    doc_path = root / args.doc
    figure_path = root / args.figure

    for path in [table_path, csv_path, md_path, audit_path, comparability_path, doc_path, figure_path]:
        if not path.exists():
            fail(errors, f"required output missing: {path.relative_to(root)}")

    if errors:
        raise SystemExit("\n".join(errors))

    table = read_json(table_path)
    audit = read_json(audit_path)
    comparability = read_json(comparability_path)
    rows = table.get("methods", [])
    by_method = {row.get("method"): row for row in rows}

    if table.get("status") != "ready":
        fail(errors, "table status is not ready")
    if table.get("title") != "Resource and Effectiveness Profile":
        fail(errors, "title mismatch")
    if table.get("generated_from_existing_artifacts_only") is not True:
        fail(errors, "table must declare existing-artifacts-only generation")
    if list(by_method) != METHODS:
        fail(errors, f"method order/name mismatch: {list(by_method)}")

    for method in METHODS:
        row = by_method.get(method)
        if not row:
            continue
        if row.get("query_count") != 500 or row.get("pair_count") != 4044:
            fail(errors, f"{method}: expected 500 queries and 4044 pairs")
        if row.get("effectiveness_comparable") is not True:
            fail(errors, f"{method}: effectiveness_comparable must be true")
        if not isinstance(row.get("resource_strictly_comparable"), bool):
            fail(errors, f"{method}: resource_strictly_comparable must be boolean")
        if row.get("runtime_scope") not in ALLOWED_RUNTIME_SCOPES:
            fail(errors, f"{method}: invalid runtime_scope")
        if row.get("measurement_status") not in ALLOWED_MEASUREMENT_STATUS:
            fail(errors, f"{method}: invalid measurement_status")
        for metric in METRIC_FIELDS:
            if not isinstance(row.get(metric), (int, float)):
                fail(errors, f"{method}: missing formal metric {metric}")
        for field in RESOURCE_FIELDS:
            value = row.get(field)
            if value in (None, "", "unknown"):
                fail(errors, f"{method}: resource field {field} must use 'not recorded', N/A, or a measured value")
            if value == 0 and method != "BM25":
                fail(errors, f"{method}: zero used as missing value for {field}")
        for evidence in row.get("evidence_paths", []):
            assert_path(root, errors, evidence)

    bm25 = by_method.get("BM25", {})
    if bm25.get("parameter_count") != "N/A":
        fail(errors, "BM25 parameter_count must be N/A")

    lora = by_method.get("Qwen2.5-1.5B LoRA 10k-rerun", {})
    lora_paths = " ".join(lora.get("evidence_paths", []))
    if "outputs/lora_ablation_data_10k_rerun" not in lora_paths:
        fail(errors, "LoRA main method must use 10k-rerun output path")
    if "outputs/lora_qwen_1_5b_10k_lr1e4_s800/" in lora_paths:
        fail(errors, "historical 10k path appears in main LoRA evidence")
    if lora.get("base_model") != "Qwen/Qwen2.5-1.5B-Instruct":
        fail(errors, "LoRA base model mismatch")

    if "outputs/lora_qwen_1_5b_10k_lr1e4_s800" in [row.get("method") for row in rows]:
        fail(errors, "historical 10k appears as a main method")

    if comparability.get("strictly_comparable_method_count") != 0:
        fail(errors, "strictly comparable method count should be zero for heterogeneous resource records")
    if comparability.get("figure_type") != "non_strict_resource_profile":
        fail(errors, "figure must be a non-strict resource profile")
    if len(comparability.get("effectiveness_comparable_methods", [])) != 6:
        fail(errors, "all six methods should be effectiveness comparable")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 6:
        fail(errors, "CSV must contain six rows")
    csv_by_method = {row["method"]: row for row in csv_rows}
    for method, row in by_method.items():
        csv_row = csv_by_method.get(method)
        if not csv_row:
            fail(errors, f"CSV missing method {method}")
            continue
        for metric in ["recall_at_1", "mrr", "ndcg_at_5"]:
            if abs(float(csv_row[metric]) - float(row[metric])) > 1e-12:
                fail(errors, f"CSV/JSON mismatch for {method} {metric}")

    md_rows = parse_md_metrics(md_path.read_text(encoding="utf-8"))
    for method, row in by_method.items():
        md_row = md_rows.get(method)
        if not md_row:
            fail(errors, f"Markdown missing method {method}")
            continue
        for metric in ["recall_at_1", "mrr", "ndcg_at_5"]:
            if abs(md_row[metric] - round(float(row[metric]), 4)) > 1e-12:
                fail(errors, f"Markdown/JSON mismatch for {method} {metric}")

    audit_methods = {item.get("method_name"): item for item in audit.get("methods", [])}
    if set(audit_methods) != set(METHODS):
        fail(errors, "audit methods mismatch")
    for method, item in audit_methods.items():
        if item.get("query_count") != 500 or item.get("pair_count") != 4044:
            fail(errors, f"audit {method}: query/pair count mismatch")
        for key, record in item.get("file_sha256", {}).items():
            if record.get("path") != "not recorded" and record.get("exists") is True:
                assert_path(root, errors, record["path"])
                if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
                    fail(errors, f"audit {method}: invalid sha256 for {key}")

    doc_text = doc_path.read_text(encoding="utf-8")
    required_doc_phrases = [
        "Resource and Effectiveness Profile",
        "not a strict hardware-normalized benchmark",
        "outputs/lora_ablation_data_10k_rerun/",
        "Historical 10k Reproducibility Reference",
        "Runtime and memory",
    ]
    for phrase in required_doc_phrases:
        if phrase not in doc_text:
            fail(errors, f"doc missing phrase: {phrase}")
    if figure_path.stat().st_size <= 0:
        fail(errors, "figure is empty")

    forbidden_refs = ["adapter_model.safetensors", "trainer_state.json", "rng_state", "optimizer.pt", "scheduler.pt"]
    generated_text = "\n".join(
        [
            table_path.read_text(encoding="utf-8"),
            csv_path.read_text(encoding="utf-8"),
            md_path.read_text(encoding="utf-8"),
            audit_path.read_text(encoding="utf-8"),
            comparability_path.read_text(encoding="utf-8"),
            doc_text,
        ]
    )
    for ref in forbidden_refs:
        if ref in generated_text:
            fail(errors, f"generated outputs reference forbidden weight/checkpoint artifact: {ref}")

    if errors:
        raise SystemExit("Stage F output check failed:\n" + "\n".join(f"- {error}" for error in errors))
    print("Stage F output check passed.")


if __name__ == "__main__":
    main()
