"""Validate Stage G error taxonomy outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PRIMARY_LABELS = {
    "lexical_trap",
    "semantic_paraphrase",
    "background_only",
    "multi_evidence_confusion",
    "small_model_semantic_limit",
    "llm_overjudgment",
    "candidate_or_label_issue",
    "ambiguous_query",
    "evidence_utilization_failure",
}
BUCKETS = set("ABCDE")
FORBIDDEN_PATTERNS = ["*.safetensors", "checkpoint*", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_rank(value: str) -> bool:
    return value == "" or (str(value).isdigit() and int(value) > 0)


def main() -> None:
    errors = []
    required = [
        Path("docs/error_taxonomy.md"),
        Path("docs/figures/error_taxonomy.png"),
        Path("outputs/error_taxonomy_cases.csv"),
        Path("outputs/error_taxonomy_summary.json"),
        Path("outputs/error_analysis_input_audit.json"),
        Path("outputs/error_analysis_all_queries.csv"),
        Path("outputs/error_analysis_all_queries.json"),
        Path("outputs/error_analysis_selection_manifest.json"),
        Path("outputs/error_analysis_selected_cases_raw.csv"),
        Path("outputs/error_analysis_blinded_cases.csv"),
        Path("outputs/error_analysis_method_mapping.json"),
        Path("outputs/error_taxonomy_annotation_round1.csv"),
        Path("outputs/error_taxonomy_reannotation_input.csv"),
        Path("outputs/error_taxonomy_annotation_round2.csv"),
        Path("outputs/error_taxonomy_self_consistency.json"),
        Path("outputs/evidence_utilization_candidates.csv"),
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required output: {path.as_posix()}")
    if errors:
        raise SystemExit("\n".join(errors))
    cases = read_csv(Path("outputs/error_taxonomy_cases.csv"))
    summary = read_json(Path("outputs/error_taxonomy_summary.json"))
    mapping = read_json(Path("outputs/error_analysis_method_mapping.json"))
    reannotation = read_csv(Path("outputs/error_taxonomy_annotation_round2.csv"))
    all_queries = read_json(Path("outputs/error_analysis_all_queries.json"))
    if len(cases) < 25:
        errors.append(f"case count below diagnostic target without enough coverage: {len(cases)}")
    qids = [row["query_id"] for row in cases]
    if len(qids) != len(set(qids)):
        errors.append("duplicate query_id in final cases")
    all_qids = {row["query_id"] for row in all_queries}
    for row in cases:
        if not row["query"] or not row["gold_passage_text"] or not row["primary_error_type"] or not row["rationale"]:
            errors.append(f"{row['case_id']}: missing query/gold/label/rationale")
        if row["primary_error_type"] not in PRIMARY_LABELS:
            errors.append(f"{row['case_id']}: invalid primary label {row['primary_error_type']}")
        if row["selection_bucket"] not in BUCKETS:
            errors.append(f"{row['case_id']}: invalid bucket {row['selection_bucket']}")
        for method in ["bm25", "jittor_mlp", "jittor_textcnn", "zero_shot", "lora_10k_rerun", "cross_encoder"]:
            if not is_rank(row.get(f"{method}_gold_rank", "")):
                errors.append(f"{row['case_id']}: invalid rank for {method}")
            pid = row.get(f"{method}_top1_id", "")
            if pid and row["query_id"] not in pid:
                errors.append(f"{row['case_id']}: suspicious passage id for {method}: {pid}")
        if row["query_id"] not in all_qids:
            errors.append(f"{row['case_id']}: query not in all query table")
    if set(mapping) != {"Method A", "Method B", "Method C", "Method D", "Method E"}:
        errors.append("blinded mapping cannot restore five methods")
    expected_reannotation = min(10, len(cases))
    if len(reannotation) != expected_reannotation:
        errors.append(f"expected {expected_reannotation} reannotation rows, found {len(reannotation)}")
    if summary["case_count"] != len(cases) or summary["total_unique_queries"] != len(set(qids)):
        errors.append("summary count does not match cases")
    if not Path("docs/figures/error_taxonomy.png").exists():
        errors.append("missing taxonomy figure")
    for pattern in FORBIDDEN_PATTERNS:
        if list(Path(".").glob(pattern)):
            errors.append(f"forbidden large/model artifact matched {pattern}")
    payload = {"status": "passed" if not errors else "failed", "errors": errors, "case_count": len(cases)}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
