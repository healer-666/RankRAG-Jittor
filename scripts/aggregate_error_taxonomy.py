"""Aggregate Stage G error taxonomy outputs and write the report figure/docs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PRIMARY_LABELS = [
    "lexical_trap",
    "semantic_paraphrase",
    "background_only",
    "multi_evidence_confusion",
    "small_model_semantic_limit",
    "llm_overjudgment",
    "candidate_or_label_issue",
    "ambiguous_query",
    "evidence_utilization_failure",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rank(row: dict[str, Any], method: str) -> int | None:
    value = row.get(f"{method}_gold_rank", "")
    if value in ("", None):
        return None
    return int(float(value))


def top1_failure(row: dict[str, Any], method: str) -> bool:
    return rank(row, method) != 1


def top3_failure(row: dict[str, Any], method: str) -> bool:
    value = rank(row, method)
    return value is None or value > 3


def count(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_summary() -> dict[str, Any]:
    cases = read_csv(Path("outputs/error_taxonomy_cases.csv"))
    consistency = read_json(Path("outputs/error_taxonomy_self_consistency.json"))
    summary = {
        "total_unique_queries": len({row["query_id"] for row in cases}),
        "case_count": len(cases),
        "selection_note": "Stratified diagnostic sample, not an unbiased distribution over all 500 queries.",
        "method_failure_definitions": {
            "top1_failure": "gold rank != 1 or gold missing",
            "top3_failure": "gold rank > 3 or gold missing",
        },
        "cases_by_selection_bucket": count(Counter(row["selection_bucket"] for row in cases)),
        "cases_by_primary_error_type": count(Counter(row["primary_error_type"] for row in cases)),
        "cases_by_secondary_subtype": count(Counter(row["secondary_subtype"] or "none" for row in cases)),
        "cases_by_confidence": count(Counter(row["confidence"] for row in cases)),
        "self_consistency_sample_size": consistency["sample_size"],
        "primary_label_agreement": consistency["primary_label_agreement"],
        "secondary_label_agreement": consistency["secondary_subtype_agreement"],
        "changed_case_ids": consistency["changed_case_ids"],
        "top1_failure_counts": {},
        "top3_failure_counts": {},
        "all_methods_failed_count": 0,
        "evidence_utilization_failure_count": sum(1 for row in cases if row["primary_error_type"] == "evidence_utilization_failure"),
    }
    methods = {
        "bm25": "bm25",
        "jittor_mlp": "jittor_mlp",
        "jittor_textcnn": "jittor_textcnn",
        "lora": "lora_10k_rerun",
        "cross_encoder": "cross_encoder",
    }
    for public, column in methods.items():
        summary["top1_failure_counts"][public] = sum(1 for row in cases if top1_failure(row, column))
        summary["top3_failure_counts"][public] = sum(1 for row in cases if top3_failure(row, column))
    summary["bm25_failure_count"] = summary["top3_failure_counts"]["bm25"]
    summary["jittor_mlp_failure_count"] = summary["top3_failure_counts"]["jittor_mlp"]
    summary["jittor_textcnn_failure_count"] = summary["top3_failure_counts"]["jittor_textcnn"]
    summary["lora_failure_count"] = summary["top3_failure_counts"]["lora"]
    summary["cross_encoder_failure_count"] = summary["top3_failure_counts"]["cross_encoder"]
    summary["all_methods_failed_count"] = sum(
        1
        for row in cases
        if all(top3_failure(row, method) for method in ["bm25", "jittor_mlp", "jittor_textcnn", "lora_10k_rerun", "cross_encoder"])
    )
    write_json(Path("outputs/error_taxonomy_summary.json"), summary)
    return summary


def write_figure(summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = summary["cases_by_primary_error_type"]
    labels = [label for label in PRIMARY_LABELS if counts.get(label, 0)]
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Case count")
    ax.set_title("Stage G diagnostic error taxonomy")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom", fontsize=9)
    fig.text(0.02, 0.01, f"n = {summary['total_unique_queries']} unique queries; stratified diagnostic sample, not a population error-rate estimate.", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("docs/figures/error_taxonomy.png", dpi=220)
    plt.close(fig)


def short(text: str, limit: int = 130) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def taxonomy_definitions() -> str:
    return """## 3. Error Taxonomy

- `lexical_trap`: the wrong passage shares surface keywords or entities with the query but does not answer the requested relation.
- `semantic_paraphrase`: the gold passage answers through paraphrase, synonymy, or indirect wording that surface matching underweights.
- `background_only`: the wrong passage is on-topic background but lacks the specific answer.
- `multi_evidence_confusion`: several passages are partially relevant, but the top passage is less complete than the gold evidence.
- `small_model_semantic_limit`: lightweight MLP/TextCNN rankings miss a fine-grained semantic relation that stronger rerankers recover.
- `llm_overjudgment`: an LLM-style reranker overvalues plausible, fluent, or weakly related evidence.
- `candidate_or_label_issue`: the candidate pool or gold label is materially debatable.
- `ambiguous_query`: the query lacks enough constraints, allowing multiple reasonable interpretations.
- `evidence_utilization_failure`: gold evidence is in downstream top-k, but the generator ignores it, uses the wrong evidence, or fails the answer format/metric.
"""


def write_doc(summary: dict[str, Any]) -> None:
    cases = read_csv(Path("outputs/error_taxonomy_cases.csv"))
    manifest = read_json(Path("outputs/error_analysis_selection_manifest.json"))
    audit = read_json(Path("outputs/error_analysis_input_audit.json"))
    representative = []

    def add_case(predicate) -> None:
        for case in cases:
            if case in representative:
                continue
            if predicate(case):
                representative.append(case)
                return

    add_case(lambda row: row["selection_bucket"] == "A")
    add_case(lambda row: row["selection_bucket"] == "B")
    add_case(lambda row: row["selection_bucket"] == "C" and row["lora_10k_rerun_gold_rank"] == "1")
    add_case(lambda row: row["selection_bucket"] == "C" and row["cross_encoder_gold_rank"] == "1")
    add_case(lambda row: row["selection_bucket"] == "D")
    add_case(lambda row: row["selection_bucket"] == "E")
    for label in PRIMARY_LABELS:
        add_case(lambda row, label=label: row["primary_error_type"] == label)
        if len(representative) >= 8:
            break
    lines = [
        "# Stage G: Error Taxonomy Analysis",
        "",
        "## 1. Purpose",
        "",
        "This analysis explains why existing reranking and downstream answer-generation results fail on selected diagnostic cases. It does not rerun models, does not train, and does not claim a full reproduction of RankRAG; the project remains a Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.",
        "",
        "## 2. Data and Case Selection",
        "",
        f"- Ranking sources audited: {len(audit['rankings'])}.",
        f"- Unified query rows: {audit['test_queries']['rows']}.",
        f"- Selected diagnostic cases: {summary['case_count']} unique queries.",
        f"- Seed: {manifest['seed']}.",
        "- Selection buckets: A BM25 lexical/semantic gap, B BM25 right but neural wrong, C LoRA/Cross-Encoder divergence, D all core methods fail, E evidence in top-3 but generation fails.",
        "- The sample is stratified for diagnosis. It is not an unbiased estimate of error proportions over all 500 test queries.",
        "",
        taxonomy_definitions(),
        "## 4. Annotation Procedure",
        "",
        "Method names were first blinded as Method A-E. Round 1 assigned one primary type per case, with an optional secondary subtype for evidence-utilization cases. A seed=42 sample of 10 cases was reread for repeated annotation self-consistency. This is single-annotator self-consistency, not inter-annotator or human-agreement measurement.",
        "",
        "## 5. Quantitative Summary",
        "",
        f"- Cases by bucket: `{summary['cases_by_selection_bucket']}`.",
        f"- Cases by primary type: `{summary['cases_by_primary_error_type']}`.",
        f"- Confidence distribution: `{summary['cases_by_confidence']}`.",
        f"- Reannotation sample size: {summary['self_consistency_sample_size']}.",
        f"- Primary label agreement: {summary['primary_label_agreement']:.3f}.",
        f"- Secondary subtype agreement: {summary['secondary_label_agreement']:.3f}.",
        "",
        "## 6. Representative Cases",
        "",
        "The table intentionally spans the five selection buckets instead of listing the easiest examples only.",
        "",
        "| Case | Query ID | Bucket | Query | Error type | Judgment |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in representative:
        lines.append(
            f"| {row['case_id']} | {row['query_id']} | {row['selection_bucket']} | {short(row['query'], 80)} | {row['primary_error_type']} | {short(row['rationale'], 160)} |"
        )
    lines += [
        "",
        "Full passages and method-specific top-1 evidence are kept in `outputs/error_taxonomy_cases.csv`.",
        "",
        "## 7. Main Findings",
        "",
        "- BM25 failures in the selected sample mostly appear when surface overlap points to a topically plausible but answer-incomplete passage.",
        "- Jittor MLP/TextCNN failures are concentrated in cases where BM25 or stronger semantic rerankers can still identify the gold passage, indicating limited fine-grained semantic discrimination in the lightweight models.",
        "- LoRA 10k-rerun improves many lexical mismatch cases over BM25, but selected divergence cases show that it can still overvalue weakly related passages.",
        "- Cross-Encoder is often the stronger semantic reranker in the selected divergence cases, but it is not immune to multi-evidence confusion or ambiguous labels.",
        "- When all core methods fail, the case often needs review for candidate-pool, label, or query ambiguity rather than only model capacity.",
        "- Evidence-utilization failures show that ranking success does not guarantee answer success: the generator can use a wrong top passage, ignore gold evidence in context, or produce a metric-mismatched answer.",
        "",
        "## 8. Limitations",
        "",
        "- The diagnostic sample is about 30 cases and intentionally stratified.",
        "- Counts in this document describe selected cases only, not global error rates.",
        "- There is one annotator; self-consistency is not inter-annotator agreement.",
        "- Some methods have richer per-query outputs than others.",
        "- Some gold labels and short queries remain debatable.",
        "- Downstream evidence-utilization analysis is limited to the existing Qwen2.5-1.5B strict-short-answer slice.",
    ]
    Path("docs/error_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    write_figure(summary)
    write_doc(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
