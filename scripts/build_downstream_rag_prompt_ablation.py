"""Build Qwen2.5-1.5B downstream RAG prompt ablation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from src.utils import read_json, resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.utils import read_json, resolve_path, write_json


METRIC_KEYS = [
    "gold_in_context",
    "answer_hit",
    "exact_match",
    "token_f1",
    "average_answer_length",
    "retrieval_failures",
    "generation_failures",
    "successes",
    "exact_insufficient_information_rate",
    "starts_with_insufficient_information_rate",
    "contains_insufficient_information_rate",
]

METHOD_LABELS = {
    "bm25": "BM25",
    "lora_v3": "LoRA v3",
    "cross_encoder": "Cross-Encoder",
}


def methods_by_name(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {row["method"]: row for row in payload["methods"]}


def metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    return {key: metrics[key] for key in METRIC_KEYS}


def fmt_rate(value: float) -> str:
    return f"{value:.4f}"


def build_json(
    original_results: Path,
    strict_results: Path,
    protocol_check: Path,
    output_path: Path,
) -> dict[str, Any]:
    original = methods_by_name(original_results)
    strict = methods_by_name(strict_results)
    protocol = read_json(protocol_check)
    methods = list(strict)
    results = {}
    for method in methods:
        results[method] = {
            "original": metric_subset(original[method]),
            "strict": metric_subset(strict[method]),
            "delta": {
                key: metric_subset(strict[method])[key] - metric_subset(original[method])[key]
                for key in METRIC_KEYS
                if isinstance(metric_subset(strict[method])[key], (int, float))
            },
        }

    payload = {
        "protocol": {
            "generator": "Qwen2.5-1.5B-Instruct",
            "questions": 50,
            "top_k": 3,
            "same_rankings": protocol["same_rankings"],
            "same_contexts": protocol["same_top3_contexts"] and protocol["same_context_order"],
            "same_generation_settings_except_prompt": protocol["same_generation_settings_except_prompt"],
            "original_prompt": "original",
            "strict_prompt": "strict_short_answer_v1",
        },
        "results": results,
    }
    write_json(output_path, payload)
    return payload


def row_for_table(method_label: str, prompt_label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {method_label} | {prompt_label} | {fmt_rate(metrics['gold_in_context'])} | "
        f"{fmt_rate(metrics['answer_hit'])} | {fmt_rate(metrics['exact_match'])} | "
        f"{fmt_rate(metrics['token_f1'])} | {metrics['average_answer_length']:.2f} | "
        f"{metrics['generation_failures']} | {metrics['successes']} | "
        f"{fmt_rate(metrics['contains_insufficient_information_rate'])} |"
    )


def delta_word(value: float) -> str:
    if value > 0:
        return "increased"
    if value < 0:
        return "decreased"
    return "did not change"


def build_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Qwen2.5-1.5B Downstream RAG Prompt Ablation",
        "",
        "## 1. Research Question",
        "",
        "This extension experiment asks whether a strict short-answer prompt changes answer quality when the generator, rankings, top-3 contexts, context order, question set, decoding settings, and evaluation metrics are fixed.",
        "",
        "The only intended main variable is the prompt style. The protocol consistency check confirms whether the fixed inputs match before interpreting the results.",
        "",
        "## 2. Prompt Comparison",
        "",
        "The original prompt asks for a short, direct answer and uses `Insufficient information` when the contexts are insufficient.",
        "",
        "The strict prompt adds explicit output-format constraints:",
        "",
        "- output only the final answer",
        "- one line",
        "- no explanation, reasoning, notes, or citations",
        "- shortest answer span",
        "- use a directly answering context rather than refusing when at least one exists",
        "- resolve conflicts by choosing the most specific context matching the entities and intent",
        "- output the refusal phrase only when no context contains enough information",
        "- never append the refusal phrase after giving an answer",
        "",
        "## 3. Complete Results",
        "",
        "| Reranker | Prompt | Gold@3 | Answer Hit | EM | Token F1 | Avg. Length | Generation Fail | Success | Contains Refusal |",
        "| -------- | ------ | -----: | ---------: | -: | -------: | ----------: | --------------: | ------: | ---------------: |",
    ]
    for method in ["bm25", "lora_v3", "cross_encoder"]:
        label = METHOD_LABELS[method]
        lines.append(row_for_table(label, "original", payload["results"][method]["original"]))
        lines.append(row_for_table(label, "strict", payload["results"][method]["strict"]))

    lines.extend(["", "## 4. Main Observations", ""])
    for method in ["bm25", "lora_v3", "cross_encoder"]:
        label = METHOD_LABELS[method]
        delta = payload["results"][method]["delta"]
        lines.append(
            f"- {label}: Token F1 {delta_word(delta['token_f1'])} by {delta['token_f1']:+.4f}; "
            f"Answer Hit {delta_word(delta['answer_hit'])} by {delta['answer_hit']:+.4f}; "
            f"average answer length {delta_word(delta['average_answer_length'])} by {delta['average_answer_length']:+.2f} words; "
            f"contains-refusal rate {delta_word(delta['contains_insufficient_information_rate'])} by {delta['contains_insufficient_information_rate']:+.4f}; "
            f"generation failures {delta_word(delta['generation_failures'])} by {delta['generation_failures']:+.0f}."
        )

    strict_deltas = [payload["results"][method]["delta"] for method in ["bm25", "lora_v3", "cross_encoder"]]
    benefited = sum(1 for delta in strict_deltas if delta["token_f1"] > 0 and delta["answer_hit"] >= 0)
    lines.extend(
        [
            "",
            f"Across the three rerankers, {benefited}/3 show higher Token F1 without lower Answer Hit under the strict prompt.",
            "Gold@3 remains unchanged because rankings and top-3 contexts are fixed. Any metric movement after the protocol check is attributable to generation behavior under the changed prompt.",
            "",
            "## 5. Conclusion Boundary",
            "",
            "Within this fixed 50-question protocol, the strict short-answer prompt has a measurable effect on output length, refusal behavior, and automatic answer metrics.",
            "",
            "This is a downstream prompt ablation and generation-format sensitivity check. It is an extension experiment, not a reranking main result and not a same-protocol reproduction of RankRAG answer generation.",
            "",
            "The run only evaluates Qwen2.5-1.5B-Instruct. It does not establish a Qwen2.5-7B strict-prompt conclusion.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-results", default="outputs/downstream_rag_eval/downstream_rag_eval_results.json")
    parser.add_argument("--strict-results", default="outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/downstream_rag_eval_results.json")
    parser.add_argument("--protocol-check", default="outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/protocol_consistency_check.json")
    parser.add_argument("--output-json", default="outputs/downstream_rag_prompt_ablation_1_5b.json")
    parser.add_argument("--output-md", default="docs/downstream_rag_prompt_ablation_1_5b.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_json(
        resolve_path(args.original_results),
        resolve_path(args.strict_results),
        resolve_path(args.protocol_check),
        resolve_path(args.output_json),
    )
    build_markdown(payload, resolve_path(args.output_md))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
