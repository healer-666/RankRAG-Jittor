"""Build the downstream RAG generator/prompt 2x2 ablation artifacts."""

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

METHODS = ["bm25", "lora_v3", "cross_encoder"]
METHOD_LABELS = {
    "bm25": "BM25",
    "lora_v3": "LoRA v3",
    "cross_encoder": "Cross-Encoder",
}

EXPERIMENTS = {
    "qwen2_1_5b_original": {
        "generator": "1.5B",
        "generator_name": "Qwen2.5-1.5B-Instruct",
        "prompt": "Original",
        "prompt_style": "original",
        "results_path": "outputs/downstream_rag_eval/downstream_rag_eval_results.json",
    },
    "qwen2_1_5b_strict": {
        "generator": "1.5B",
        "generator_name": "Qwen2.5-1.5B-Instruct",
        "prompt": "Strict",
        "prompt_style": "strict_short_answer_v1",
        "results_path": "outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/downstream_rag_eval_results.json",
        "protocol_check_path": "outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/protocol_consistency_check.json",
    },
    "qwen2_7b_original": {
        "generator": "7B",
        "generator_name": "Qwen2.5-7B-Instruct",
        "prompt": "Original",
        "prompt_style": "original",
        "results_path": "outputs/downstream_rag_eval_qwen2_7b/downstream_rag_eval_results.json",
    },
    "qwen2_7b_strict": {
        "generator": "7B",
        "generator_name": "Qwen2.5-7B-Instruct",
        "prompt": "Strict",
        "prompt_style": "strict_short_answer_v1",
        "results_path": "outputs/downstream_rag_eval_qwen2_7b_strict_prompt/downstream_rag_eval_results.json",
        "protocol_check_path": "outputs/downstream_rag_eval_qwen2_7b_strict_prompt/protocol_consistency_check.json",
    },
}


def methods_by_name(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {row["method"]: row for row in payload["methods"]}


def metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    return {key: metrics[key] for key in METRIC_KEYS}


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        key: after[key] - before[key]
        for key in METRIC_KEYS
        if isinstance(after[key], (int, float)) and isinstance(before[key], (int, float))
    }


def load_experiment_results(experiments: dict[str, dict[str, Any]] = EXPERIMENTS) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for exp_id, meta in experiments.items():
        rows = methods_by_name(resolve_path(meta["results_path"]))
        results[exp_id] = {
            "generator": meta["generator"],
            "generator_name": meta["generator_name"],
            "prompt": meta["prompt"],
            "prompt_style": meta["prompt_style"],
            "source": meta["results_path"],
            "methods": {method: metric_subset(rows[method]) for method in METHODS},
        }
    return results


def load_protocol_checks(experiments: dict[str, dict[str, Any]] = EXPERIMENTS) -> dict[str, Any]:
    checks = {}
    for exp_id, meta in experiments.items():
        path = meta.get("protocol_check_path")
        if path:
            checks[exp_id] = read_json(resolve_path(path))
    return checks


def build_comparisons(results: dict[str, Any]) -> dict[str, Any]:
    prompt_effects = {}
    scale_effects = {}
    gold_consistency = {}

    pairs = {
        "1.5B": ("qwen2_1_5b_original", "qwen2_1_5b_strict"),
        "7B": ("qwen2_7b_original", "qwen2_7b_strict"),
    }
    for generator, (original_id, strict_id) in pairs.items():
        prompt_effects[generator] = {
            method: delta(
                results[strict_id]["methods"][method],
                results[original_id]["methods"][method],
            )
            for method in METHODS
        }

    scale_pairs = {
        "Original": ("qwen2_1_5b_original", "qwen2_7b_original"),
        "Strict": ("qwen2_1_5b_strict", "qwen2_7b_strict"),
    }
    for prompt, (small_id, large_id) in scale_pairs.items():
        scale_effects[prompt] = {
            method: delta(
                results[large_id]["methods"][method],
                results[small_id]["methods"][method],
            )
            for method in METHODS
        }

    for method in METHODS:
        values = {
            exp_id: results[exp_id]["methods"][method]["gold_in_context"]
            for exp_id in EXPERIMENTS
        }
        gold_consistency[method] = {
            "values": values,
            "consistent": len(set(values.values())) == 1,
        }

    return {
        "prompt_effects_strict_minus_original": prompt_effects,
        "generator_effects_7b_minus_1_5b": scale_effects,
        "gold_in_context_consistency": gold_consistency,
    }


def build_payload() -> dict[str, Any]:
    results = load_experiment_results()
    protocol_checks = load_protocol_checks()
    checks_passed = all(check.get("status") == "passed" for check in protocol_checks.values())
    controls = {
        "same_rankings": all(check.get("same_rankings") for check in protocol_checks.values()),
        "same_questions": all(check.get("same_questions") for check in protocol_checks.values()),
        "same_contexts": all(
            check.get("same_top3_contexts") and check.get("same_context_order")
            for check in protocol_checks.values()
        ),
        "same_context_order": all(check.get("same_context_order") for check in protocol_checks.values()),
        "same_reranking_metrics": all(check.get("same_reranking_metrics") for check in protocol_checks.values()),
        "same_generation_settings_except_prompt": all(
            check.get("same_generation_settings_except_prompt") for check in protocol_checks.values()
        ),
    }
    comparisons = build_comparisons(results)
    controls["same_gold_in_context_within_reranker"] = all(
        row["consistent"] for row in comparisons["gold_in_context_consistency"].values()
    )

    return {
        "protocol": {
            "questions": 50,
            "top_k": 3,
            **controls,
            "protocol_checks_passed": checks_passed,
            "generators": [
                "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-7B-Instruct",
            ],
            "prompt_styles": [
                "original",
                "strict_short_answer_v1",
            ],
            "strict_prompt_version": "strict_short_answer_v1",
            "notes": [
                "Prompt comparisons keep questions, rankings, top-3 contexts, context order, decoding settings, and metrics fixed.",
                "Generator-scale comparisons keep the question set, rerankers, rankings, top-3 contexts, decoding settings, and metrics fixed while changing the generator.",
            ],
        },
        "protocol_checks": protocol_checks,
        "results": results,
        "comparisons": comparisons,
    }


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def table_row(method: str, exp: dict[str, Any]) -> str:
    metrics = exp["methods"][method]
    return (
        f"| {METHOD_LABELS[method]} | {exp['generator']} | {exp['prompt']} | "
        f"{metrics['gold_in_context']:.2f} | {metrics['answer_hit']:.2f} | "
        f"{metrics['exact_match']:.2f} | {fmt4(metrics['token_f1'])} | "
        f"{metrics['average_answer_length']:.2f} | {metrics['generation_failures']} | "
        f"{metrics['successes']} | {metrics['contains_insufficient_information_rate']:.2f} |"
    )


def result_table_lines(payload: dict[str, Any]) -> list[str]:
    results = payload["results"]
    lines = [
        "| Reranker | Generator | Prompt | Gold@3 | Answer Hit | EM | Token F1 | Avg Len | Gen Fail | Success | Contains Refusal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        for exp_id in [
            "qwen2_1_5b_original",
            "qwen2_1_5b_strict",
            "qwen2_7b_original",
            "qwen2_7b_strict",
        ]:
            lines.append(table_row(method, results[exp_id]))
    return lines


def build_markdown(payload: dict[str, Any], output_path: Path) -> None:
    results = payload["results"]
    prompt_effects = payload["comparisons"]["prompt_effects_strict_minus_original"]
    strict_scale = payload["comparisons"]["generator_effects_7b_minus_1_5b"]["Strict"]

    lines = [
        "# Downstream RAG Generator-Prompt 2x2 Ablation",
        "",
        "## 1. Experiment Questions",
        "",
        "This extension experiment asks two questions:",
        "",
        "- Does the strict short-answer prompt improve generation quality?",
        "- Does the prompt effect change with generator scale?",
        "",
        "## 2. Fixed Protocol",
        "",
        "The four formal runs use the same 50 MS MARCO questions, the same BM25, LoRA v3, and Cross-Encoder rerankers, the same ranking files, the same top-3 contexts, the same context order, the same decoding settings, and the same automatic metrics.",
        "",
        "Prompt comparisons only change the prompt. Generator-scale comparisons only change the generator. Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct were run on different hardware, but the committed results are deterministic offline generation outputs; hardware affects runtime, not the protocol semantics.",
        "",
        "This is a downstream prompt ablation, generator-scale sensitivity check, generation-format sensitivity check, and extension experiment. It is not a reranking leaderboard result.",
        "",
        "## 3. Complete 2x2 Results",
        "",
        *result_table_lines(payload),
        "",
        "## 4. Main Observations",
        "",
        "Prompt choice clearly affects final metrics, so the original-prompt results alone should not be used to judge generator capability.",
        "",
        "For Qwen2.5-1.5B-Instruct, the strict prompt provides stronger format control: Token F1 improves for all three rerankers, EM improves, average answer length drops, and contains-refusal rates are reduced to near zero.",
        "",
        "For Qwen2.5-7B-Instruct, the strict prompt also helps answer containment: BM25 Answer Hit changes from "
        f"{results['qwen2_7b_original']['methods']['bm25']['answer_hit']:.2f} to {results['qwen2_7b_strict']['methods']['bm25']['answer_hit']:.2f}, "
        f"LoRA v3 from {results['qwen2_7b_original']['methods']['lora_v3']['answer_hit']:.2f} to {results['qwen2_7b_strict']['methods']['lora_v3']['answer_hit']:.2f}, "
        f"and Cross-Encoder from {results['qwen2_7b_original']['methods']['cross_encoder']['answer_hit']:.2f} to {results['qwen2_7b_strict']['methods']['cross_encoder']['answer_hit']:.2f}. "
        "Generation failures decrease for all three rerankers, and average answer length decreases substantially.",
        "",
        "The 7B strict run still has visible output-format problems. Across rerankers, 60% to 62% of answers contain `Insufficient information`, and sampled outputs show repeated answers, multiple `Answer:` continuations, or a refusal phrase appended after a correct answer.",
        "",
        "Under the strict prompt, 7B has higher Answer Hit than 1.5B: BM25 "
        f"{results['qwen2_7b_strict']['methods']['bm25']['answer_hit']:.2f} vs {results['qwen2_1_5b_strict']['methods']['bm25']['answer_hit']:.2f}, "
        f"LoRA v3 {results['qwen2_7b_strict']['methods']['lora_v3']['answer_hit']:.2f} vs {results['qwen2_1_5b_strict']['methods']['lora_v3']['answer_hit']:.2f}, "
        f"and Cross-Encoder {results['qwen2_7b_strict']['methods']['cross_encoder']['answer_hit']:.2f} vs {results['qwen2_1_5b_strict']['methods']['cross_encoder']['answer_hit']:.2f}.",
        "",
        "However, 1.5B strict obtains better EM and Token F1. This suggests that 7B more often includes a correct answer somewhere in the output, while 1.5B more reliably follows the short-answer format. Prompt effects are therefore model-dependent, and natural-language prompting alone does not fully guarantee strict output formatting.",
        "",
        "Higher Gold@3 still does not fully convert into final answer quality: Cross-Encoder reaches Gold@3 = 0.88, while the best Answer Hit in the 2x2 table is 0.40.",
        "",
        "## 5. Conclusion Boundary",
        "",
        "Within the current fixed 50-question protocol, the strict short-answer prompt has measurable effects for both generators, but the effect is strongly model-dependent. Qwen2.5-1.5B-Instruct reaches higher EM and Token F1 under the strict prompt, while Qwen2.5-7B-Instruct more often includes the correct answer in its output but still frequently generates repeated answers or appended refusal phrases. Model scale, prompt design, and output-format control jointly determine the downstream RAG automatic evaluation results.",
        "",
        "These results should not be simplified to `1.5B is stronger than 7B` or `7B is stronger than 1.5B`. Answer Hit measures whether a gold answer appears in the generated text, while EM and Token F1 are more sensitive to formatting and extra text.",
        "",
        "## 6. Relation to RankRAG",
        "",
        "This is a follow-up extension experiment, not a same-protocol reproduction of the original RankRAG final QA setup. The current system uses fixed rerankers plus a general Qwen Instruct generator. RankRAG jointly trains ranking and generation. This experiment mainly analyzes the retrieval-generation interface, prompt sensitivity, and output-format failure modes in this reproduction system.",
        "",
        "## 7. Source Artifacts",
        "",
        "- `outputs/downstream_rag_prompt_ablation_2x2.json`",
        "- `outputs/downstream_rag_eval/downstream_rag_eval_results.json`",
        "- `outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/downstream_rag_eval_results.json`",
        "- `outputs/downstream_rag_eval_qwen2_7b/downstream_rag_eval_results.json`",
        "- `outputs/downstream_rag_eval_qwen2_7b_strict_prompt/downstream_rag_eval_results.json`",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="outputs/downstream_rag_prompt_ablation_2x2.json")
    parser.add_argument("--output-md", default="docs/downstream_rag_prompt_ablation_2x2.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    write_json(args.output_json, payload)
    build_markdown(payload, resolve_path(args.output_md))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
