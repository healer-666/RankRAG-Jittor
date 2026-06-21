"""Evaluate JittorLLM zero-shot reranking on grouped MS MARCO JSONL data.

The script is intentionally conservative: if the local JittorLLM inference
environment is unavailable, it writes blocked outputs instead of fabricating
scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metrics import evaluate_grouped
from utils import ensure_parent, resolve_path, write_json
from jittorllm_reranker.prompt_utils import DEFAULT_PROMPT_TEMPLATE, build_prompt, parse_relevance_label


METRIC_KEYS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "mrr",
    "pairwise_accuracy",
]
JITTORLLM_MODULE_CANDIDATES = ["jittorllm", "jittor_llm", "jittorllms", "jittor_llms", "jtllm"]


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class ScoreResult:
    score: float
    raw_output: str
    score_source: str
    parse_fail: bool


class JittorLLMBackend:
    """Thin adapter placeholder for local JittorLLM text generation/logprob APIs."""

    def __init__(self, model_name: str, max_length: int) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.module_name = self._find_module()

    def _find_module(self) -> str:
        for module_name in JITTORLLM_MODULE_CANDIDATES:
            if importlib.util.find_spec(module_name) is not None:
                return module_name
        raise BackendUnavailable(
            "No JittorLLM Python package was found. Checked modules: "
            + ", ".join(JITTORLLM_MODULE_CANDIDATES)
            + ". Install/configure JittorLLM and a local model before running real zero-shot inference."
        )

    def score(self, prompt: str) -> ScoreResult:
        raise BackendUnavailable(
            f"Found JittorLLM-like module `{self.module_name}`, but this project does not yet have a verified adapter "
            "for its generation/logprob API. Please add a backend adapter that returns either label token logprobs "
            "or generated text labels."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jittorllm_zeroshot_medium.yaml")
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--output_metrics", default=None)
    parser.add_argument("--output_rankings", default=None)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--run_summary", default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def choose(args_value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    if args_value is not None:
        return args_value
    return config.get(key, default)


def read_jsonl(path: str | Path) -> list[dict]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def candidate_text(candidate: dict) -> str:
    title = candidate.get("title", "")
    text = candidate.get("text", "")
    return f"{title}. {text}" if title else str(text)


def cache_key(query_id: str, doc_id: str) -> str:
    return f"{query_id}\t{doc_id}"


def load_cache(path: str | Path) -> dict[str, dict]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}
    cache = {}
    with resolved.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[cache_key(row["query_id"], row["doc_id"])] = row
    return cache


def append_cache(path: str | Path, row: dict) -> None:
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_pairs(records: list[dict]) -> int:
    total = 0
    for record in records:
        labels = [int(candidate.get("label", 0)) for candidate in record.get("candidates", [])]
        total += sum(labels) * (len(labels) - sum(labels))
    return total


def build_rankings(records: list[dict], scored_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in scored_rows:
        grouped.setdefault(row["query_id"], []).append(row)

    rankings = []
    for record in records:
        rows = grouped.get(record["query_id"], [])
        ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
        rankings.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "ranking": [
                    {
                        "rank": index,
                        "doc_id": row["doc_id"],
                        "label": row["label"],
                        "score": row["score"],
                        "score_source": row["score_source"],
                        "parse_fail": row["parse_fail"],
                        "raw_output": row["raw_output"],
                        "text": row["text"],
                    }
                    for index, row in enumerate(ranked, start=1)
                ],
            }
        )
    return rankings


def null_metrics(status: str, reason: str, model_name: str, records: list[dict], elapsed: float) -> dict:
    payload = {metric: None for metric in METRIC_KEYS}
    payload.update(
        {
            "status": status,
            "blocked_reason": reason,
            "model_name": model_name,
            "num_queries": 0,
            "num_pairs": 0,
            "candidate_queries_loaded": len(records),
            "candidate_pairs_loaded": count_pairs(records),
            "inference_time_sec": elapsed,
            "pairs_per_second": 0.0,
            "parse_fail_count": 0,
            "parse_fail_rate": None,
            "logprob_scoring_available": False,
            "fallback_generated_label_scoring": False,
        }
    )
    return payload


def write_summary(path: str | Path, metrics: dict, prompt_template: str, test_path: str, cache_path: str, notes: str) -> None:
    lines = [
        "# JittorLLM Zero-shot Reranking Run Summary",
        "",
        f"- Status: {metrics.get('status')}",
        f"- Model/backend: {metrics.get('model_name')}",
        f"- Dataset: `{test_path}`",
        f"- Cache path: `{cache_path}`",
        f"- Num queries: {metrics.get('num_queries')}",
        f"- Num pairs: {metrics.get('num_pairs')}",
        f"- Inference time sec: {metrics.get('inference_time_sec')}",
        f"- Pairs per second: {metrics.get('pairs_per_second')}",
        f"- Logprob scoring available: {metrics.get('logprob_scoring_available')}",
        f"- Fallback generated-label scoring: {metrics.get('fallback_generated_label_scoring')}",
        f"- Parse fail count: {metrics.get('parse_fail_count')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in METRIC_KEYS:
        value = metrics.get(key)
        lines.append(f"| {key} | {'N/A' if value is None else f'{float(value):.4f}'} |")
    lines.extend(
        [
            "",
            "## Prompt Template",
            "",
            "```text",
            prompt_template,
            "```",
            "",
            "## Limitations",
            "",
            notes,
            "",
        ]
    )
    ensure_parent(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    config = load_config(args.config)

    test_path = choose(args.test_path, config, "test_path", "data/processed/msmarco_medium/test.jsonl")
    output_metrics = choose(args.output_metrics, config, "output_metrics", "outputs/jittorllm_zeroshot_medium/metrics.json")
    output_rankings = choose(args.output_rankings, config, "output_rankings", "outputs/jittorllm_zeroshot_medium/rankings.json")
    cache_path = choose(args.cache_path, config, "cache_path", "outputs/jittorllm_zeroshot_medium/score_cache.jsonl")
    run_summary = choose(args.run_summary, config, "run_summary", "outputs/jittorllm_zeroshot_medium/run_summary.md")
    max_queries = int(choose(args.max_queries, config, "max_queries", 100))
    max_length = int(choose(args.max_length, config, "max_length", 512))
    model_name = str(choose(args.model_name, config, "model_name", "jittorllm-unconfigured"))
    seed = int(choose(args.seed, config, "seed", 42))
    prompt_template = str(config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
    random.seed(seed)

    records = read_jsonl(test_path)[:max_queries]
    ensure_parent(output_metrics)
    ensure_parent(output_rankings)
    ensure_parent(cache_path).touch(exist_ok=True)

    try:
        backend = JittorLLMBackend(model_name=model_name, max_length=max_length)
    except BackendUnavailable as exc:
        elapsed = time.time() - started
        metrics = null_metrics("blocked", str(exc), model_name, records, elapsed)
        write_json(output_metrics, metrics)
        write_json(output_rankings, {"status": "blocked", "blocked_reason": str(exc), "rankings": []})
        write_summary(
            run_summary,
            metrics,
            prompt_template,
            test_path,
            cache_path,
            "JittorLLM inference is blocked because the local JittorLLM package/model environment is unavailable. No relevance scores were fabricated.",
        )
        print(f"blocked: {exc}")
        print(f"Wrote {resolve_path(output_metrics)}")
        return

    cache = load_cache(cache_path)
    scored_rows: list[dict] = []
    parse_fail_count = 0
    inferred_pairs = 0

    for record in records:
        for candidate in record.get("candidates", []):
            query_id = record["query_id"]
            doc_id = candidate["doc_id"]
            key = cache_key(query_id, doc_id)
            if key in cache:
                cached = cache[key]
                score_result = ScoreResult(
                    score=float(cached["score"]),
                    raw_output=str(cached.get("raw_output", "")),
                    score_source=str(cached.get("score_source", "cache")),
                    parse_fail=bool(cached.get("parse_fail", False)),
                )
            else:
                prompt = build_prompt(record["query"], candidate_text(candidate), max_length=max_length, template=prompt_template)
                try:
                    score_result = backend.score(prompt)
                except BackendUnavailable as exc:
                    elapsed = time.time() - started
                    metrics = null_metrics("blocked", str(exc), model_name, records, elapsed)
                    write_json(output_metrics, metrics)
                    write_json(output_rankings, {"status": "blocked", "blocked_reason": str(exc), "rankings": []})
                    write_summary(
                        run_summary,
                        metrics,
                        prompt_template,
                        test_path,
                        cache_path,
                        "A JittorLLM-like package was detected, but no verified adapter is available. No relevance scores were fabricated.",
                    )
                    print(f"blocked: {exc}")
                    return
                cache_row = {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "score": score_result.score,
                    "raw_output": score_result.raw_output,
                    "score_source": score_result.score_source,
                    "parse_fail": score_result.parse_fail,
                }
                append_cache(cache_path, cache_row)
                inferred_pairs += 1

            parse_fail_count += int(score_result.parse_fail)
            scored_rows.append(
                {
                    "query_id": record["query_id"],
                    "query": record["query"],
                    "doc_id": candidate["doc_id"],
                    "label": int(candidate["label"]),
                    "text": candidate.get("text", ""),
                    "score": float(score_result.score),
                    "score_source": score_result.score_source,
                    "parse_fail": score_result.parse_fail,
                    "raw_output": score_result.raw_output,
                }
            )

    elapsed = time.time() - started
    metrics = evaluate_grouped(scored_rows, topk=[1, 3, 5, 10])
    metrics.update(
        {
            "status": "ready",
            "model_name": model_name,
            "num_queries": len(records),
            "num_pairs": count_pairs(records),
            "candidate_pairs_scored": len(scored_rows),
            "new_inference_calls": inferred_pairs,
            "inference_time_sec": elapsed,
            "pairs_per_second": 0.0 if elapsed <= 0 else len(scored_rows) / elapsed,
            "parse_fail_count": parse_fail_count,
            "parse_fail_rate": parse_fail_count / max(len(scored_rows), 1),
            "logprob_scoring_available": False,
            "fallback_generated_label_scoring": True,
        }
    )
    write_json(output_metrics, metrics)
    write_json(output_rankings, {"status": "ready", "rankings": build_rankings(records, scored_rows)})
    write_summary(
        run_summary,
        metrics,
        prompt_template,
        test_path,
        cache_path,
        "Zero-shot generated-label scoring is coarser than logprob scoring and may be less stable than Cross-Encoder or LoRA rerankers.",
    )
    print(f"Wrote {resolve_path(output_metrics)}")
    print(f"Wrote {resolve_path(output_rankings)}")


if __name__ == "__main__":
    main()
