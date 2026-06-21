"""Evaluate Qwen2.5-0.5B generated-label reranking with JittorLLMs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jittorllm_reranker.backend_qwen2_jittor import Qwen2JittorBackend
from metrics import evaluate_grouped
from utils import ensure_parent, resolve_path, write_json


DEFAULT_PROMPT = """Given a search query and a candidate passage, determine whether the passage is relevant to the query.

Query:
{query}

Passage:
{passage}

Answer with exactly one label:
Relevant or Irrelevant."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id", row.get("doc_id"))
            rows[(str(row["query_id"]), str(candidate_id))] = row
    return rows


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def gpu_memory_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return sum(values) if values else 0
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def score_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows]
    positives = [float(row["score"]) for row in rows if int(row["label"]) == 1]
    negatives = [float(row["score"]) for row in rows if int(row["label"]) == 0]
    counts = Counter(scores)
    return {
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": statistics.fmean(scores),
        "score_std": statistics.pstdev(scores),
        "positive_score_mean": statistics.fmean(positives) if positives else math.nan,
        "negative_score_mean": statistics.fmean(negatives) if negatives else math.nan,
        "score_unique_values": len(counts),
        "score_tie_groups": sum(count > 1 for count in counts.values()),
        "score_tied_items": sum(count for count in counts.values() if count > 1),
        "score_tied_pairs": sum(count * (count - 1) // 2 for count in counts.values()),
    }


def parse_label(output: str) -> tuple[float, bool]:
    normalized = " ".join(str(output).strip().lower().split())
    if normalized.startswith("irrelevant") or normalized == "0" or normalized.startswith("0 "):
        return 0.0, False
    if normalized.startswith("relevant") or normalized == "1" or normalized.startswith("1 "):
        return 1.0, False
    if "irrelevant" in normalized:
        return 0.0, False
    if "relevant" in normalized:
        return 1.0, False
    return 0.5, True


def passage_text(candidate: dict[str, Any]) -> str:
    title = str(candidate.get("title", "")).strip()
    text = str(candidate.get("text", "")).strip()
    return f"{title}. {text}" if title else text


def build_rankings(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    output = []
    for record in records:
        candidates = grouped[str(record["query_id"])]
        ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        output.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "candidate_ids": [row["candidate_id"] for row in candidates],
                "labels": [row["label"] for row in candidates],
                "scores": [row["score"] for row in candidates],
                "raw_outputs": [row["raw_output"] for row in candidates],
                "sorted_candidate_ids": [row["candidate_id"] for row in ranked],
                "ranking": [
                    {
                        "rank": rank,
                        "candidate_id": row["candidate_id"],
                        "label": row["label"],
                        "score": row["score"],
                        "raw_output": row["raw_output"],
                        "parse_fail": row["parse_fail"],
                    }
                    for rank, row in enumerate(ranked, 1)
                ],
            }
        )
    return output


def top1_cases(rankings: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_key = {(str(row["query_id"]), str(row["candidate_id"])): row for row in rows}
    cases: dict[str, list[dict[str, Any]]] = {"correct": [], "incorrect": []}
    for ranking in rankings:
        top = ranking["ranking"][0]
        row = by_key[(str(ranking["query_id"]), str(top["candidate_id"]))]
        key = "correct" if int(top["label"]) == 1 else "incorrect"
        if len(cases[key]) < 5:
            cases[key].append(
                {
                    "query_id": ranking["query_id"],
                    "query": ranking["query"],
                    "candidate_id": top["candidate_id"],
                    "label": int(top["label"]),
                    "score": float(top["score"]),
                    "passage": row["passage"],
                }
            )
    return cases


def write_summary(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    prompt: str,
    cases: dict[str, list[dict[str, Any]]],
) -> None:
    metric_names = [
        "recall@1", "recall@3", "recall@5", "recall@10",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
        "mrr", "pairwise_accuracy",
    ]
    lines = [
        "# Qwen2.5-0.5B JittorLLM Reranking Run",
        "",
        f"- Status: {metrics['status']}",
        f"- Backend: {metrics['backend']}",
        f"- Model: {metrics['model_name']}",
        f"- Dataset: `{config['test_path']}`",
        f"- Num queries: {metrics['num_queries']}",
        f"- Num candidate pairs: {metrics['num_pairs']}",
        f"- Cached pairs reused: {metrics['cached_pairs_used']}",
        f"- New inference calls: {metrics['new_inference_calls']}",
        f"- Scoring method: {metrics['scoring_method']}",
        f"- Parse failures: {metrics['parse_fail']} ({metrics['parse_fail_rate']:.2%})",
        f"- Model inference time: {metrics['inference_time_sec']:.3f}s",
        f"- Pairs per second: {metrics['pairs_per_second']:.4f}",
        f"- Peak observed GPU memory: {metrics['gpu_memory_peak_observed_mb']} MiB",
        "- Label-token logit scoring: available",
        "- Generated-label fallback: not used",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {metrics[name]:.4f} |" for name in metric_names)
    lines.extend(
        [
            "",
            "## Score Statistics",
            "",
            "| Statistic | Value |",
            "| --- | ---: |",
            f"| min | {metrics['score_min']:.6f} |",
            f"| max | {metrics['score_max']:.6f} |",
            f"| mean | {metrics['score_mean']:.6f} |",
            f"| std | {metrics['score_std']:.6f} |",
            f"| positive mean | {metrics['positive_score_mean']:.6f} |",
            f"| negative mean | {metrics['negative_score_mean']:.6f} |",
            f"| tie groups | {metrics['score_tie_groups']} |",
            f"| tied items | {metrics['score_tied_items']} |",
            f"| tied pairs | {metrics['score_tied_pairs']} |",
            "",
            "## Prompt",
            "",
            "```text",
            prompt,
            "```",
            "",
            "## Limitations",
            "",
            "This is label-token logit-margin zero-shot scoring on a small subset. It is a Jittor LLM inference proof of concept, not a replacement for a trained cross-encoder.",
            "",
        ]
    )
    for title, key in (("Correct Top-1 Cases", "correct"), ("Incorrect Top-1 Cases", "incorrect")):
        lines.extend([f"## {title}", ""])
        for case in cases[key]:
            passage = " ".join(case["passage"].split())[:240]
            lines.append(
                f"- `{case['query_id']}` / `{case['candidate_id']}` score={case['score']:.6f}: "
                f"{case['query']} | {passage}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("backend") != "qwen2_jittorllms":
        raise ValueError("This evaluator requires backend: qwen2_jittorllms")

    test_path = resolve_path(config["test_path"])
    cache_path = resolve_path(config["cache_path"])
    metrics_path = resolve_path(config["metrics_path"])
    rankings_path = resolve_path(config["rankings_path"])
    summary_path = resolve_path(Path(config["output_dir"]) / "run_summary.md")
    records = read_jsonl(test_path)[: int(config["max_queries"])]
    cache = load_cache(cache_path)
    initial_cache_size = len(cache)
    prompt_template = str(config.get("prompt_template", DEFAULT_PROMPT))

    backend = Qwen2JittorBackend(
        model_path=resolve_path(config["model_path"]),
        jittorllms_root=resolve_path(config["jittorllms_root"]),
        max_input_length=int(config["max_input_length"]),
        max_new_tokens=int(config["max_new_tokens"]),
    )
    gpu_samples = [value for value in [gpu_memory_mb()] if value is not None]
    rows: list[dict[str, Any]] = []
    new_calls = 0
    inference_time = 0.0
    started = time.perf_counter()

    for query_index, record in enumerate(records, 1):
        for candidate_index, candidate in enumerate(record["candidates"], 1):
            candidate_id = str(candidate["doc_id"])
            key = (str(record["query_id"]), candidate_id)
            if key in cache:
                row = cache[key]
            else:
                passage = passage_text(candidate)
                prompt = prompt_template.format(query=record["query"], passage=passage)
                scored = backend.score_binary_tokens(prompt)
                row = {
                    "query_id": record["query_id"],
                    "candidate_id": candidate_id,
                    "query": record["query"],
                    "passage": passage,
                    "label": int(candidate["label"]),
                    "score": scored.score,
                    "raw_output": scored.raw_output,
                    "generated_token_ids": [scored.selected_token_id],
                    "parse_fail": False,
                    "runtime_sec": scored.runtime_sec,
                    "negative_logit": scored.negative_logit,
                    "positive_logit": scored.positive_logit,
                    "score_source": "logit(1)-logit(0)",
                    "backend": backend.backend_name,
                    "model_name": backend.model_name,
                }
                append_cache(cache_path, row)
                cache[key] = row
                new_calls += 1
                inference_time += scored.runtime_sec
                print(
                    f"[{query_index}/{len(records)} {candidate_index}/{len(record['candidates'])}] "
                    f"{candidate_id}: {scored.raw_output!r} score={scored.score:.4f} "
                    f"{scored.runtime_sec:.3f}s",
                    flush=True,
                )
            rows.append(row)
        sampled_memory = gpu_memory_mb()
        if sampled_memory is not None:
            gpu_samples.append(sampled_memory)

    backend.close()
    elapsed = time.perf_counter() - started
    values = evaluate_grouped(rows, topk=[1, 3, 5, 10])
    stats = score_statistics(rows)
    parse_fail_count = sum(bool(row["parse_fail"]) for row in rows)
    metrics: dict[str, Any] = {
        "status": "ready",
        "backend": "qwen2_jittorllms",
        "model_name": str(config.get("model_name", backend.model_name)),
        "scoring_method": "label_token_logit_margin",
        "num_queries": len(records),
        "num_pairs": len(rows),
        "new_inference_calls": new_calls,
        "cached_pairs_used": len(rows) - new_calls,
        "initial_cache_entries": initial_cache_size,
        "parse_fail": parse_fail_count,
        "parse_fail_rate": parse_fail_count / max(len(rows), 1),
        "inference_time_sec": inference_time,
        "wall_time_sec": elapsed,
        "pairs_per_second": new_calls / inference_time if inference_time else 0.0,
        "all_pair_runtime_sec": sum(float(row.get("runtime_sec", 0.0)) for row in rows),
        "gpu_memory_peak_observed_mb": max(gpu_samples) if gpu_samples else None,
        **values,
        **stats,
    }
    rankings = build_rankings(records, rows)
    cases = top1_cases(rankings, rows)
    write_json(metrics_path, metrics)
    write_json(rankings_path, rankings)
    write_summary(summary_path, config, metrics, prompt_template, cases)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
