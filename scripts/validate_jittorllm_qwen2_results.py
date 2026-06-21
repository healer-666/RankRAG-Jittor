#!/usr/bin/env python3
"""Recompute Qwen2 Jittor reranking metrics directly from the score cache."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def recompute(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    totals = {f"recall@{k}": 0.0 for k in (1, 3, 5, 10)}
    totals.update({f"ndcg@{k}": 0.0 for k in (1, 3, 5, 10)})
    totals.update({"mrr": 0.0, "pairwise_accuracy": 0.0})
    for candidates in grouped.values():
        ranked = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)
        labels = [int(row["label"]) for row in ranked]
        positives = sum(labels)
        ideal = sorted(labels, reverse=True)
        for k in (1, 3, 5, 10):
            totals[f"recall@{k}"] += sum(labels[:k]) / positives if positives else 0.0
            dcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(labels[:k], 1))
            idcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(ideal[:k], 1))
            totals[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0
        totals["mrr"] += next((1.0 / rank for rank, label in enumerate(labels, 1) if label), 0.0)
        pos_scores = [float(row["score"]) for row in candidates if int(row["label"]) == 1]
        neg_scores = [float(row["score"]) for row in candidates if int(row["label"]) == 0]
        comparisons = [positive > negative for positive in pos_scores for negative in neg_scores]
        totals["pairwise_accuracy"] += mean([float(value) for value in comparisons])
    return {name: value / len(grouped) for name, value in totals.items()}


def main() -> None:
    output_dir = args().output_dir
    rows = [json.loads(line) for line in (output_dir / "score_cache.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    rankings = json.loads((output_dir / "rankings.json").read_text(encoding="utf-8"))
    query_ids = {str(item["query_id"]) for item in rankings}
    expected_keys = {
        (str(item["query_id"]), str(candidate_id))
        for item in rankings
        for candidate_id in item["candidate_ids"]
    }
    all_keys = [(str(row["query_id"]), str(row["candidate_id"])) for row in rows]
    selected = [row for row in rows if str(row["query_id"]) in query_ids]
    selected_keys = [(str(row["query_id"]), str(row["candidate_id"])) for row in selected]
    selected_counts = Counter(selected_keys)
    duplicate_pairs = sum(count - 1 for count in selected_counts.values() if count > 1)
    duplicate_pair_keys = [list(key) for key, count in selected_counts.items() if count > 1]
    missing_pair_keys = sorted(expected_keys - set(selected_keys))
    extra_pair_keys = sorted(set(selected_keys) - expected_keys)
    independent = recompute(selected)
    differences = {name: independent[name] - float(metrics[name]) for name in independent}
    max_metric_abs_diff = max(abs(value) for value in differences.values()) if differences else 0.0
    metrics_consistent = max_metric_abs_diff < 1e-12
    scores = [float(row["score"]) for row in selected]
    positive = [float(row["score"]) for row in selected if int(row["label"]) == 1]
    negative = [float(row["score"]) for row in selected if int(row["label"]) == 0]
    counts = Counter(scores)
    stats = {
        "min": min(scores), "max": max(scores), "mean": statistics.fmean(scores),
        "std": statistics.pstdev(scores), "positive_mean": statistics.fmean(positive),
        "negative_mean": statistics.fmean(negative),
        "tie_groups": sum(count > 1 for count in counts.values()),
        "tied_items": sum(count for count in counts.values() if count > 1),
        "tied_pairs": sum(count * (count - 1) // 2 for count in counts.values()),
    }
    by_key = {(str(row["query_id"]), str(row["candidate_id"])): row for row in selected}
    cases = {"correct": [], "incorrect": []}
    rankings_consistent = True
    ranking_mismatches = []
    for item in rankings:
        expected = sorted(
            [row for row in selected if str(row["query_id"]) == str(item["query_id"])],
            key=lambda row: float(row["score"]), reverse=True,
        )
        expected_ids = [str(row["candidate_id"]) for row in expected]
        actual_ids = [str(value) for value in item["sorted_candidate_ids"]]
        if expected_ids != actual_ids:
            rankings_consistent = False
            if len(ranking_mismatches) < 5:
                ranking_mismatches.append(
                    {
                        "query_id": item["query_id"],
                        "expected_top5": expected_ids[:5],
                        "actual_top5": actual_ids[:5],
                    }
                )
        top = item["ranking"][0]
        row = by_key[(str(item["query_id"]), str(top["candidate_id"]))]
        bucket = "correct" if int(row["label"]) == 1 else "incorrect"
        if len(cases[bucket]) < 5:
            cases[bucket].append({key: row[key] for key in ("query_id", "query", "candidate_id", "passage", "label", "score")})
    parse_fail = sum(bool(row.get("parse_fail")) for row in selected)
    expected_num_queries = int(metrics.get("num_queries", len(query_ids)))
    expected_num_pairs = int(metrics.get("num_pairs", len(expected_keys)))
    counts_consistent = len(query_ids) == expected_num_queries and len(selected) == expected_num_pairs
    pairs_complete = not duplicate_pairs and not missing_pair_keys and not extra_pair_keys
    passed = metrics_consistent and rankings_consistent and counts_consistent and pairs_complete and parse_fail == 0
    result = {
        "status": "passed" if passed else "failed",
        "num_queries": len(query_ids),
        "expected_num_queries": expected_num_queries,
        "num_pairs": len(selected),
        "expected_num_pairs": expected_num_pairs,
        "cache_rows": len(rows),
        "cache_unique_pairs": len(set(all_keys)),
        "duplicate_pairs": duplicate_pairs,
        "duplicate_pair_keys_sample": duplicate_pair_keys[:10],
        "missing_pairs": len(missing_pair_keys),
        "missing_pair_keys_sample": [list(key) for key in missing_pair_keys[:10]],
        "extra_pairs": len(extra_pair_keys),
        "extra_pair_keys_sample": [list(key) for key in extra_pair_keys[:10]],
        "parse_fail": parse_fail,
        "rankings_consistent": rankings_consistent,
        "ranking_consistent": rankings_consistent,
        "ranking_mismatches_sample": ranking_mismatches,
        "metrics_consistent": metrics_consistent,
        "max_metric_abs_diff": max_metric_abs_diff,
        "recomputed_metrics": independent,
        "independent_metrics": independent,
        "metric_differences": differences,
        "score_statistics": stats,
        "top1_cases": cases,
    }
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
