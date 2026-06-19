"""Build pair-level instruction data for LoRA reranker debugging."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

try:
    from src.lora_reranker.lora_utils import truncate_text, write_jsonl
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.lora_reranker.lora_utils import truncate_text, write_jsonl


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_query(row: dict) -> str:
    for key in ["query", "question"]:
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Cannot find query/question field in row keys: {sorted(row.keys())}")


def get_query_id(row: dict, index: int) -> str:
    for key in ["query_id", "qid", "id"]:
        value = row.get(key)
        if value:
            return str(value)
    return f"query_{index:06d}"


def get_candidates(row: dict) -> list[dict]:
    for key in ["candidates", "passages", "contexts"]:
        value = row.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(f"Cannot find candidate list in row keys: {sorted(row.keys())}")


def candidate_text(candidate: dict) -> str:
    for key in ["text", "passage", "content", "body"]:
        value = candidate.get(key)
        if value:
            return str(value)
    title = candidate.get("title")
    if title:
        return str(title)
    raise ValueError(f"Cannot find candidate text in keys: {sorted(candidate.keys())}")


def candidate_id(candidate: dict, index: int) -> str:
    for key in ["doc_id", "passage_id", "id", "pid"]:
        value = candidate.get(key)
        if value:
            return str(value)
    return f"passage_{index:03d}"


def candidate_label(candidate: dict) -> int:
    for key in ["label", "is_relevant", "relevance"]:
        if key not in candidate:
            continue
        value = candidate[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return 1 if value > 0 else 0
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "relevant", "positive"}:
            return 1
        if text in {"0", "false", "no", "irrelevant", "negative"}:
            return 0
    raise ValueError(f"Cannot find explicit relevance label in candidate keys: {sorted(candidate.keys())}")


def build_pair_rows(
    rows: list[dict],
    *,
    split: str,
    max_pairs: int,
    negatives_per_query: int,
    rng: random.Random,
) -> tuple[list[dict], Counter]:
    pairs: list[dict] = []
    stats: Counter = Counter()
    for row_index, row in enumerate(rows):
        query = get_query(row)
        query_id = get_query_id(row, row_index)
        candidates = get_candidates(row)
        positives = []
        negatives = []
        for cand_index, cand in enumerate(candidates):
            label = candidate_label(cand)
            item = {
                "query_id": query_id,
                "query": query,
                "passage_id": candidate_id(cand, cand_index),
                "passage": truncate_text(candidate_text(cand)),
                "source_split": split,
            }
            if label == 1:
                positives.append(item)
            else:
                negatives.append(item)
        if not positives or not negatives:
            stats["skipped_queries_without_pos_or_neg"] += 1
            continue
        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected = [(positives[0], "Relevant")]
        selected.extend((neg, "Irrelevant") for neg in negatives[:negatives_per_query])
        for item, label_name in selected:
            pair = dict(item)
            pair["label"] = label_name
            pairs.append(pair)
            stats[label_name.lower()] += 1
        stats["queries"] += 1
        if max_pairs and len(pairs) >= max_pairs:
            pairs = pairs[:max_pairs]
            break
    stats["pairs"] = len(pairs)
    return pairs, stats


def build_test_queries(rows: list[dict], *, max_queries: int) -> tuple[list[dict], Counter]:
    output: list[dict] = []
    stats: Counter = Counter()
    for row_index, row in enumerate(rows):
        query = get_query(row)
        query_id = get_query_id(row, row_index)
        candidates_out = []
        pos_count = 0
        neg_count = 0
        for cand_index, cand in enumerate(get_candidates(row)):
            label = candidate_label(cand)
            pos_count += int(label == 1)
            neg_count += int(label == 0)
            candidates_out.append(
                {
                    "passage_id": candidate_id(cand, cand_index),
                    "passage": truncate_text(candidate_text(cand)),
                    "label": label,
                }
            )
        if pos_count == 0 or neg_count == 0:
            stats["skipped_queries_without_pos_or_neg"] += 1
            continue
        output.append({"query_id": query_id, "query": query, "candidates": candidates_out})
        stats["queries"] += 1
        stats["positive"] += pos_count
        stats["negative"] += neg_count
        if max_queries and len(output) >= max_queries:
            break
    stats["candidates"] = stats["positive"] + stats["negative"]
    return output, stats


def write_data_card(path: Path, summary: dict) -> None:
    lines = [
        "# LoRA reranker debug data",
        "",
        "| Split | Queries | Pairs/Candidates | Positive | Negative |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "valid", "test"]:
        stats = summary[split]
        total_key = "pairs" if split != "test" else "candidates"
        lines.append(
            f"| {split} | {stats.get('queries', 0)} | {stats.get(total_key, 0)} | "
            f"{stats.get('relevant', stats.get('positive', 0))} | {stats.get('irrelevant', stats.get('negative', 0))} |"
        )
    lines.extend(
        [
            "",
            "Labels are copied from the processed MS MARCO data. No synthetic relevance labels are created.",
            "",
            "Generation arguments:",
            "",
        ]
    )
    for key, value in summary["args"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_in", default="data/processed/msmarco_medium/train.jsonl")
    parser.add_argument("--valid_in", default="data/processed/msmarco_medium/valid.jsonl")
    parser.add_argument("--test_in", default="data/processed/msmarco_medium/test.jsonl")
    parser.add_argument("--out_dir", default="data/processed/lora_debug")
    parser.add_argument("--max_train_pairs", type=int, default=500)
    parser.add_argument("--max_valid_pairs", type=int, default=200)
    parser.add_argument("--max_test_queries", type=int, default=50)
    parser.add_argument("--negatives_per_query", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, train_stats = build_pair_rows(
        load_rows(Path(args.train_in)),
        split="train",
        max_pairs=args.max_train_pairs,
        negatives_per_query=args.negatives_per_query,
        rng=rng,
    )
    valid_pairs, valid_stats = build_pair_rows(
        load_rows(Path(args.valid_in)),
        split="valid",
        max_pairs=args.max_valid_pairs,
        negatives_per_query=args.negatives_per_query,
        rng=rng,
    )
    test_queries, test_stats = build_test_queries(load_rows(Path(args.test_in)), max_queries=args.max_test_queries)

    if not train_pairs or not valid_pairs or not test_queries:
        raise SystemExit("Failed to build LoRA debug data with explicit positive/negative labels.")

    write_jsonl(train_pairs, out_dir / "train_pairs.jsonl")
    write_jsonl(valid_pairs, out_dir / "valid_pairs.jsonl")
    write_jsonl(test_queries, out_dir / "test_queries.jsonl")
    write_data_card(
        out_dir / "data_card.md",
        {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
            "args": vars(args),
        },
    )
    print(f"Wrote LoRA debug data to {out_dir}")
    print(f"train pairs={len(train_pairs)} valid pairs={len(valid_pairs)} test queries={len(test_queries)}")


if __name__ == "__main__":
    main()
