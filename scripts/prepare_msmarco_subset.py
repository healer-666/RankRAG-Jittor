#!/usr/bin/env python3
"""Prepare a small MS MARCO query-passage ranking subset.

The script keeps the project JSONL schema used by the synthetic benchmark:
one query per line, with candidate passages and binary labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "msmarco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_queries", type=int, default=1000)
    parser.add_argument("--max_valid_queries", type=int, default=200)
    parser.add_argument("--max_test_queries", type=int, default=200)
    parser.add_argument("--candidates_per_query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(value: object) -> str:
    return str(value or "").replace("\n", " ").strip()


def build_record(
    *,
    query_id: str,
    query: str,
    positives: list[str],
    negatives: list[str],
    candidates_per_query: int,
    rng: random.Random,
) -> dict | None:
    positives = [text for text in dict.fromkeys(clean_text(text) for text in positives) if text]
    negatives = [text for text in dict.fromkeys(clean_text(text) for text in negatives) if text]
    if not positives or not negatives:
        return None

    rng.shuffle(positives)
    rng.shuffle(negatives)
    kept_pos = positives[:1]
    kept_neg = negatives[: max(candidates_per_query - len(kept_pos), 1)]
    candidates = []
    for idx, text in enumerate(kept_pos, start=1):
        candidates.append({"doc_id": f"{query_id}_pos_{idx}", "title": "", "text": text, "label": 1})
    for idx, text in enumerate(kept_neg, start=1):
        candidates.append({"doc_id": f"{query_id}_neg_{idx}", "title": "", "text": text, "label": 0})

    if len(candidates) < 2:
        return None
    rng.shuffle(candidates)
    return {"query_id": query_id, "query": clean_text(query), "candidates": candidates[:candidates_per_query]}


def iter_microsoft_records(dataset: Iterable[dict], split_name: str, limit: int, candidates_per_query: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    seen = 0
    skipped = 0

    for row in dataset:
        seen += 1
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") if isinstance(passages, dict) else None
        labels = passages.get("is_selected") if isinstance(passages, dict) else None
        if not texts or labels is None:
            skipped += 1
            continue

        positives = [text for text, label in zip(texts, labels) if int(label) == 1]
        negatives = [text for text, label in zip(texts, labels) if int(label) == 0]
        record = build_record(
            query_id=f"msmarco_{split_name}_{len(records) + 1:06d}",
            query=row.get("query", ""),
            positives=positives,
            negatives=negatives,
            candidates_per_query=candidates_per_query,
            rng=rng,
        )
        if record is None:
            skipped += 1
            continue
        records.append(record)
        if len(records) >= limit:
            break

    print(f"microsoft/ms_marco {split_name}: kept={len(records)} seen={seen} skipped={skipped}")
    return records


def prepare_from_microsoft(args: argparse.Namespace) -> tuple[str, dict[str, list[dict]]]:
    from datasets import load_dataset

    print("Trying data source: microsoft/ms_marco config=v1.1")
    train_stream = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    valid_stream = load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)

    train = iter_microsoft_records(
        train_stream,
        "train",
        args.max_train_queries,
        args.candidates_per_query,
        args.seed,
    )
    valid_and_test = iter_microsoft_records(
        valid_stream,
        "validation",
        args.max_valid_queries + args.max_test_queries,
        args.candidates_per_query,
        args.seed + 1,
    )

    valid = []
    test = []
    for idx, record in enumerate(valid_and_test):
        target = valid if idx < args.max_valid_queries else test
        split = "valid" if idx < args.max_valid_queries else "test"
        record = dict(record)
        record["query_id"] = f"msmarco_{split}_{len(target) + 1:06d}"
        for cand_idx, candidate in enumerate(record["candidates"], start=1):
            label_name = "pos" if int(candidate["label"]) == 1 else "neg"
            candidate["doc_id"] = f"{record['query_id']}_{label_name}_{cand_idx}"
        target.append(record)

    if not train or not valid or not test:
        raise RuntimeError(
            "microsoft/ms_marco did not yield enough labeled queries. "
            f"train={len(train)} valid={len(valid)} test={len(test)}"
        )
    return "microsoft/ms_marco:v1.1", {"train": train, "valid": valid, "test": test}


def collect_sentence_transformers_records(dataset: Iterable[dict], limit: int, candidates_per_query: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for row in dataset:
        keys = set(row.keys())
        query = row.get("query") or row.get("sentence1") or row.get("anchor")
        positives: list[str] = []
        negatives: list[str] = []

        for key in ["positive", "pos", "positive_passage", "positive_passages"]:
            if key in keys:
                value = row[key]
                positives.extend(value if isinstance(value, list) else [value])
        for key in ["negative", "neg", "negative_passage", "negative_passages", "hard_negative"]:
            if key in keys:
                value = row[key]
                negatives.extend(value if isinstance(value, list) else [value])
        if "passage" in keys and "label" in keys:
            label = int(row["label"])
            (positives if label == 1 else negatives).append(row["passage"])

        record = build_record(
            query_id=f"msmarco_st_{len(records) + 1:06d}",
            query=query or "",
            positives=positives,
            negatives=negatives,
            candidates_per_query=candidates_per_query,
            rng=rng,
        )
        if record:
            records.append(record)
        if len(records) >= limit:
            break
    return records


def prepare_from_sentence_transformers(args: argparse.Namespace) -> tuple[str, dict[str, list[dict]]]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    print("Trying fallback data source: sentence-transformers/msmarco")
    configs = get_dataset_config_names("sentence-transformers/msmarco")
    print("Available configs:", configs)
    errors = []
    total_needed = args.max_train_queries + args.max_valid_queries + args.max_test_queries

    for config in configs:
        try:
            splits = get_dataset_split_names("sentence-transformers/msmarco", config)
            print(f"Config {config} splits: {splits}")
            split = "train" if "train" in splits else splits[0]
            stream = load_dataset("sentence-transformers/msmarco", config, split=split, streaming=True)
            iterator = iter(stream)
            first = next(iterator)
            print(f"Config {config} first-row columns: {sorted(first.keys())}")

            def chained():
                yield first
                yield from iterator

            records = collect_sentence_transformers_records(chained(), total_needed, args.candidates_per_query, args.seed)
            if len(records) < total_needed:
                raise RuntimeError(f"Only collected {len(records)} usable records from {config}")
            train_end = args.max_train_queries
            valid_end = train_end + args.max_valid_queries
            return (
                f"sentence-transformers/msmarco:{config}",
                {
                    "train": relabel_records(records[:train_end], "train"),
                    "valid": relabel_records(records[train_end:valid_end], "valid"),
                    "test": relabel_records(records[valid_end:total_needed], "test"),
                },
            )
        except Exception as exc:  # pragma: no cover - depends on remote dataset shape.
            errors.append(f"{config}: {exc}")
            print(f"Config {config} failed: {exc}", file=sys.stderr)

    raise RuntimeError("sentence-transformers/msmarco fallback failed:\n" + "\n".join(errors))


def relabel_records(records: list[dict], split: str) -> list[dict]:
    relabeled = []
    for idx, record in enumerate(records, start=1):
        record = dict(record)
        record["query_id"] = f"msmarco_{split}_{idx:06d}"
        for cand_idx, candidate in enumerate(record["candidates"], start=1):
            label_name = "pos" if int(candidate["label"]) == 1 else "neg"
            candidate["doc_id"] = f"{record['query_id']}_{label_name}_{cand_idx}"
        relabeled.append(record)
    return relabeled


def summarize(records: list[dict]) -> dict[str, float | int]:
    candidate_counts = [len(record["candidates"]) for record in records]
    labels = Counter(int(candidate["label"]) for record in records for candidate in record["candidates"])
    return {
        "queries": len(records),
        "avg_candidates": sum(candidate_counts) / max(len(candidate_counts), 1),
        "positives": labels[1],
        "negatives": labels[0],
    }


def write_dataset_card(source: str, splits: dict[str, list[dict]], candidates_per_query: int) -> None:
    lines = [
        "# MS MARCO Small Subset",
        "",
        f"Data source: `{source}`.",
        "",
        "This is a small MS MARCO query-passage ranking subset prepared for the lightweight RankRAG-style context ranking reproduction.",
        "It is not the full MS MARCO dataset and is not a leaderboard submission setup.",
        "",
        "Construction:",
        "- Keep queries with at least one positive passage and one negative passage.",
        "- Positive passages come from `is_selected == 1` or official positive fields when available.",
        "- Negative passages come from same-query unselected passages or official negative fields.",
        f"- Keep at most `{candidates_per_query}` candidates per query.",
        "- Use deterministic streaming/sample order with seed-controlled candidate shuffling.",
        "",
        "| Split | Queries | Avg candidates | Positives | Negatives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, records in splits.items():
        stats = summarize(records)
        lines.append(
            f"| {split} | {stats['queries']} | {stats['avg_candidates']:.2f} | {stats['positives']} | {stats['negatives']} |"
        )
    lines.append("")
    (OUTPUT_DIR / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    try:
        source, splits = prepare_from_microsoft(args)
    except Exception as exc:
        errors.append(f"microsoft/ms_marco failed: {exc}")
        print(errors[-1], file=sys.stderr)
        try:
            source, splits = prepare_from_sentence_transformers(args)
        except Exception as fallback_exc:
            errors.append(str(fallback_exc))
            print("\n".join(errors), file=sys.stderr)
            raise SystemExit(1) from None

    for split, records in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", records)
    write_dataset_card(source, splits, args.candidates_per_query)

    print(f"Wrote MS MARCO subset to {OUTPUT_DIR}")
    for split, records in splits.items():
        stats = summarize(records)
        print(
            f"{split}: queries={stats['queries']} avg_candidates={stats['avg_candidates']:.2f} "
            f"positives={stats['positives']} negatives={stats['negatives']}"
        )


if __name__ == "__main__":
    main()
