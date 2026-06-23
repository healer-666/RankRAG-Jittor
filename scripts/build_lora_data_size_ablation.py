"""Build nested LoRA reranker training subsets for data-size ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.lora_reranker.build_lora_data import build_pair_rows, load_rows
    from src.lora_reranker.lora_utils import write_jsonl
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.lora_reranker.build_lora_data import build_pair_rows, load_rows
    from src.lora_reranker.lora_utils import write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict[str, Any]) -> str:
    stable = {
        "query_id": row.get("query_id"),
        "query": row.get("query"),
        "passage_id": row.get("passage_id"),
        "passage": row.get("passage"),
        "label": row.get("label"),
        "source_split": row.get("source_split"),
    }
    return json.dumps(stable, ensure_ascii=False, sort_keys=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_training_rows(rows: list[dict[str, Any]], *, source_name: str) -> None:
    errors: list[str] = []
    seen_keys: set[str] = set()
    valid_labels = {"Relevant", "Irrelevant"}
    for index, row in enumerate(rows, start=1):
        key = row_key(row)
        if key in seen_keys:
            errors.append(f"{source_name}: duplicate stable row key at row {index}")
        seen_keys.add(key)
        if not str(row.get("query", "")).strip():
            errors.append(f"{source_name}: empty query at row {index}")
        if not str(row.get("passage", "")).strip():
            errors.append(f"{source_name}: empty passage at row {index}")
        if row.get("label") not in valid_labels:
            errors.append(f"{source_name}: invalid label at row {index}: {row.get('label')!r}")
    if errors:
        preview = "\n".join(errors[:10])
        suffix = f"\n... {len(errors) - 10} more errors" if len(errors) > 10 else ""
        raise RuntimeError(f"Training row validation failed:\n{preview}{suffix}")


def stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_counts: dict[str, int] = defaultdict(int)
    label_counts: Counter[str] = Counter()
    for row in rows:
        query_counts[str(row.get("query_id", row.get("query", "")))] += 1
        label_counts[str(row.get("label", ""))] += 1
    pair_counts = list(query_counts.values())
    return {
        "pairs": len(rows),
        "unique_queries": len(query_counts),
        "avg_pairs_per_query": len(rows) / max(len(query_counts), 1),
        "min_pairs_per_query": min(pair_counts) if pair_counts else 0,
        "max_pairs_per_query": max(pair_counts) if pair_counts else 0,
        "label_distribution": dict(sorted(label_counts.items())),
    }


def build_source_if_needed(args: argparse.Namespace) -> tuple[Path, bool]:
    source_path = Path(args.source_train_path)
    if source_path.exists():
        return source_path, False

    source_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows, stats = build_pair_rows(
        load_rows(Path(args.train_in)),
        split="train",
        max_pairs=args.source_pairs,
        negatives_per_query=args.negatives_per_query,
        rng=rng,
    )
    if len(rows) != args.source_pairs:
        raise RuntimeError(f"Expected {args.source_pairs} source rows, got {len(rows)}")
    write_jsonl(rows, source_path)
    (source_path.parent / "source_rebuild_summary.json").write_text(
        json.dumps(
            {
                "source_train_path": source_path.as_posix(),
                "train_in": args.train_in,
                "seed": args.seed,
                "source_pairs": args.source_pairs,
                "negatives_per_query": args.negatives_per_query,
                "stats": dict(stats),
                "note": "Rebuilt from the committed data-generation script parameters used by the 10k v3 data card.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_path, True


def build_subsets(args: argparse.Namespace) -> dict[str, Any]:
    source_path, source_rebuilt = build_source_if_needed(args)
    source_rows = load_jsonl(source_path)
    if len(source_rows) != args.source_pairs:
        raise RuntimeError(f"Expected {args.source_pairs} source rows, got {len(source_rows)}")
    validate_training_rows(source_rows, source_name=str(source_path))
    source_keys = [row_key(row) for row in source_rows]
    if len(set(source_keys)) != len(source_keys):
        raise RuntimeError("Source training rows contain duplicate stable row keys.")

    rng = random.Random(args.seed)
    order = list(range(len(source_rows)))
    rng.shuffle(order)
    permuted_rows = [source_rows[index] for index in order]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_specs = {"1k": 1000, "3k": 3000, "10k": args.source_pairs}
    subsets: dict[str, Any] = {}
    for name, count in subset_specs.items():
        rows = permuted_rows[:count]
        path = out_dir / f"train_{name}.jsonl"
        if name != "10k":
            write_jsonl(rows, path)
        else:
            path = source_path
        row_stats = stats_for_rows(rows)
        subsets[name] = {
            "path": path.as_posix(),
            "pairs": count,
            "sha256": sha256_file(path) if path.exists() else None,
            **{key: value for key, value in row_stats.items() if key != "pairs"},
        }

    keys_1k = [row_key(row) for row in permuted_rows[:1000]]
    keys_3k = [row_key(row) for row in permuted_rows[:3000]]
    keys_10k = [row_key(row) for row in source_rows]
    manifest = {
        "source_path": source_path.as_posix(),
        "source_sha256": sha256_file(source_path),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "selection_strategy": "nested subsets selected from one deterministic permutation",
        "selection_note": (
            "1k and 3k are prefixes of the deterministic permutation. "
            "The committed 10k source file keeps its original row order, so 3k is a subset of 10k but not a prefix of "
            "that file."
        ),
        "source_rebuild": {
            "used_existing_source": not source_rebuilt,
            "rebuilt_source": source_rebuilt,
            "train_in": args.train_in,
            "source_pairs": args.source_pairs,
            "negatives_per_query": args.negatives_per_query,
        },
        "subsets": subsets,
        "checks": {
            "subset_1k_in_3k": keys_1k == keys_3k[:1000],
            "subset_3k_in_10k": set(keys_3k).issubset(set(keys_10k)),
            "subset_3k_is_prefix_of_permutation": keys_3k == [row_key(row) for row in permuted_rows[:3000]],
            "source_has_no_duplicate_keys": len(set(keys_10k)) == len(keys_10k),
            "validation_or_test_rows_used": False,
        },
    }
    manifest_path = out_dir / "data_size_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default="data/processed/msmarco_medium/train.jsonl")
    parser.add_argument("--source-train-path", default="data/processed/lora_qwen_1_5b_10k/train_pairs.jsonl")
    parser.add_argument("--out-dir", default="data/processed/lora_ablation")
    parser.add_argument("--source-pairs", type=int, default=10000)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild-source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force_rebuild_source:
        source = Path(args.source_train_path)
        if source.exists():
            raise RuntimeError(
                f"Refusing to delete existing source training file: {source}. "
                "Use a separate --source-train-path for a rebuild so formal 10k inputs are not overwritten."
            )
    build_subsets(args)


if __name__ == "__main__":
    main()
