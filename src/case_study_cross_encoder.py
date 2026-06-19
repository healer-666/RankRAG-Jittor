"""Case study comparing BM25, Jittor rerankers, and Cross-Encoder outputs."""

from __future__ import annotations

import json
from pathlib import Path

from dataset import read_jsonl
from utils import ensure_parent, resolve_path, write_json


DATA_PATH = "data/processed/msmarco_medium/test.jsonl"
OUTPUT_JSON = "outputs/msmarco_medium_cross_encoder_case_study.json"
OUTPUT_MD = "docs/msmarco_medium_cross_encoder_case_study.md"

METHOD_FILES = {
    "BM25": ("outputs/msmarco_medium_retrieval_baseline_rankings.json", "bm25"),
    "MLP Jittor": ("outputs/msmarco_medium_jittor_test_rankings.json", None),
    "TextCNN Jittor": ("outputs/msmarco_medium_textcnn_jittor_test_rankings.json", None),
    "Cross-Encoder": ("outputs/msmarco_medium_cross_encoder_rankings.json", None),
}


def load_rankings() -> dict[str, dict[str, dict]]:
    loaded: dict[str, dict[str, dict]] = {}
    for method, (path, key) in METHOD_FILES.items():
        resolved = resolve_path(path)
        if not resolved.exists():
            print(f"case study: missing {method}: {path}")
            continue
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        rankings = payload.get(key, []) if key else payload
        loaded[method] = {item["query_id"]: item for item in rankings}
    return loaded


def positive_rank(ranking: dict | None) -> int | None:
    if not ranking:
        return None
    for item in ranking.get("ranking", []):
        if int(item.get("label", 0)) == 1:
            return int(item["rank"])
    return None


def compact_ranking(ranking: dict | None) -> list[dict]:
    if not ranking:
        return []
    rows = []
    for item in ranking.get("ranking", [])[:5]:
        rows.append(
            {
                "rank": item["rank"],
                "doc_id": item["doc_id"],
                "label": int(item["label"]),
                "score": float(item["score"]),
                "text_preview": item.get("text", "")[:220],
            }
        )
    return rows


def summarize_case(record: dict, rankings: dict[str, dict[str, dict]]) -> dict:
    methods = {}
    for method, by_query in rankings.items():
        ranking = by_query.get(record["query_id"])
        methods[method] = {
            "positive_rank": positive_rank(ranking),
            "top5": compact_ranking(ranking),
        }
    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "methods": methods,
    }


def pick_cases(records: list[dict], rankings: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    buckets = {
        "cross_encoder_success_bm25_failure": [],
        "bm25_success_cross_encoder_failure": [],
        "all_methods_failure": [],
        "lightweight_gap": [],
    }
    for record in records:
        qid = record["query_id"]
        bm25 = positive_rank(rankings.get("BM25", {}).get(qid))
        ce = positive_rank(rankings.get("Cross-Encoder", {}).get(qid))
        mlp = positive_rank(rankings.get("MLP Jittor", {}).get(qid))
        textcnn = positive_rank(rankings.get("TextCNN Jittor", {}).get(qid))

        if ce == 1 and (bm25 is None or bm25 > 3) and len(buckets["cross_encoder_success_bm25_failure"]) < 3:
            buckets["cross_encoder_success_bm25_failure"].append(summarize_case(record, rankings))
        if bm25 == 1 and (ce is None or ce > 3) and len(buckets["bm25_success_cross_encoder_failure"]) < 3:
            buckets["bm25_success_cross_encoder_failure"].append(summarize_case(record, rankings))
        ranks = [rank for rank in [bm25, ce, mlp, textcnn] if rank is not None]
        if ranks and all(rank > 5 for rank in ranks) and len(buckets["all_methods_failure"]) < 3:
            buckets["all_methods_failure"].append(summarize_case(record, rankings))
        light_ranks = [rank for rank in [mlp, textcnn] if rank is not None]
        if ce == 1 and light_ranks and min(light_ranks) > 3 and len(buckets["lightweight_gap"]) < 3:
            buckets["lightweight_gap"].append(summarize_case(record, rankings))
    return buckets


def write_markdown(cases: dict[str, list[dict]], path: str) -> None:
    lines = [
        "# MS MARCO Medium Cross-Encoder Case Study",
        "",
        "This qualitative analysis compares BM25, lightweight Jittor rerankers, and an external pretrained Cross-Encoder. The Cross-Encoder is not the Jittor reproduction body.",
        "",
        "The pretrained cross-encoder may better capture semantic relevance, while BM25 is strong when lexical overlap is reliable. Lightweight Jittor models still lack pretrained semantic knowledge.",
        "",
    ]
    titles = {
        "cross_encoder_success_bm25_failure": "Cross-Encoder succeeds while BM25 fails",
        "bm25_success_cross_encoder_failure": "BM25 succeeds while Cross-Encoder fails",
        "all_methods_failure": "All methods fail",
        "lightweight_gap": "Jittor lightweight models lag Cross-Encoder",
    }
    for bucket, title in titles.items():
        lines.extend([f"## {title}", ""])
        if not cases[bucket]:
            lines.extend(["No matching case found in the available rankings.", ""])
            continue
        for case in cases[bucket]:
            lines.extend([f"### {case['query_id']}", "", f"Query: {case['query']}", ""])
            for method, payload in case["methods"].items():
                lines.append(f"**{method}** positive rank: {payload['positive_rank']}")
                for item in payload["top5"]:
                    lines.append(
                        f"- rank {item['rank']} | label={item['label']} | score={item['score']:.4f} | {item['doc_id']} | {item['text_preview']}"
                    )
                lines.append("")
    ensure_parent(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = read_jsonl(DATA_PATH)
    rankings = load_rankings()
    cases = pick_cases(records, rankings)
    write_json(OUTPUT_JSON, cases)
    write_markdown(cases, OUTPUT_MD)
    print(f"Saved {OUTPUT_JSON} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
