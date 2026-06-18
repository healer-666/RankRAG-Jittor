"""Generate MS MARCO case study and lightweight error analysis."""

from __future__ import annotations

import json
from pathlib import Path

from dataset import read_jsonl
from text_dataset import tokenize
from utils import ensure_parent, resolve_path, write_json


METHOD_FILES = {
    "TF-IDF": ("outputs/msmarco_retrieval_baseline_rankings.json", "tfidf"),
    "BM25": ("outputs/msmarco_retrieval_baseline_rankings.json", "bm25"),
    "MLP Jittor": ("outputs/msmarco_jittor_rankings.json", None),
    "TextCNN Jittor": ("outputs/msmarco_textcnn_jittor_rankings.json", None),
}


def load_rankings() -> dict[str, dict[str, dict]]:
    loaded: dict[str, dict[str, dict]] = {}
    for method, (path, key) in METHOD_FILES.items():
        resolved = resolve_path(path)
        if not resolved.exists():
            print(f"case study: skipping {method}, missing {path}")
            continue
        data = json.loads(resolved.read_text(encoding="utf-8"))
        rankings = data.get(key, []) if key else data
        loaded[method] = {item["query_id"]: item for item in rankings}
    return loaded


def positive_rank(ranking: dict) -> int | None:
    for item in ranking.get("ranking", []):
        if int(item.get("label", 0)) == 1:
            return int(item["rank"])
    return None


def overlap_note(query: str, text: str) -> str:
    q_terms = set(tokenize(query))
    p_terms = set(tokenize(text))
    overlap = sorted(q_terms & p_terms)
    if len(overlap) >= 3:
        return f"Keyword overlap is visible ({', '.join(overlap[:6])}), which can help lexical baselines."
    if overlap:
        return f"Keyword overlap is limited ({', '.join(overlap[:6])}), so the ranker needs more than surface matching."
    return "Keyword overlap is very low, which makes the case difficult for lightweight lexical and neural rankers."


def analysis_for(query: str, ranking: dict, rank: int | None) -> str:
    if rank is None:
        return "No positive passage was found in the ranking output."
    top = ranking.get("ranking", [{}])[0]
    note = overlap_note(query, top.get("text", ""))
    if rank == 1:
        return f"The positive passage is ranked first. {note}"
    if rank <= 3:
        return f"The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. {note}"
    return f"The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. {note}"


def compact_ranking(ranking: dict) -> list[dict]:
    return [
        {
            "rank": item["rank"],
            "doc_id": item["doc_id"],
            "label": int(item["label"]),
            "score": float(item["score"]),
            "text_preview": item.get("text", "")[:200],
        }
        for item in ranking.get("ranking", [])
    ]


def choose_cases(records: list[dict], rankings_by_method: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    primary_name = "TextCNN Jittor" if "TextCNN Jittor" in rankings_by_method else next(iter(rankings_by_method), "")
    if not primary_name:
        return {"success": [], "middle": [], "failure": []}

    buckets = {"success": [], "middle": [], "failure": []}
    primary = rankings_by_method[primary_name]
    for record in records:
        query_id = record["query_id"]
        ranking = primary.get(query_id)
        if not ranking:
            continue
        rank = positive_rank(ranking)
        if rank == 1:
            bucket = "success"
        elif rank in {2, 3}:
            bucket = "middle"
        elif rank in {4, 5}:
            bucket = "failure"
        else:
            continue
        if len(buckets[bucket]) >= 3:
            continue

        method_summaries = {}
        for method, method_rankings in rankings_by_method.items():
            method_ranking = method_rankings.get(query_id)
            if not method_ranking:
                continue
            method_summaries[method] = {
                "positive_rank": positive_rank(method_ranking),
                "ranking": compact_ranking(method_ranking),
                "analysis": analysis_for(record["query"], method_ranking, positive_rank(method_ranking)),
            }

        buckets[bucket].append(
            {
                "query_id": query_id,
                "query": record["query"],
                "bucket": bucket,
                "primary_method": primary_name,
                "methods": method_summaries,
            }
        )
    return buckets


def write_markdown(cases: dict[str, list[dict]], path: str) -> None:
    lines = [
        "# MS MARCO Case Study and Error Analysis",
        "",
        "These examples are selected from the small MS MARCO subset and use automatic rule-based notes. They should be read as qualitative diagnostics, not proof of generalization.",
        "",
    ]
    for bucket, title in [("success", "Success Cases"), ("middle", "Middle Cases"), ("failure", "Failure Cases")]:
        lines.extend([f"## {title}", ""])
        if not cases[bucket]:
            lines.extend(["No cases available for this bucket.", ""])
            continue
        for case in cases[bucket]:
            lines.extend([f"### {case['query_id']}", "", f"Query: {case['query']}", ""])
            for method, payload in case["methods"].items():
                lines.extend([f"**{method}** positive rank: {payload['positive_rank']}", "", payload["analysis"], ""])
                for item in payload["ranking"][:5]:
                    lines.append(
                        f"- rank {item['rank']} | {item['doc_id']} | label={item['label']} | score={item['score']:.4f} | {item['text_preview']}"
                    )
                lines.append("")
    ensure_parent(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = read_jsonl("data/processed/msmarco/test.jsonl")
    rankings = load_rankings()
    cases = choose_cases(records, rankings)
    write_json("outputs/msmarco_case_study.json", cases)
    write_markdown(cases, "docs/msmarco_case_study.md")
    print("Saved outputs/msmarco_case_study.json and docs/msmarco_case_study.md")


if __name__ == "__main__":
    main()
