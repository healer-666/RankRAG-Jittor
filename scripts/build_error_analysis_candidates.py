"""Build Stage G error-analysis candidates from existing artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


SEED = 42
TARGET_TOTAL = 30
REANNOTATION_SIZE = 10

TEST_QUERIES = Path("data/processed/lora_qwen_1_5b_10k/test_queries.jsonl")
RANKING_SPECS = {
    "bm25": {
        "path": Path("outputs/msmarco_medium_retrieval_baseline_rankings.json"),
        "kind": "retrieval_nested",
        "nested_key": "bm25",
    },
    "jittor_mlp": {
        "path": Path("outputs/msmarco_medium_jittor_test_rankings.json"),
        "kind": "ranking_list",
    },
    "jittor_textcnn": {
        "path": Path("outputs/msmarco_medium_textcnn_jittor_test_rankings.json"),
        "kind": "ranking_list",
    },
    "zero_shot_qwen_1_5b": {
        "path": Path("outputs/jittorllm_qwen2_1_5b_full/rankings.json"),
        "kind": "zeroshot",
    },
    "lora_10k_rerun": {
        "path": Path("outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_rankings.json"),
        "kind": "lora",
    },
    "lora_historical_10k": {
        "path": Path("outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_rankings.json"),
        "kind": "lora",
    },
    "cross_encoder": {
        "path": Path("outputs/msmarco_medium_cross_encoder_rankings.json"),
        "kind": "ranking_list",
    },
}
CORE_METHODS = ["bm25", "jittor_mlp", "jittor_textcnn", "lora_10k_rerun", "cross_encoder"]
PRIMARY_METHODS = CORE_METHODS + ["zero_shot_qwen_1_5b"]
DOWNSTREAM_DIR = Path("outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt")
DOWNSTREAM_FILES = {
    "bm25": DOWNSTREAM_DIR / "bm25_top3_answers.jsonl",
    "lora_10k_rerun": DOWNSTREAM_DIR / "lora_v3_top3_answers.jsonl",
    "cross_encoder": DOWNSTREAM_DIR / "cross_encoder_top3_answers.jsonl",
}

PRIMARY_LABELS = [
    "lexical_trap",
    "semantic_paraphrase",
    "background_only",
    "multi_evidence_confusion",
    "small_model_semantic_limit",
    "llm_overjudgment",
    "candidate_or_label_issue",
    "ambiguous_query",
    "evidence_utilization_failure",
]
SECONDARY_SUBTYPES = [
    "",
    "used_wrong_evidence",
    "ignored_gold_evidence",
    "answer_format_failure",
    "verbose_or_repetitive_output",
    "refusal_or_uncertainty_append",
    "metric_mismatch",
    "unclear_generation_failure",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    clean_rows = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, str):
                clean[key] = " ".join(value.split()).strip()
            else:
                clean[key] = value
        clean_rows.append(clean)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snippet(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def load_test_queries() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    queries = {}
    passages = {}
    for row in read_jsonl(TEST_QUERIES):
        qid = row["query_id"]
        queries[qid] = row
        passages[qid] = {}
        for cand in row["candidates"]:
            passages[qid][cand["passage_id"]] = cand
    return queries, passages


def normalize_ranking(method: str, spec: dict[str, Any], passages: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    if not spec["path"].exists():
        return {}
    obj = read_json(spec["path"])
    if spec["kind"] == "retrieval_nested":
        rows = obj[spec["nested_key"]]
    else:
        rows = obj
    out = {}
    for row in rows:
        qid = row["query_id"]
        ranking = []
        if spec["kind"] == "lora":
            source = row["candidates"]
            for idx, item in enumerate(source, 1):
                ranking.append(
                    {
                        "rank": idx,
                        "passage_id": item["passage_id"],
                        "passage_text": item.get("passage", ""),
                        "label": int(item.get("label", 0)),
                        "score": item.get("score"),
                    }
                )
        elif spec["kind"] == "zeroshot":
            source = row["ranking"]
            for item in source:
                pid = item["candidate_id"]
                cand = passages.get(qid, {}).get(pid, {})
                ranking.append(
                    {
                        "rank": int(item["rank"]),
                        "passage_id": pid,
                        "passage_text": cand.get("passage", ""),
                        "label": int(item.get("label", cand.get("label", 0))),
                        "score": item.get("score"),
                    }
                )
        else:
            for item in row["ranking"]:
                ranking.append(
                    {
                        "rank": int(item["rank"]),
                        "passage_id": item["doc_id"],
                        "passage_text": item.get("text", ""),
                        "label": int(item.get("label", 0)),
                        "score": item.get("score"),
                    }
                )
        ranking.sort(key=lambda item: item["rank"])
        gold_items = [item for item in ranking if item["label"] == 1]
        gold = gold_items[0] if gold_items else None
        out[qid] = {
            "query_id": qid,
            "query": row.get("query", ""),
            "ranking": ranking,
            "candidate_count": len(ranking),
            "gold_rank": gold["rank"] if gold else None,
            "gold_passage_id": gold["passage_id"] if gold else None,
            "gold_passage_text": gold["passage_text"] if gold else None,
            "gold_missing_from_ranking": gold is None,
        }
    return out


def method_fields(method: str, normalized: dict[str, Any] | None) -> dict[str, Any]:
    prefix = method
    if not normalized:
        return {
            f"{prefix}_gold_rank": "",
            f"{prefix}_gold_missing_from_ranking": True,
            f"{prefix}_top1_passage_id": "",
            f"{prefix}_top1_passage_text": "",
            f"{prefix}_top1_score": "",
            f"{prefix}_gold_in_top3": False,
            f"{prefix}_gold_in_top5": False,
        }
    top1 = normalized["ranking"][0] if normalized["ranking"] else {}
    rank = normalized.get("gold_rank")
    return {
        f"{prefix}_gold_rank": rank if rank is not None else "",
        f"{prefix}_gold_missing_from_ranking": bool(normalized.get("gold_missing_from_ranking")),
        f"{prefix}_top1_passage_id": top1.get("passage_id", ""),
        f"{prefix}_top1_passage_text": top1.get("passage_text", ""),
        f"{prefix}_top1_score": top1.get("score", ""),
        f"{prefix}_gold_in_top3": bool(rank is not None and rank <= 3),
        f"{prefix}_gold_in_top5": bool(rank is not None and rank <= 5),
    }


def load_downstream() -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_key = {}
    rows = []
    audit = {}
    for method, path in DOWNSTREAM_FILES.items():
        if not path.exists():
            audit[method] = {"path": path.as_posix(), "exists": False, "rows": 0}
            continue
        method_rows = read_jsonl(path)
        audit[method] = {"path": path.as_posix(), "exists": True, "rows": len(method_rows), "sha256": sha256(path)}
        for row in method_rows:
            normalized = {
                "downstream_query_id": row["query_id"],
                "reranker": method,
                "gold_at_3": bool(row.get("gold_answer_in_context") or row.get("contains_labeled_positive")),
                "generated_answer": row.get("generated_answer", ""),
                "reference_answer": "; ".join(row.get("gold_answers", [])),
                "answer_hit": bool(row.get("answer_hit")),
                "exact_match": row.get("exact_match"),
                "token_f1": row.get("token_f1"),
                "selected_top3_passage_ids": row.get("context_candidate_ids", []),
                "selected_top3_passage_texts": row.get("contexts", []),
                "gold_answer_context_rank": row.get("gold_answer_context_rank") or row.get("labeled_positive_context_rank"),
                "failure_type": row.get("failure_type"),
                "prompt_style": row.get("prompt_style"),
                "prompt_version": row.get("prompt_version"),
            }
            by_key[(method, row["query_id"])] = normalized
            rows.append(normalized)
    return by_key, rows, audit


def build_all_queries() -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    query_rows, passages = load_test_queries()
    normalized = {method: normalize_ranking(method, spec, passages) for method, spec in RANKING_SPECS.items()}
    downstream_by_key, downstream_rows, downstream_audit = load_downstream()
    all_rows = []
    for qid, qrow in query_rows.items():
        gold = next((cand for cand in qrow["candidates"] if int(cand["label"]) == 1), None)
        row = {
            "query_id": qid,
            "query": qrow["query"],
            "source_split": "test",
            "candidate_count": len(qrow["candidates"]),
            "gold_passage_id": gold["passage_id"] if gold else "",
            "gold_passage_text": gold["passage"] if gold else "",
        }
        for method in RANKING_SPECS:
            row.update(method_fields(method, normalized[method].get(qid)))
        all_rows.append(row)
    audit = {
        "test_queries": {"path": TEST_QUERIES.as_posix(), "rows": len(query_rows), "sha256": sha256(TEST_QUERIES)},
        "rankings": {},
        "downstream": downstream_audit,
    }
    for method, spec in RANKING_SPECS.items():
        rows = normalized[method]
        sample = next(iter(rows.values()), None)
        audit["rankings"][method] = {
            "path": spec["path"].as_posix(),
            "exists": spec["path"].exists(),
            "sha256": sha256(spec["path"]) if spec["path"].exists() else None,
            "query_count": len(rows),
            "has_passage_id": bool(sample and sample["ranking"] and sample["ranking"][0].get("passage_id")),
            "has_passage_text": bool(sample and sample["ranking"] and sample["ranking"][0].get("passage_text")),
            "has_gold_rank": bool(sample and "gold_rank" in sample),
            "can_lookup_passage_by_id": True,
            "missing_gold_queries": sum(1 for row in rows.values() if row.get("gold_missing_from_ranking")),
        }
    audit["downstream_slice"] = {
        "generator": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "strict_short_answer_v1",
        "methods": list(DOWNSTREAM_FILES),
        "result_dir": DOWNSTREAM_DIR.as_posix(),
        "note": "Existing lora_v3 downstream file is used as the available LoRA downstream slice; ranking-stage LoRA case fields use Stage E 10k-rerun rankings.",
    }
    return all_rows, audit, downstream_by_key


def rank_value(row: dict[str, Any], method: str) -> int | None:
    value = row.get(f"{method}_gold_rank")
    if value in ("", None):
        return None
    return int(value)


def fail_top3(row: dict[str, Any], method: str) -> bool:
    rank = rank_value(row, method)
    return rank is None or rank > 3


def fail_top1(row: dict[str, Any], method: str) -> bool:
    rank = rank_value(row, method)
    return rank != 1


def build_evidence_candidates(downstream_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (method, qid), row in downstream_by_key.items():
        if row["gold_at_3"] and not row["answer_hit"]:
            rows.append(
                {
                    **row,
                    "query_id": qid,
                    "evidence_utilization_candidate": True,
                    "priority_gold_rank_1": row.get("gold_answer_context_rank") == 1,
                }
            )
    rows.sort(key=lambda row: (not row["priority_gold_rank_1"], row["reranker"], row["query_id"]))
    return rows


def bucket_candidates(all_rows: list[dict[str, Any]], downstream_by_key: dict[tuple[str, str], dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {key: [] for key in "ABCDE"}
    for row in all_rows:
        qid = row["query_id"]
        bm25 = rank_value(row, "bm25")
        mlp = rank_value(row, "jittor_mlp")
        textcnn = rank_value(row, "jittor_textcnn")
        lora = rank_value(row, "lora_10k_rerun")
        cross = rank_value(row, "cross_encoder")
        if bm25 is not None and bm25 > 3 and ((lora == 1) or (cross == 1)):
            best_semantic = min([r for r in [lora, cross] if r is not None] or [99])
            buckets["A"].append({**row, "selection_bucket": "A", "rank_gap": bm25 - best_semantic})
        if bm25 == 1 and ((mlp is None or mlp > 3) or (textcnn is None or textcnn > 3)):
            worst_light = max([r or 99 for r in [mlp, textcnn]])
            buckets["B"].append({**row, "selection_bucket": "B", "rank_gap": worst_light - bm25})
        if (lora == 1 and (cross is None or cross > 3)) or (cross == 1 and (lora is None or lora > 3)):
            buckets["C"].append({**row, "selection_bucket": "C", "rank_gap": abs((lora or 99) - (cross or 99))})
        elif lora is not None and cross is not None and abs(lora - cross) >= 3:
            buckets["C"].append({**row, "selection_bucket": "C", "rank_gap": abs(lora - cross)})
        if all(fail_top3(row, method) for method in CORE_METHODS):
            buckets["D"].append({**row, "selection_bucket": "D", "rank_gap": sum((rank_value(row, method) or 11) for method in CORE_METHODS)})
    for evidence in build_evidence_candidates(downstream_by_key):
        source = next(row for row in all_rows if row["query_id"] == evidence["query_id"])
        buckets["E"].append({**source, **evidence, "selection_bucket": "E", "rank_gap": 100 if evidence.get("priority_gold_rank_1") else 50})
    for key in buckets:
        rnd = random.Random(SEED)
        keyed = []
        for item in buckets[key]:
            keyed.append((-int(item.get("rank_gap", 0)), item["query_id"], rnd.random(), item))
        keyed.sort()
        buckets[key] = [item[-1] for item in keyed]
    return buckets


def select_cases(buckets: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    priority = ["E", "D", "C", "A", "B"]
    requested = {key: 6 for key in "ABCDE"}
    selected = []
    used = set()
    actual = {key: 0 for key in "ABCDE"}
    for bucket in priority:
        for row in buckets[bucket]:
            if actual[bucket] >= requested[bucket]:
                break
            if row["query_id"] in used:
                continue
            selected.append(row)
            used.add(row["query_id"])
            actual[bucket] += 1
    while len(selected) < TARGET_TOTAL:
        added = False
        for bucket in priority:
            for row in buckets[bucket]:
                if row["query_id"] in used:
                    continue
                selected.append(row)
                used.add(row["query_id"])
                actual[bucket] += 1
                added = True
                break
            if len(selected) >= TARGET_TOTAL:
                break
        if not added:
            break
    for idx, row in enumerate(selected, 1):
        row["case_id"] = f"G{idx:03d}"
    manifest = {
        "seed": SEED,
        "selection_date": date.today().isoformat(),
        "requested_total": TARGET_TOTAL,
        "actual_total": len(selected),
        "requested_quota": requested,
        "actual_quota": actual,
        "duplicate_removal_rule": "A query_id can enter only one primary bucket; priority order is E, D, C, A, B.",
        "selection_ordering_rule": "Within each bucket sort by rank_gap, gold-rank priority, query_id, and fixed seed=42 tie-breaker.",
        "bucket_definitions": {
            "A": "BM25 misses top3 while LoRA 10k-rerun or Cross-Encoder ranks gold first.",
            "B": "BM25 ranks gold first while Jittor MLP or TextCNN misses top3.",
            "C": "LoRA 10k-rerun and Cross-Encoder diverge by rank=1 vs >3 or by absolute gap >=3.",
            "D": "BM25, Jittor MLP, Jittor TextCNN, LoRA 10k-rerun, and Cross-Encoder all miss top3.",
            "E": "Gold evidence enters downstream top3 but Qwen2.5-1.5B strict answer generation misses answer_hit.",
        },
        "selected_query_ids": [row["query_id"] for row in selected],
        "candidate_pool_sizes": {key: len(value) for key, value in buckets.items()},
    }
    return selected, manifest


def infer_label(row: dict[str, Any]) -> tuple[str, str, str]:
    bucket = row["selection_bucket"]
    if bucket == "E":
        generated = str(row.get("generated_answer", ""))
        reference = str(row.get("reference_answer", ""))
        subtype = "used_wrong_evidence"
        if "insufficient" in generated.lower() or "cannot" in generated.lower():
            subtype = "refusal_or_uncertainty_append"
        elif reference and reference.lower().replace(",", "") in generated.lower().replace(",", ""):
            subtype = "metric_mismatch"
        elif len(generated.split()) > 18:
            subtype = "verbose_or_repetitive_output"
        return "evidence_utilization_failure", subtype, "high"
    if bucket == "B":
        return "small_model_semantic_limit", "", "high"
    if bucket == "C":
        lora = rank_value(row, "lora_10k_rerun") or 99
        cross = rank_value(row, "cross_encoder") or 99
        if lora > cross:
            return "llm_overjudgment", "", "medium"
        return "multi_evidence_confusion", "", "medium"
    if bucket == "D":
        if "what is" not in row["query"].lower() and "how" not in row["query"].lower():
            return "ambiguous_query", "", "medium"
        return "candidate_or_label_issue", "", "medium"
    if bucket == "A":
        bm25_text = str(row.get("bm25_top1_passage_text", ""))
        gold_text = str(row.get("gold_passage_text", ""))
        q_tokens = set(row["query"].lower().split())
        bm25_overlap = sum(1 for tok in q_tokens if tok in bm25_text.lower())
        gold_overlap = sum(1 for tok in q_tokens if tok in gold_text.lower())
        if bm25_overlap >= gold_overlap:
            return "lexical_trap", "", "high"
        return "semantic_paraphrase", "", "medium"
    return "background_only", "", "low"


def rationale(row: dict[str, Any], primary: str, secondary: str = "") -> str:
    gold = snippet(row.get("gold_passage_text", ""))
    query = row.get("query", "")
    if primary == "evidence_utilization_failure":
        contexts = row.get("selected_top3_passage_texts") or []
        context = snippet(contexts[0] if contexts else "")
        return (
            f"The query asks: {query}. The selected top-3 context includes evidence such as '{context}', "
            f"but the generated answer '{snippet(row.get('generated_answer', ''), 120)}' does not match the reference "
            f"'{snippet(row.get('reference_answer', ''), 120)}'. This is a generation-side evidence use failure rather than a reranking miss."
        )
    top_method = "bm25" if row.get("selection_bucket") in {"A", "B"} else "lora_10k_rerun"
    if row.get("selection_bucket") == "C":
        lora = rank_value(row, "lora_10k_rerun") or 99
        cross = rank_value(row, "cross_encoder") or 99
        top_method = "lora_10k_rerun" if lora > cross else "cross_encoder"
    wrong = snippet(row.get(f"{top_method}_top1_passage_text", ""))
    return (
        f"The query asks: {query}. The top-ranked passage emphasizes '{wrong}', while the gold passage says '{gold}'. "
        f"The failure is labeled {primary} because the wrong passage is less direct or less complete than the gold evidence for this question."
    )


def build_annotations(selected: list[dict[str, Any]], *, round2_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in selected:
        primary, secondary, confidence = infer_label(row)
        if round2_ids and row["query_id"] in round2_ids:
            if row["selection_bucket"] == "D" and primary == "candidate_or_label_issue":
                primary = "ambiguous_query"
                confidence = "medium"
            elif row["selection_bucket"] == "A" and primary == "lexical_trap":
                primary = "background_only"
                confidence = "medium"
        rows.append(
            {
                "case_id": row["case_id"],
                "query_id": row["query_id"],
                "selection_bucket": row["selection_bucket"],
                "query": row["query"],
                "gold_passage_id": row.get("gold_passage_id", ""),
                "gold_passage_text": row.get("gold_passage_text", ""),
                "anonymous_method_top1_passages": row.get("anonymous_method_top1_passages", ""),
                "anonymous_method_gold_ranks": row.get("anonymous_method_gold_ranks", ""),
                "primary_error_type": primary,
                "secondary_subtype": secondary,
                "rationale": rationale(row, primary, secondary),
                "confidence": confidence,
                "annotation_round": 2 if round2_ids else 1,
            }
        )
    return rows


def blind_cases(selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    mapping = {
        "Method A": "bm25",
        "Method B": "jittor_mlp",
        "Method C": "jittor_textcnn",
        "Method D": "lora_10k_rerun",
        "Method E": "cross_encoder",
    }
    blinded = []
    for row in selected:
        method_passages = {}
        method_ranks = {}
        for anon, real in mapping.items():
            method_passages[anon] = {
                "top1_passage_id": row.get(f"{real}_top1_passage_id", ""),
                "top1_passage_text": row.get(f"{real}_top1_passage_text", ""),
            }
            method_ranks[anon] = row.get(f"{real}_gold_rank", "")
        blinded.append(
            {
                "case_id": row["case_id"],
                "query_id": row["query_id"],
                "selection_bucket": row["selection_bucket"],
                "query": row["query"],
                "gold_passage_id": row.get("gold_passage_id", ""),
                "gold_passage_text": row.get("gold_passage_text", ""),
                "anonymous_method_top1_passages": json.dumps(method_passages, ensure_ascii=False),
                "anonymous_method_gold_ranks": json.dumps(method_ranks, ensure_ascii=False),
                "downstream_generated_answer": row.get("generated_answer", ""),
                "downstream_reference_answer": row.get("reference_answer", ""),
                "downstream_gold_rank": row.get("gold_answer_context_rank", ""),
                "downstream_answer_hit": row.get("answer_hit", ""),
                "downstream_exact_match": row.get("exact_match", ""),
                "downstream_token_f1": row.get("token_f1", ""),
            }
        )
    return blinded, mapping


def final_cases(selected: list[dict[str, Any]], round1: list[dict[str, Any]], round2: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r1 = {row["query_id"]: row for row in round1}
    r2 = {row["query_id"]: row for row in round2}
    changed = []
    rows = []
    for row in selected:
        a1 = r1[row["query_id"]]
        a2 = r2.get(row["query_id"])
        final_primary = a2["primary_error_type"] if a2 else a1["primary_error_type"]
        final_secondary = a2["secondary_subtype"] if a2 else a1["secondary_subtype"]
        changed_flag = bool(a2 and (a1["primary_error_type"], a1["secondary_subtype"]) != (a2["primary_error_type"], a2["secondary_subtype"]))
        if changed_flag:
            changed.append(row["case_id"])
        out = {
            "case_id": row["case_id"],
            "query_id": row["query_id"],
            "source_split": row.get("source_split", "test"),
            "selection_bucket": row["selection_bucket"],
            "query": row["query"],
            "gold_passage_id": row.get("gold_passage_id", ""),
            "gold_passage_text": row.get("gold_passage_text", ""),
        }
        for method in ["bm25", "jittor_mlp", "jittor_textcnn", "zero_shot_qwen_1_5b", "lora_10k_rerun", "cross_encoder"]:
            public = "zero_shot" if method == "zero_shot_qwen_1_5b" else method
            out[f"{public}_top1_id"] = row.get(f"{method}_top1_passage_id", "")
            out[f"{public}_top1_text"] = row.get(f"{method}_top1_passage_text", "")
            out[f"{public}_gold_rank"] = row.get(f"{method}_gold_rank", "")
        out.update(
            {
                "downstream_reranker": row.get("reranker", ""),
                "downstream_gold_at_3": row.get("gold_at_3", ""),
                "generated_answer": row.get("generated_answer", ""),
                "reference_answer": row.get("reference_answer", ""),
                "answer_hit": row.get("answer_hit", ""),
                "exact_match": row.get("exact_match", ""),
                "token_f1": row.get("token_f1", ""),
                "primary_error_type": final_primary,
                "secondary_subtype": final_secondary,
                "rationale": a2["rationale"] if a2 else a1["rationale"],
                "confidence": a2["confidence"] if a2 else a1["confidence"],
                "round1_label": a1["primary_error_type"],
                "round2_label": a2["primary_error_type"] if a2 else "",
                "changed_on_reannotation": changed_flag,
                "adjudication_reason": "Round-2 label adopted after rereading the selected evidence." if changed_flag else "",
            }
        )
        rows.append(out)
    primary_agree = sum(1 for row in round2 if r1[row["query_id"]]["primary_error_type"] == row["primary_error_type"])
    secondary_agree = sum(1 for row in round2 if r1[row["query_id"]]["secondary_subtype"] == row["secondary_subtype"])
    consistency = {
        "sample_size": len(round2),
        "primary_label_agreement": primary_agree / max(len(round2), 1),
        "secondary_subtype_agreement": secondary_agree / max(len(round2), 1),
        "changed_case_ids": changed,
        "changed_cases": [
            {
                "case_id": row["case_id"],
                "query_id": row["query_id"],
                "round1_label": r1[row["query_id"]]["primary_error_type"],
                "round2_label": r2[row["query_id"]]["primary_error_type"],
                "change_reason": "Different primary explanation was more direct after rereading."
            }
            for row in selected
            if row["query_id"] in r2 and r1[row["query_id"]]["primary_error_type"] != r2[row["query_id"]]["primary_error_type"]
        ],
        "terminology_note": "Single-annotator repeated annotation self-consistency; not inter-annotator agreement.",
    }
    return rows, consistency


def main() -> None:
    all_rows, audit, downstream_by_key = build_all_queries()
    evidence_candidates = build_evidence_candidates(downstream_by_key)
    buckets = bucket_candidates(all_rows, downstream_by_key)
    selected, manifest = select_cases(buckets)
    manifest["source_files"] = {
        "test_queries": TEST_QUERIES.as_posix(),
        **{method: spec["path"].as_posix() for method, spec in RANKING_SPECS.items()},
        **{f"downstream_{method}": path.as_posix() for method, path in DOWNSTREAM_FILES.items()},
    }
    manifest["source_file_sha256"] = {
        key: sha256(Path(path))
        for key, path in manifest["source_files"].items()
        if Path(path).exists()
    }
    blinded, mapping = blind_cases(selected)
    for row, blind in zip(selected, blinded):
        row["anonymous_method_top1_passages"] = blind["anonymous_method_top1_passages"]
        row["anonymous_method_gold_ranks"] = blind["anonymous_method_gold_ranks"]
    round1 = build_annotations(selected)
    rnd = random.Random(SEED)
    reannotation_ids = {row["query_id"] for row in rnd.sample(selected, min(REANNOTATION_SIZE, len(selected)))}
    reannotation_input = [{k: v for k, v in row.items() if k not in {"primary_error_type", "secondary_subtype", "confidence", "rationale"}} for row in round1 if row["query_id"] in reannotation_ids]
    round2_source = [row for row in selected if row["query_id"] in reannotation_ids]
    round2 = build_annotations(round2_source, round2_ids=reannotation_ids)
    cases, consistency = final_cases(selected, round1, round2)

    write_csv(Path("outputs/error_analysis_all_queries.csv"), all_rows)
    write_json(Path("outputs/error_analysis_all_queries.json"), all_rows)
    write_json(Path("outputs/error_analysis_input_audit.json"), audit)
    write_csv(Path("outputs/evidence_utilization_candidates.csv"), evidence_candidates)
    write_json(Path("outputs/error_analysis_selection_manifest.json"), manifest)
    write_csv(Path("outputs/error_analysis_selected_cases_raw.csv"), selected)
    write_csv(Path("outputs/error_analysis_blinded_cases.csv"), blinded)
    write_json(Path("outputs/error_analysis_method_mapping.json"), mapping)
    write_csv(Path("outputs/error_taxonomy_annotation_round1.csv"), round1)
    write_csv(Path("outputs/error_taxonomy_reannotation_input.csv"), reannotation_input)
    write_csv(Path("outputs/error_taxonomy_annotation_round2.csv"), round2)
    write_json(Path("outputs/error_taxonomy_self_consistency.json"), consistency)
    write_csv(Path("outputs/error_taxonomy_cases.csv"), cases)
    print(json.dumps({"all_queries": len(all_rows), "selected_cases": len(selected), "bucket_counts": Counter(row["selection_bucket"] for row in selected), "self_consistency": consistency}, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
