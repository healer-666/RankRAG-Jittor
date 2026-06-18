"""Evaluate the Jittor scorer and write metrics plus demo rankings."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dataset import load_candidate_data, load_demo_candidates
from metrics import evaluate_grouped
from model_jittor import MLPScorer, require_jittor
from utils import load_config, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_path", default=None)
    return parser.parse_args()


def score_rows(model: MLPScorer, rows: list[dict], jt) -> list[dict]:
    model.eval()
    features = np.stack([row["feature"] for row in rows]).astype("float32")
    scores = model(jt.array(features)).numpy().reshape(-1).tolist()

    scored = []
    for row, score in zip(rows, scores):
        scored_row = {key: value for key, value in row.items() if key != "feature"}
        scored_row["score"] = float(score)
        scored.append(scored_row)
    return scored


def rank_demo(scored_rows: list[dict]) -> dict:
    ranked = sorted(scored_rows, key=lambda row: row["score"], reverse=True)
    return {
        "query_id": ranked[0]["query_id"] if ranked else "",
        "query": ranked[0]["query"] if ranked else "",
        "ranking": [
            {
                "rank": index,
                "doc_id": row["doc_id"],
                "title": row.get("title", ""),
                "score": row["score"],
                "label": row["label"],
                "text": row["text"],
            }
            for index, row in enumerate(ranked, start=1)
        ],
    }


def rank_groups(scored_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in scored_rows:
        grouped.setdefault(row["query_id"], []).append(row)

    results = []
    for query_id, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
        results.append(
            {
                "query_id": query_id,
                "query": ranked[0]["query"] if ranked else "",
                "ranking": [
                    {
                        "rank": index,
                        "doc_id": row["doc_id"],
                        "title": row.get("title", ""),
                        "score": row["score"],
                        "label": row["label"],
                        "text": row["text"],
                    }
                    for index, row in enumerate(ranked, start=1)
                ],
            }
        )
    return results


def output_path(config: dict, key: str, default: str) -> str:
    return str(config.get("outputs", {}).get(key, default))


def main() -> None:
    try:
        require_jittor()
        import jittor as jt
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None

    args = parse_args()
    config = load_config(args.config)

    model = MLPScorer(
        input_dim=int(config["model"]["input_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model_path = args.model_path or output_path(config, "jittor_model", "outputs/jittor_model.pkl")
    model.load(str(resolve_path(model_path)))

    test_rows = load_candidate_data(config["data"]["test_path"])
    scored_test = score_rows(model, test_rows, jt)
    metrics = evaluate_grouped(scored_test, topk=list(config["eval"]["topk"]))
    write_json(output_path(config, "jittor_metrics", "outputs/jittor_metrics.json"), metrics)
    write_json(output_path(config, "jittor_rankings", "outputs/jittor_test_rankings.json"), rank_groups(scored_test))

    demo_rows = load_demo_candidates(config["data"]["demo_path"])
    write_json(
        output_path(config, "jittor_demo", "outputs/demo_ranking_result_jittor.json"),
        rank_demo(score_rows(model, demo_rows, jt)),
    )

    print("Jittor metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
