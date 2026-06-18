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
    parser.add_argument("--model_path", default="outputs/jittor_model.pkl")
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
    model.load(str(resolve_path(args.model_path)))

    test_rows = load_candidate_data(config["data"]["test_path"])
    scored_test = score_rows(model, test_rows, jt)
    metrics = evaluate_grouped(scored_test, topk=list(config["eval"]["topk"]))
    write_json("outputs/jittor_metrics.json", metrics)

    demo_rows = load_demo_candidates(config["data"]["demo_path"])
    write_json("outputs/demo_ranking_result_jittor.json", rank_demo(score_rows(model, demo_rows, jt)))

    print("Jittor metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
