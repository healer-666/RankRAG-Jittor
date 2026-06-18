"""Evaluate the Jittor TextCNN reranker."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("use_cuda", "0")
os.environ.setdefault("nvcc_path", "")

import numpy as np

from metrics import evaluate_grouped
from model_textcnn_jittor import TextCNNScorer, require_jittor
from text_dataset import load_text_candidate_data, load_text_demo_candidates, load_vocab
from utils import load_config, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msmarco_textcnn.yaml")
    parser.add_argument("--model_path", default=None)
    return parser.parse_args()


def output_path(config: dict, key: str, default: str) -> str:
    return str(config.get("outputs", {}).get(key, default))


def score_rows(model: TextCNNScorer, rows: list[dict], jt, batch_size: int = 128) -> list[dict]:
    model.eval()
    scored: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids = np.stack([row["input_ids"] for row in batch]).astype("int64")
        scores = model(jt.array(input_ids)).numpy().reshape(-1).tolist()
        for row, score in zip(batch, scores):
            scored_row = {key: value for key, value in row.items() if key != "input_ids"}
            scored_row["score"] = float(score)
            scored.append(scored_row)
    return scored


def rank_group(scored_rows: list[dict]) -> list[dict]:
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


def main() -> None:
    try:
        require_jittor()
        import jittor as jt
        jt.flags.use_cuda = 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None

    args = parse_args()
    config = load_config(args.config)
    vocab = load_vocab(config["text"]["vocab_path"])
    model = TextCNNScorer(
        vocab_size=len(vocab),
        embed_dim=int(config["model"]["embed_dim"]),
        num_filters=int(config["model"]["num_filters"]),
        kernel_sizes=list(config["model"]["kernel_sizes"]),
        dropout=float(config["model"]["dropout"]),
    )
    model_path = args.model_path or output_path(config, "jittor_model", "outputs/msmarco_textcnn_jittor_model.pkl")
    model.load(str(resolve_path(model_path)))

    test_rows = load_text_candidate_data(config["data"]["test_path"], vocab, int(config["text"]["max_len"]))
    scored_test = score_rows(model, test_rows, jt)
    metrics = evaluate_grouped(scored_test, topk=list(config["eval"]["topk"]))
    write_json(output_path(config, "jittor_metrics", "outputs/msmarco_textcnn_jittor_metrics.json"), metrics)
    write_json("outputs/msmarco_textcnn_jittor_rankings.json", rank_group(scored_test))

    demo_rows = load_text_demo_candidates(config["data"]["demo_path"], vocab, int(config["text"]["max_len"]))
    demo_rankings = rank_group(score_rows(model, demo_rows, jt))
    write_json(output_path(config, "jittor_demo", "outputs/msmarco_textcnn_demo_ranking_result_jittor.json"), demo_rankings[0] if demo_rankings else {})

    print("Jittor TextCNN metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
