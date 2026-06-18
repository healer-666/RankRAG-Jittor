"""Evaluate the PyTorch TextCNN reranker."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from metrics import evaluate_grouped
from model_textcnn_torch import TextCNNScorer
from text_dataset import load_text_candidate_data, load_text_demo_candidates, load_vocab
from utils import load_config, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msmarco_textcnn.yaml")
    parser.add_argument("--model_path", default=None)
    return parser.parse_args()


def output_path(config: dict, key: str, default: str) -> str:
    return str(config.get("outputs", {}).get(key, default))


def score_rows(model: TextCNNScorer, rows: list[dict], batch_size: int = 128) -> list[dict]:
    model.eval()
    scored: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            input_ids = torch.tensor(np.stack([row["input_ids"] for row in batch]), dtype=torch.long)
            scores = model(input_ids).cpu().numpy().reshape(-1).tolist()
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
    model_path = args.model_path or output_path(config, "torch_model", "outputs/msmarco_textcnn_torch_model.pt")
    model.load_state_dict(torch.load(resolve_path(model_path), map_location="cpu"))

    test_rows = load_text_candidate_data(config["data"]["test_path"], vocab, int(config["text"]["max_len"]))
    scored_test = score_rows(model, test_rows)
    metrics = evaluate_grouped(scored_test, topk=list(config["eval"]["topk"]))
    write_json(output_path(config, "torch_metrics", "outputs/msmarco_textcnn_torch_metrics.json"), metrics)
    write_json("outputs/msmarco_textcnn_torch_rankings.json", rank_group(scored_test))

    demo_rows = load_text_demo_candidates(config["data"]["demo_path"], vocab, int(config["text"]["max_len"]))
    demo_rankings = rank_group(score_rows(model, demo_rows))
    write_json(output_path(config, "torch_demo", "outputs/msmarco_textcnn_demo_ranking_result_torch.json"), demo_rankings[0] if demo_rankings else {})

    print("PyTorch TextCNN metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
