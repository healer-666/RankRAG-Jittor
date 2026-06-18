"""Train the PyTorch TextCNN reranker on MS MARCO small subset."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics import evaluate_grouped
from model_textcnn_torch import TextCNNScorer, pairwise_ranking_loss
from text_dataset import load_or_build_vocab, load_pairwise_text_data, load_text_candidate_data
from utils import ensure_parent, load_config, resolve_path, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msmarco_textcnn.yaml")
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    set_seed(seed)
    torch.manual_seed(seed)

    text_cfg = config["text"]
    vocab = load_or_build_vocab(config["data"]["train_path"], text_cfg["vocab_path"], int(text_cfg["max_vocab_size"]))
    max_len = int(text_cfg["max_len"])
    x_pos, x_neg = load_pairwise_text_data(config["data"]["train_path"], vocab, max_len)
    valid_rows = load_text_candidate_data(config["data"]["valid_path"], vocab, max_len)

    batch_size = int(config["train"]["batch_size"])
    dataset = TensorDataset(torch.tensor(x_pos, dtype=torch.long), torch.tensor(x_neg, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextCNNScorer(
        vocab_size=len(vocab),
        embed_dim=int(config["model"]["embed_dim"]),
        num_filters=int(config["model"]["num_filters"]),
        kernel_sizes=list(config["model"]["kernel_sizes"]),
        dropout=float(config["model"]["dropout"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    logger = setup_logger(output_path(config, "torch_log", "logs/msmarco_textcnn_torch_train.log"), "textcnn_torch_train")

    logger.info("Training PyTorch TextCNN on %d pairwise samples; vocab=%d", len(dataset), len(vocab))
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        losses = []
        for batch_pos, batch_neg in loader:
            optimizer.zero_grad()
            loss = pairwise_ranking_loss(model(batch_pos), model(batch_neg))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        valid_metrics = evaluate_grouped(score_rows(model, valid_rows), topk=list(config["eval"]["topk"]))
        logger.info(
            "epoch=%d train_loss=%.6f valid_pairwise_acc=%.4f valid_recall@1=%.4f valid_recall@3=%.4f valid_recall@5=%.4f valid_mrr=%.4f valid_ndcg@1=%.4f valid_ndcg@3=%.4f valid_ndcg@5=%.4f",
            epoch,
            float(np.mean(losses)),
            valid_metrics["pairwise_accuracy"],
            valid_metrics["recall@1"],
            valid_metrics["recall@3"],
            valid_metrics["recall@5"],
            valid_metrics["mrr"],
            valid_metrics["ndcg@1"],
            valid_metrics["ndcg@3"],
            valid_metrics["ndcg@5"],
        )

    model_path = ensure_parent(output_path(config, "torch_model", "outputs/msmarco_textcnn_torch_model.pt"))
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to %s", resolve_path(model_path))


if __name__ == "__main__":
    main()
