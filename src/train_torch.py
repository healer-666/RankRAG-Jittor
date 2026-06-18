"""Train the PyTorch lightweight RankRAG-style context scorer."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_candidate_data, load_pairwise_data
from metrics import evaluate_grouped
from model_torch import MLPScorer, pairwise_ranking_loss
from utils import ensure_parent, load_config, resolve_path, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=None)
    return parser.parse_args()


def score_rows(model: MLPScorer, rows: list[dict]) -> list[dict]:
    model.eval()
    features = torch.tensor(np.stack([row["feature"] for row in rows]), dtype=torch.float32)
    with torch.no_grad():
        scores = model(features).cpu().numpy().tolist()
    scored = []
    for row, score in zip(rows, scores):
        scored_row = {key: value for key, value in row.items() if key != "feature"}
        scored_row["score"] = float(score)
        scored.append(scored_row)
    return scored


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    logger = setup_logger("logs/torch_train.log", "torch_train")
    epochs = args.epochs if args.epochs is not None else int(config["train"]["epochs"])
    batch_size = int(config["train"]["batch_size"])

    x_pos, x_neg = load_pairwise_data(config["data"]["train_path"], max_pairs=args.max_train_pairs)
    valid_rows = load_candidate_data(config["data"]["valid_path"])

    dataset = TensorDataset(torch.tensor(x_pos, dtype=torch.float32), torch.tensor(x_neg, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPScorer(
        input_dim=int(config["model"]["input_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    logger.info("Training PyTorch scorer on %d pairwise samples", len(dataset))
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_pos, batch_neg in loader:
            optimizer.zero_grad()
            score_pos = model(batch_pos)
            score_neg = model(batch_neg)
            loss = pairwise_ranking_loss(score_pos, score_neg)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        valid_metrics = evaluate_grouped(score_rows(model, valid_rows), topk=list(config["eval"]["topk"]))
        logger.info(
            "epoch=%d train_loss=%.6f valid_pairwise_acc=%.4f valid_recall@1=%.4f valid_recall@3=%.4f valid_recall@5=%.4f valid_mrr=%.4f valid_ndcg@5=%.4f",
            epoch,
            float(np.mean(losses)),
            valid_metrics["pairwise_accuracy"],
            valid_metrics["recall@1"],
            valid_metrics["recall@3"],
            valid_metrics["recall@5"],
            valid_metrics["mrr"],
            valid_metrics["ndcg@5"],
        )

    model_path = ensure_parent("outputs/torch_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to %s", resolve_path(model_path))


if __name__ == "__main__":
    main()
