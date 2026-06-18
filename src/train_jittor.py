"""Train the Jittor lightweight RankRAG-style context scorer."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dataset import load_candidate_data, load_pairwise_data
from metrics import evaluate_grouped
from model_jittor import MLPScorer, pairwise_ranking_loss, require_jittor
from utils import ensure_parent, load_config, resolve_path, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=None)
    return parser.parse_args()


def iter_batches(x_pos: np.ndarray, x_neg: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(len(x_pos))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x_pos[batch_idx], x_neg[batch_idx]


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
    set_seed(int(config["seed"]))
    jt.set_global_seed(int(config["seed"]))

    logger = setup_logger(output_path(config, "jittor_log", "logs/jittor_train.log"), "jittor_train")
    epochs = args.epochs if args.epochs is not None else int(config["train"]["epochs"])
    batch_size = int(config["train"]["batch_size"])

    x_pos, x_neg = load_pairwise_data(config["data"]["train_path"], max_pairs=args.max_train_pairs)
    valid_rows = load_candidate_data(config["data"]["valid_path"])

    model = MLPScorer(
        input_dim=int(config["model"]["input_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    optimizer = jt.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    rng = np.random.default_rng(int(config["seed"]))

    logger.info("Training Jittor scorer on %d pairwise samples", len(x_pos))
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_pos, batch_neg in iter_batches(x_pos, x_neg, batch_size, rng):
            score_pos = model(jt.array(batch_pos.astype("float32")))
            score_neg = model(jt.array(batch_neg.astype("float32")))
            loss = pairwise_ranking_loss(score_pos, score_neg)
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.numpy()).reshape(-1)[0]))

        valid_metrics = evaluate_grouped(score_rows(model, valid_rows, jt), topk=list(config["eval"]["topk"]))
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

    model_path = ensure_parent(output_path(config, "jittor_model", "outputs/jittor_model.pkl"))
    jt.save(model.state_dict(), str(model_path))
    logger.info("Saved model to %s", resolve_path(model_path))


if __name__ == "__main__":
    main()
