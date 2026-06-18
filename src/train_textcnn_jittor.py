"""Train the Jittor TextCNN reranker on MS MARCO small subset."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("use_cuda", "0")
os.environ.setdefault("nvcc_path", "")

import numpy as np

from metrics import evaluate_grouped
from model_textcnn_jittor import TextCNNScorer, pairwise_ranking_loss, require_jittor
from text_dataset import load_or_build_vocab, load_pairwise_text_data, load_text_candidate_data
from utils import ensure_parent, load_config, resolve_path, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msmarco_textcnn.yaml")
    return parser.parse_args()


def output_path(config: dict, key: str, default: str) -> str:
    return str(config.get("outputs", {}).get(key, default))


def iter_batches(x_pos: np.ndarray, x_neg: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(len(x_pos))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x_pos[batch_idx], x_neg[batch_idx]


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
    seed = int(config["seed"])
    set_seed(seed)
    jt.set_global_seed(seed)

    text_cfg = config["text"]
    vocab = load_or_build_vocab(config["data"]["train_path"], text_cfg["vocab_path"], int(text_cfg["max_vocab_size"]))
    max_len = int(text_cfg["max_len"])
    x_pos, x_neg = load_pairwise_text_data(config["data"]["train_path"], vocab, max_len)
    valid_rows = load_text_candidate_data(config["data"]["valid_path"], vocab, max_len)

    model = TextCNNScorer(
        vocab_size=len(vocab),
        embed_dim=int(config["model"]["embed_dim"]),
        num_filters=int(config["model"]["num_filters"]),
        kernel_sizes=list(config["model"]["kernel_sizes"]),
        dropout=float(config["model"]["dropout"]),
    )
    optimizer = jt.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    batch_size = int(config["train"]["batch_size"])
    rng = np.random.default_rng(seed)
    logger = setup_logger(output_path(config, "jittor_log", "logs/msmarco_textcnn_jittor_train.log"), "textcnn_jittor_train")

    logger.info("Training Jittor TextCNN on %d pairwise samples; vocab=%d", len(x_pos), len(vocab))
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        losses = []
        for batch_pos, batch_neg in iter_batches(x_pos, x_neg, batch_size, rng):
            score_pos = model(jt.array(batch_pos.astype("int64")))
            score_neg = model(jt.array(batch_neg.astype("int64")))
            loss = pairwise_ranking_loss(score_pos, score_neg)
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.numpy()).reshape(-1)[0]))

        valid_metrics = evaluate_grouped(score_rows(model, valid_rows, jt), topk=list(config["eval"]["topk"]))
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

    model_path = ensure_parent(output_path(config, "jittor_model", "outputs/msmarco_textcnn_jittor_model.pkl"))
    jt.save(model.state_dict(), str(model_path))
    logger.info("Saved model to %s", resolve_path(model_path))


if __name__ == "__main__":
    main()
