"""Utilities for the LoRA reranker debug pipeline."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from src.metrics import evaluate_grouped
except ModuleNotFoundError:  # direct script execution from repository root
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.metrics import evaluate_grouped


ANSWER_LABELS = ("Relevant", "Irrelevant")


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(query: str, passage: str) -> str:
    return (
        "Instruction:\n"
        "Determine whether the passage is relevant to the question. "
        "Answer with exactly one word: Relevant or Irrelevant.\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Passage:\n"
        f"{passage}\n\n"
        "Answer:\n"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _answer_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer {answer!r} produced no tokens.")
    return ids


@torch.no_grad()
def compute_label_logprob(model, tokenizer, prompt: str, answer: str, *, max_length: int, device: torch.device) -> float:
    """Compute conditional log P(answer | prompt) under a causal LM."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = _answer_token_ids(tokenizer, answer)
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        keep_prompt = max(1, max_length - len(answer_ids))
        prompt_ids = prompt_ids[:keep_prompt]
        input_ids = prompt_ids + answer_ids

    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=tensor)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    score = 0.0
    for offset, token_id in enumerate(answer_ids):
        token_position = prompt_len + offset
        prediction_position = token_position - 1
        if prediction_position < 0:
            continue
        score += float(log_probs[0, prediction_position, token_id].detach().cpu())
    return score


def compute_ranking_metrics(rows: list[dict], topk: list[int] | None = None) -> dict[str, float]:
    topk = topk or [1, 3, 5, 10]
    metrics = evaluate_grouped(rows, topk)
    metrics["num_queries"] = len({row["query_id"] for row in rows})
    metrics["num_pairs"] = len(rows)
    return metrics


def save_markdown_metrics(metrics: dict, path: str | Path, *, title: str = "LoRA reranker debug metrics") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "pairwise_accuracy",
        "num_queries",
        "num_pairs",
        "device",
        "inference_time_sec",
        "pairs_per_second",
    ]
    lines = [f"# {title}", "", "| Metric | Value |", "| --- | ---: |"]
    for key in preferred:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            value_text = f"{value:.6f}" if math.isfinite(value) else str(value)
        else:
            value_text = str(value)
        lines.append(f"| {key} | {value_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_loss_curve(log_rows: list[dict], output_path: str | Path) -> None:
    if not log_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in log_rows if "loss" in row]
    losses = [row["loss"] for row in log_rows if "loss" in row]
    if not steps:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(steps, losses, marker="o", linewidth=1.8, color="#0F4D92")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("LoRA debug training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def truncate_text(text: str, max_chars: int = 1200) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
