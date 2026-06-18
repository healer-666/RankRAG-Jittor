"""Token-id dataset helpers for TextCNN reranking."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from dataset import candidate_feature_text, read_jsonl
from utils import ensure_parent, resolve_path


PAD = "<pad>"
UNK = "<unk>"
SEP = "<sep>"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(records: list[dict], max_vocab_size: int = 30000) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(tokenize(record["query"]))
        for candidate in record["candidates"]:
            counter.update(tokenize(candidate_feature_text(candidate)))

    vocab = {PAD: 0, UNK: 1, SEP: 2}
    for token, _ in counter.most_common(max(0, max_vocab_size - len(vocab))):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    ensure_parent(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def load_or_build_vocab(train_path: str | Path, vocab_path: str | Path, max_vocab_size: int) -> dict[str, int]:
    resolved = resolve_path(vocab_path)
    if resolved.exists():
        return load_vocab(resolved)
    records = read_jsonl(train_path)
    vocab = build_vocab(records, max_vocab_size=max_vocab_size)
    save_vocab(vocab, resolved)
    return vocab


def encode_pair(query: str, candidate_text: str, vocab: dict[str, int], max_len: int) -> np.ndarray:
    tokens = tokenize(query) + [SEP] + tokenize(candidate_text)
    unk = vocab[UNK]
    ids = [vocab.get(token, unk) for token in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab[PAD]] * (max_len - len(ids)))
    return np.asarray(ids, dtype=np.int64)


def make_pairwise_text_samples(records: list[dict], vocab: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    pos_inputs: list[np.ndarray] = []
    neg_inputs: list[np.ndarray] = []
    for record in records:
        positives = [candidate for candidate in record["candidates"] if int(candidate["label"]) == 1]
        negatives = [candidate for candidate in record["candidates"] if int(candidate["label"]) == 0]
        for positive in positives:
            pos_ids = encode_pair(record["query"], candidate_feature_text(positive), vocab, max_len)
            for negative in negatives:
                pos_inputs.append(pos_ids)
                neg_inputs.append(encode_pair(record["query"], candidate_feature_text(negative), vocab, max_len))

    if not pos_inputs:
        raise ValueError("No pairwise TextCNN samples were created; check dataset labels.")
    return np.stack(pos_inputs).astype(np.int64), np.stack(neg_inputs).astype(np.int64)


def make_eval_text_rows(records: list[dict], vocab: dict[str, int], max_len: int) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        for candidate in record["candidates"]:
            rows.append(
                {
                    "query_id": record["query_id"],
                    "query": record["query"],
                    "doc_id": candidate["doc_id"],
                    "title": candidate.get("title", ""),
                    "text": candidate["text"],
                    "label": int(candidate["label"]),
                    "input_ids": encode_pair(record["query"], candidate_feature_text(candidate), vocab, max_len),
                }
            )
    return rows


def load_pairwise_text_data(path: str | Path, vocab: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    return make_pairwise_text_samples(read_jsonl(path), vocab, max_len=max_len)


def load_text_candidate_data(path: str | Path, vocab: dict[str, int], max_len: int) -> list[dict]:
    return make_eval_text_rows(read_jsonl(path), vocab, max_len=max_len)


def load_text_demo_candidates(path: str | Path, vocab: dict[str, int], max_len: int) -> list[dict]:
    demo = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    return make_eval_text_rows([demo], vocab, max_len=max_len)
