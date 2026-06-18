"""Dataset and feature construction for query-candidate context ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from utils import resolve_path


EMBED_DIM = 384
INPUT_DIM = EMBED_DIM * 4 + 1


def make_vectorizer(n_features: int = EMBED_DIM) -> HashingVectorizer:
    """Use a stateless vectorizer so train/eval backends share identical features."""

    return HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )


def read_jsonl(path: str | Path) -> list[dict]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def vectorize_text(vectorizer: HashingVectorizer, text: str) -> np.ndarray:
    return vectorizer.transform([text]).toarray().astype(np.float32)[0]


def build_feature(query: str, candidate_text: str, vectorizer: HashingVectorizer | None = None) -> np.ndarray:
    """Build q/c interaction features: q, c, |q-c|, q*c, cosine."""

    vectorizer = vectorizer or make_vectorizer()
    q_emb = vectorize_text(vectorizer, query)
    c_emb = vectorize_text(vectorizer, candidate_text)
    cosine = float(np.dot(q_emb, c_emb) / ((np.linalg.norm(q_emb) * np.linalg.norm(c_emb)) + 1e-8))
    feature = np.concatenate([q_emb, c_emb, np.abs(q_emb - c_emb), q_emb * c_emb, [cosine]]).astype(np.float32)
    return feature


def make_pairwise_samples(records: list[dict], max_pairs: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Create feature_pos, feature_neg samples for pairwise ranking loss."""

    vectorizer = make_vectorizer()
    pos_features: list[np.ndarray] = []
    neg_features: list[np.ndarray] = []

    for record in records:
        positives = [cand for cand in record["candidates"] if int(cand["label"]) == 1]
        negatives = [cand for cand in record["candidates"] if int(cand["label"]) == 0]
        for pos in positives:
            pos_text = candidate_feature_text(pos)
            pos_feature = build_feature(record["query"], pos_text, vectorizer)
            for neg in negatives:
                pos_features.append(pos_feature)
                neg_features.append(build_feature(record["query"], candidate_feature_text(neg), vectorizer))
                if max_pairs is not None and len(pos_features) >= max_pairs:
                    return np.stack(pos_features), np.stack(neg_features)

    if not pos_features:
        raise ValueError("No pairwise samples were created; check labels in the dataset.")
    return np.stack(pos_features), np.stack(neg_features)


def make_candidate_table(records: list[dict]) -> list[dict]:
    """Flatten query-candidate records for scoring and grouped metric evaluation."""

    vectorizer = make_vectorizer()
    rows: list[dict] = []
    for record in records:
        for candidate in record["candidates"]:
            feature_text = candidate_feature_text(candidate)
            rows.append(
                {
                    "query_id": record["query_id"],
                    "query": record["query"],
                    "doc_id": candidate["doc_id"],
                    "title": candidate.get("title", ""),
                    "text": candidate["text"],
                    "label": int(candidate["label"]),
                    "feature": build_feature(record["query"], feature_text, vectorizer),
                }
            )
    return rows


def candidate_feature_text(candidate: dict) -> str:
    """Use title plus body text when a title is available, matching academic search inputs."""

    title = candidate.get("title", "")
    if title:
        return f"{title}. {candidate['text']}"
    return candidate["text"]


def load_pairwise_data(path: str | Path, max_pairs: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    return make_pairwise_samples(read_jsonl(path), max_pairs=max_pairs)


def load_candidate_data(path: str | Path) -> list[dict]:
    return make_candidate_table(read_jsonl(path))


def load_demo_candidates(path: str | Path) -> list[dict]:
    demo = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    return make_candidate_table([demo])
