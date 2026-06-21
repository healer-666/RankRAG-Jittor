"""Answer normalization and QA metrics for downstream RAG evaluation."""

from __future__ import annotations

import re
import string
from collections import Counter


ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1_single(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_single(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def contains_gold(prediction: str, gold_answers: list[str]) -> int:
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0
    for gold in gold_answers:
        normalized_gold = normalize_answer(gold)
        if normalized_gold and normalized_gold in normalized_prediction:
            return 1
    return 0


def answer_metrics(prediction: str, gold_answers: list[str]) -> dict[str, float | int]:
    if not gold_answers:
        return {"exact_match": 0, "token_f1": 0.0, "gold_containment": 0, "answer_hit": 0}
    exact = max(exact_match_single(prediction, gold) for gold in gold_answers)
    f1 = max(token_f1_single(prediction, gold) for gold in gold_answers)
    containment = contains_gold(prediction, gold_answers)
    return {
        "exact_match": exact,
        "token_f1": f1,
        "gold_containment": containment,
        "answer_hit": int(bool(exact or containment)),
    }


def gold_in_context(contexts: list[str], gold_answers: list[str]) -> bool:
    joined = "\n".join(contexts)
    normalized_context = normalize_answer(joined)
    for gold in gold_answers:
        normalized_gold = normalize_answer(gold)
        if normalized_gold and normalized_gold in normalized_context:
            return True
    return False


def classify_failure(
    *,
    gold_answer_in_context: bool,
    answer_hit: int,
) -> str:
    if not gold_answer_in_context:
        return "retrieval_failure"
    if answer_hit:
        return "success"
    return "generation_failure"
