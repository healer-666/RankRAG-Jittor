"""Prompt construction and label parsing for zero-shot relevance scoring."""

from __future__ import annotations

import re


DEFAULT_PROMPT_TEMPLATE = """Given a search query and a candidate passage, determine whether the passage is relevant to the query.

Query:
{query}

Passage:
{passage}

Answer with exactly one label:
Relevant or Irrelevant."""


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").replace("\n", " ").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def build_prompt(query: str, passage: str, max_length: int = 512, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    # max_length is treated as a conservative character budget because the
    # JittorLLM tokenizer interface is not guaranteed to be available.
    query = truncate_text(query, max(64, max_length // 4))
    passage = truncate_text(passage, max(128, max_length))
    return template.format(query=query, passage=passage)


def parse_relevance_label(text: str) -> tuple[float, bool, str]:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[^a-z]+", " ", normalized).strip()
    first = normalized.split(" ", 1)[0] if normalized else ""
    if first == "relevant":
        return 1.0, False, "generated_label"
    if first == "irrelevant":
        return 0.0, False, "generated_label"
    if "irrelevant" in normalized and "relevant" not in normalized.replace("irrelevant", ""):
        return 0.0, False, "generated_label"
    if "relevant" in normalized:
        return 1.0, False, "generated_label"
    return 0.5, True, "unparsed_generated_label"
