"""Utilities for downstream RAG answer generation evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from src.evaluate_answer_quality import answer_metrics, classify_failure, gold_in_context
    from src.utils import resolve_path
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.evaluate_answer_quality import answer_metrics, classify_failure, gold_in_context
    from src.utils import resolve_path


PROMPT_TEMPLATE = """Answer the question using only the provided contexts.

Give a short and direct answer.
Do not add information that is not supported by the contexts.
If the contexts do not contain enough information, answer exactly:
Insufficient information

Question:
{question}

Contexts:
{contexts}

Answer:
"""

STRICT_SHORT_ANSWER_PROMPT_TEMPLATE = """You are a short-answer question answering system.

Answer the question using only the provided contexts.

Rules:
1. Output only the final answer on one line.
2. Do not provide explanations, reasoning, notes, or citations.
3. Use the shortest answer span that directly answers the question.
4. If at least one context directly answers the question, use that answer even if other contexts are irrelevant, incomplete, or approximate.
5. If contexts conflict, choose the most specific context that directly matches the entities and intent of the question.
6. Output exactly "Insufficient information" only when none of the contexts contains enough information to answer.
7. Never append "Insufficient information" after giving an answer.

Question:
{question}

Contexts:
{contexts}

Answer:
"""

PROMPT_VERSIONS = {
    "original": "original_v1",
    "strict_short_answer": "strict_short_answer_v1",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def format_contexts(contexts: list[str]) -> str:
    return "\n".join(f"[{idx}] {text}" for idx, text in enumerate(contexts, start=1))


def prompt_version(prompt_style: str) -> str:
    if prompt_style not in PROMPT_VERSIONS:
        supported = ", ".join(sorted(PROMPT_VERSIONS))
        raise ValueError(f"Unknown prompt_style '{prompt_style}'. Supported values: {supported}")
    return PROMPT_VERSIONS[prompt_style]


def build_prompt(question: str, contexts: list[str], prompt_style: str = "original") -> str:
    context_text = format_contexts(contexts)
    if prompt_style == "original":
        return PROMPT_TEMPLATE.format(question=question, contexts=context_text)
    if prompt_style == "strict_short_answer":
        return STRICT_SHORT_ANSWER_PROMPT_TEMPLATE.format(question=question, contexts=context_text)
    supported = ", ".join(sorted(PROMPT_VERSIONS))
    raise ValueError(f"Unknown prompt_style '{prompt_style}'. Supported values: {supported}")


def build_qa_prompt(question: str, contexts: list[str], prompt_style: str = "original") -> str:
    return build_prompt(question, contexts, prompt_style)


def truncate_contexts_by_words(contexts: list[str], max_words: int) -> list[str]:
    if max_words <= 0:
        return contexts
    remaining = max_words
    truncated = []
    for context in contexts:
        words = context.split()
        if remaining <= 0:
            truncated.append("")
            continue
        kept = words[:remaining]
        truncated.append(" ".join(kept))
        remaining -= len(kept)
    return truncated


def candidate_id(item: dict[str, Any]) -> str:
    for key in ["doc_id", "candidate_id", "passage_id"]:
        if key in item:
            return str(item[key])
    raise KeyError(f"Cannot find candidate id in keys: {sorted(item)}")


def candidate_text(item: dict[str, Any]) -> str:
    for key in ["text", "passage", "content"]:
        if item.get(key):
            return str(item[key])
    return ""


class RankingsLoader:
    def __init__(self, candidate_pool: dict[str, dict[str, dict[str, Any]]]):
        self.candidate_pool = candidate_pool

    def load_rankings(self, method_name: str, rankings_path: str, nested_key: str | None = None) -> dict[str, list[dict[str, Any]]]:
        payload = json.loads(resolve_path(rankings_path).read_text(encoding="utf-8"))
        if nested_key:
            payload = payload[nested_key]
        if not isinstance(payload, list):
            raise ValueError(f"{method_name} rankings must resolve to a list, got {type(payload).__name__}")

        result: dict[str, list[dict[str, Any]]] = {}
        for row in payload:
            query_id = str(row["query_id"])
            if query_id not in self.candidate_pool:
                continue
            ranking = row.get("ranking") or row.get("candidates")
            if ranking is None:
                raise ValueError(f"{method_name}/{query_id} has no ranking or candidates field")
            seen = set()
            normalized = []
            for index, item in enumerate(ranking, start=1):
                cid = candidate_id(item)
                if cid in seen:
                    raise ValueError(f"{method_name}/{query_id} has duplicate candidate {cid}")
                if cid not in self.candidate_pool.get(query_id, {}):
                    raise ValueError(f"{method_name}/{query_id} candidate {cid} is not in candidate pool")
                seen.add(cid)
                normalized.append(
                    {
                        "candidate_id": cid,
                        "score": float(item.get("score", 0.0)),
                        "rank": int(item.get("rank", index)),
                    }
                )
            normalized.sort(key=lambda item: item["rank"])
            result[query_id] = normalized
        return result


class ContextBuilder:
    def __init__(
        self,
        qa_rows: list[dict[str, Any]],
        top_k: int,
        max_context_tokens: int = 2048,
        prompt_style: str = "original",
    ):
        self.qa_rows = qa_rows
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.prompt_style = prompt_style
        self.prompt_version = prompt_version(prompt_style)
        self.candidate_pool = {
            row["query_id"]: {candidate["candidate_id"]: candidate for candidate in row["candidates"]}
            for row in qa_rows
        }

    def build(self, method: str, rankings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        missing = sorted({row["query_id"] for row in self.qa_rows} - set(rankings))
        if missing:
            raise ValueError(f"{method} missing {len(missing)} QA query ids; first={missing[:3]}")

        contexts = []
        for row in self.qa_rows:
            query_id = row["query_id"]
            ranked = rankings[query_id][: self.top_k]
            context_ids = [item["candidate_id"] for item in ranked]
            candidates = [self.candidate_pool[query_id][cid] for cid in context_ids]
            context_texts = truncate_contexts_by_words(
                [candidate["passage"] for candidate in candidates],
                self.max_context_tokens,
            )
            contains_positive = any(int(candidate.get("label", 0)) == 1 for candidate in candidates)
            gold_context = gold_in_context(context_texts, row["gold_answers"])
            labeled_positive_rank = next(
                (idx for idx, candidate in enumerate(candidates, start=1) if int(candidate.get("label", 0)) == 1),
                None,
            )
            gold_answer_context_rank = next(
                (
                    idx
                    for idx, context in enumerate(context_texts, start=1)
                    if gold_in_context([context], row["gold_answers"])
                ),
                None,
            )
            prompt = build_prompt(row["question"], context_texts, self.prompt_style)
            contexts.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "question": row["question"],
                    "gold_answers": row["gold_answers"],
                    "top_k": self.top_k,
                    "context_candidate_ids": context_ids,
                    "contexts": context_texts,
                    "contains_labeled_positive": contains_positive,
                    "gold_answer_in_context": gold_context,
                    "labeled_positive_context_rank": labeled_positive_rank,
                    "gold_answer_context_rank": gold_answer_context_rank,
                    "prompt_style": self.prompt_style,
                    "prompt_version": self.prompt_version,
                    "prompt_hash": prompt_hash(prompt),
                    "prompt": prompt,
                }
            )
        return contexts


class TransformersGenerator:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        random.seed(int(config.get("seed", 42)))
        torch.manual_seed(int(config.get("seed", 42)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.get("seed", 42)))

        model_name = (
            config.get("runtime_generator_model_path")
            or os.environ.get("QWEN_GENERATOR_MODEL_PATH")
            or config.get("generator_model_path")
            or config["generator_model"]
        )
        self.model_id = config["generator_model"]
        self.used_local_path = bool(
            config.get("runtime_generator_model_path")
            or os.environ.get("QWEN_GENERATOR_MODEL_PATH")
            or config.get("generator_model_path")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        load_start = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise with actionable context.
            raise RuntimeError(
                f"Cannot load generator model '{model_name}' from local files. "
                "Set generator_model_path to a local Qwen2.5-1.5B-Instruct directory or populate the HF cache."
            ) from exc
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        self.load_time_sec = time.time() - load_start
        self.max_new_tokens = int(config.get("max_new_tokens", 64))
        self.max_context_tokens = int(config.get("max_context_tokens", 2048))
        self.do_sample = bool(config.get("do_sample", False))
        self.temperature = float(config.get("temperature", 0.0))
        self.dtype = str(dtype)
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    def generate(self, prompt: str) -> tuple[str, float]:
        start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
        except self.torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA OOM during downstream RAG generation. Try top_k=3 and shorter contexts.") from exc
        generated = output[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, time.time() - start


def evaluate_generated_row(context_row: dict[str, Any], generated_answer: str, runtime_sec: float) -> dict[str, Any]:
    metrics = answer_metrics(generated_answer, context_row["gold_answers"])
    failure_type = classify_failure(
        gold_answer_in_context=bool(context_row["gold_answer_in_context"]),
        answer_hit=int(metrics["answer_hit"]),
    )
    result = {
        key: value
        for key, value in context_row.items()
        if key
        not in {
            "prompt",
        }
    }
    result.update(metrics)
    result.update(
        {
            "generated_answer": generated_answer,
            "failure_type": failure_type,
            "generation_runtime_sec": runtime_sec,
        }
    )
    return result


def load_existing_results(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {(row["method"], row["query_id"], int(row["top_k"])): row for row in rows}
