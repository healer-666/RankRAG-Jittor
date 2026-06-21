import pytest

from src.rag_answer_generation import (
    ContextBuilder,
    PROMPT_TEMPLATE,
    build_prompt,
    build_qa_prompt,
    prompt_version,
)


def test_missing_prompt_style_defaults_to_original_prompt():
    question = "Where was Ada born?"
    contexts = ["Ada was born in London.", "Ada wrote about computing."]

    assert build_qa_prompt(question, contexts) == build_prompt(question, contexts, "original")
    assert build_qa_prompt(question, contexts) == PROMPT_TEMPLATE.format(
        question=question,
        contexts="[1] Ada was born in London.\n[2] Ada wrote about computing.",
    )


def test_strict_short_answer_prompt_contains_all_rules():
    prompt = build_qa_prompt("Who wrote Hamlet?", ["Hamlet was written by William Shakespeare."], "strict_short_answer")

    expected_rules = [
        "1. Output only the final answer on one line.",
        "2. Do not provide explanations, reasoning, notes, or citations.",
        "3. Use the shortest answer span that directly answers the question.",
        "4. If at least one context directly answers the question, use that answer even if other contexts are irrelevant, incomplete, or approximate.",
        "5. If contexts conflict, choose the most specific context that directly matches the entities and intent of the question.",
        '6. Output exactly "Insufficient information" only when none of the contexts contains enough information to answer.',
        '7. Never append "Insufficient information" after giving an answer.',
    ]
    for rule in expected_rules:
        assert rule in prompt


def test_strict_short_answer_prompt_has_one_question_marker_and_contexts():
    prompt = build_qa_prompt(
        "What is the capital of France?",
        ["Paris is the capital of France.", "France is in Europe."],
        "strict_short_answer",
    )

    assert prompt.count("Question:") == 1
    assert "\nContexts:\n[1] Paris is the capital of France.\n[2] France is in Europe.\n\nAnswer:\n" in prompt


def test_unknown_prompt_style_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown prompt_style"):
        build_qa_prompt("Question?", ["Context."], "verbose")


def test_prompt_versions_are_stable():
    assert prompt_version("original") == "original_v1"
    assert prompt_version("strict_short_answer") == "strict_short_answer_v1"


def test_context_builder_records_prompt_metadata():
    qa_rows = [
        {
            "query_id": "q1",
            "question": "Who wrote Hamlet?",
            "gold_answers": ["William Shakespeare"],
            "candidates": [
                {
                    "candidate_id": "d1",
                    "passage": "Hamlet was written by William Shakespeare.",
                    "label": 1,
                }
            ],
        }
    ]
    rankings = {"q1": [{"candidate_id": "d1", "rank": 1, "score": 1.0}]}

    row = ContextBuilder(qa_rows, top_k=1, prompt_style="strict_short_answer").build("bm25", rankings)[0]

    assert row["prompt_style"] == "strict_short_answer"
    assert row["prompt_version"] == "strict_short_answer_v1"
    assert row["prompt"].startswith("You are a short-answer question answering system.")
