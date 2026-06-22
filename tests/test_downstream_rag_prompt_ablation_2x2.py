from scripts.build_downstream_rag_prompt_ablation_2x2 import METHODS, build_payload


def test_prompt_ablation_2x2_loads_all_formal_results():
    payload = build_payload()

    assert set(payload["results"]) == {
        "qwen2_1_5b_original",
        "qwen2_1_5b_strict",
        "qwen2_7b_original",
        "qwen2_7b_strict",
    }
    assert sum(len(row["methods"]) for row in payload["results"].values()) == 12


def test_prompt_ablation_2x2_gold_at_3_is_fixed_by_reranker():
    payload = build_payload()

    for method in METHODS:
        gold_values = {
            row["methods"][method]["gold_in_context"]
            for row in payload["results"].values()
        }
        assert len(gold_values) == 1


def test_prompt_ablation_2x2_prompt_labels_are_explicit():
    payload = build_payload()

    assert payload["results"]["qwen2_1_5b_original"]["prompt_style"] == "original"
    assert payload["results"]["qwen2_7b_original"]["prompt_style"] == "original"
    assert payload["results"]["qwen2_1_5b_strict"]["prompt_style"] == "strict_short_answer_v1"
    assert payload["results"]["qwen2_7b_strict"]["prompt_style"] == "strict_short_answer_v1"


def test_prompt_ablation_2x2_requires_protocol_checks():
    payload = build_payload()

    assert payload["protocol"]["protocol_checks_passed"] is True
    assert payload["protocol"]["same_rankings"] is True
    assert payload["protocol"]["same_questions"] is True
    assert payload["protocol"]["same_contexts"] is True
    assert payload["protocol"]["same_generation_settings_except_prompt"] is True
