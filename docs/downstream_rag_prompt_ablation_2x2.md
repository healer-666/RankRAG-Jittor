# Downstream RAG Generator-Prompt 2x2 Ablation

## 1. Experiment Questions

This extension experiment asks two questions:

- Does the strict short-answer prompt improve generation quality?
- Does the prompt effect change with generator scale?

## 2. Fixed Protocol

The four formal runs use the same 50 MS MARCO questions, the same BM25, LoRA v3, and Cross-Encoder rerankers, the same ranking files, the same top-3 contexts, the same context order, the same decoding settings, and the same automatic metrics.

Prompt comparisons only change the prompt. Generator-scale comparisons only change the generator. Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct were run on different hardware, but the committed results are deterministic offline generation outputs; hardware affects runtime, not the protocol semantics.

This is a downstream prompt ablation, generator-scale sensitivity check, generation-format sensitivity check, and extension experiment. It is not a reranking leaderboard result.

## 3. Complete 2x2 Results

| Reranker | Generator | Prompt | Gold@3 | Answer Hit | EM | Token F1 | Avg Len | Gen Fail | Success | Contains Refusal |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 1.5B | Original | 0.68 | 0.22 | 0.00 | 0.1646 | 38.40 | 23 | 11 | 0.12 |
| BM25 | 1.5B | Strict | 0.68 | 0.32 | 0.06 | 0.2684 | 32.08 | 18 | 16 | 0.00 |
| BM25 | 7B | Original | 0.68 | 0.32 | 0.00 | 0.1641 | 42.06 | 18 | 16 | 0.86 |
| BM25 | 7B | Strict | 0.68 | 0.34 | 0.02 | 0.2072 | 30.28 | 17 | 17 | 0.60 |
| LoRA v3 | 1.5B | Original | 0.72 | 0.28 | 0.04 | 0.2108 | 37.94 | 22 | 14 | 0.04 |
| LoRA v3 | 1.5B | Strict | 0.72 | 0.24 | 0.10 | 0.3335 | 24.72 | 24 | 12 | 0.02 |
| LoRA v3 | 7B | Original | 0.72 | 0.24 | 0.00 | 0.1569 | 44.48 | 24 | 12 | 0.82 |
| LoRA v3 | 7B | Strict | 0.72 | 0.34 | 0.02 | 0.2078 | 30.14 | 19 | 17 | 0.62 |
| Cross-Encoder | 1.5B | Original | 0.88 | 0.28 | 0.00 | 0.2058 | 38.98 | 30 | 14 | 0.06 |
| Cross-Encoder | 1.5B | Strict | 0.88 | 0.32 | 0.10 | 0.3249 | 28.98 | 28 | 16 | 0.00 |
| Cross-Encoder | 7B | Original | 0.88 | 0.30 | 0.00 | 0.1635 | 45.24 | 29 | 15 | 0.96 |
| Cross-Encoder | 7B | Strict | 0.88 | 0.40 | 0.00 | 0.1800 | 29.20 | 24 | 20 | 0.62 |

## 4. Main Observations

Prompt choice clearly affects final metrics, so the original-prompt results alone should not be used to judge generator capability.

For Qwen2.5-1.5B-Instruct, the strict prompt provides stronger format control: Token F1 improves for all three rerankers, EM improves, average answer length drops, and contains-refusal rates are reduced to near zero.

For Qwen2.5-7B-Instruct, the strict prompt also helps answer containment: BM25 Answer Hit changes from 0.32 to 0.34, LoRA v3 from 0.24 to 0.34, and Cross-Encoder from 0.30 to 0.40. Generation failures decrease for all three rerankers, and average answer length decreases substantially.

The 7B strict run still has visible output-format problems. Across rerankers, 60% to 62% of answers contain `Insufficient information`, and sampled outputs show repeated answers, multiple `Answer:` continuations, or a refusal phrase appended after a correct answer.

Under the strict prompt, 7B has higher Answer Hit than 1.5B: BM25 0.34 vs 0.32, LoRA v3 0.34 vs 0.24, and Cross-Encoder 0.40 vs 0.32.

However, 1.5B strict obtains better EM and Token F1. This suggests that 7B more often includes a correct answer somewhere in the output, while 1.5B more reliably follows the short-answer format. Prompt effects are therefore model-dependent, and natural-language prompting alone does not fully guarantee strict output formatting.

Higher Gold@3 still does not fully convert into final answer quality: Cross-Encoder reaches Gold@3 = 0.88, while the best Answer Hit in the 2x2 table is 0.40.

## 5. Conclusion Boundary

Within the current fixed 50-question protocol, the strict short-answer prompt has measurable effects for both generators, but the effect is strongly model-dependent. Qwen2.5-1.5B-Instruct reaches higher EM and Token F1 under the strict prompt, while Qwen2.5-7B-Instruct more often includes the correct answer in its output but still frequently generates repeated answers or appended refusal phrases. Model scale, prompt design, and output-format control jointly determine the downstream RAG automatic evaluation results.

These results should not be simplified to `1.5B is stronger than 7B` or `7B is stronger than 1.5B`. Answer Hit measures whether a gold answer appears in the generated text, while EM and Token F1 are more sensitive to formatting and extra text.

## 6. Relation to RankRAG

This is a follow-up extension experiment, not a same-protocol reproduction of the original RankRAG final QA setup. The current system uses fixed rerankers plus a general Qwen Instruct generator. RankRAG jointly trains ranking and generation. This experiment mainly analyzes the retrieval-generation interface, prompt sensitivity, and output-format failure modes in this reproduction system.

## 7. Source Artifacts

- `outputs/downstream_rag_prompt_ablation_2x2.json`
- `outputs/downstream_rag_eval/downstream_rag_eval_results.json`
- `outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt/downstream_rag_eval_results.json`
- `outputs/downstream_rag_eval_qwen2_7b/downstream_rag_eval_results.json`
- `outputs/downstream_rag_eval_qwen2_7b_strict_prompt/downstream_rag_eval_results.json`
