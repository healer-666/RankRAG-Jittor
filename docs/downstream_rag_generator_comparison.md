# Downstream RAG Generator Comparison

## Experiment Question

This diagnostic checks whether scaling the fixed answer generator from Qwen2.5-1.5B-Instruct to Qwen2.5-7B-Instruct can more reliably convert stronger reranking into better final short answers.

The Qwen2.5-1.5B strict-prompt ablation is reported separately in [downstream_rag_prompt_ablation_1_5b.md](downstream_rag_prompt_ablation_1_5b.md). That is a downstream prompt ablation / generation-format sensitivity check and does not include a Qwen2.5-7B strict-prompt run.

The protocol keeps the following factors fixed:

- the same 50 MS MARCO questions;
- the same BM25, LoRA v3, and Cross-Encoder ranking files;
- the same top-3 contexts per method;
- the same prompt template;
- the same generation settings: `do_sample=false`, `temperature=0.0`, `max_context_tokens=2048`, `max_new_tokens=64`.

The main changed factor is the generator: Qwen2.5-1.5B-Instruct versus Qwen2.5-7B-Instruct. Because the rankings and top-3 contexts are unchanged, Gold@3 should stay unchanged for each reranker.

## Core Results

| Reranker | Generator | Gold@3 | Answer Hit | EM | Token F1 | Retrieval Fail | Generation Fail | Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | Qwen2.5-1.5B | 0.68 | 0.22 | 0.00 | 0.1646 | 16 | 23 | 11 |
| BM25 | Qwen2.5-7B | 0.68 | 0.32 | 0.00 | 0.1641 | 16 | 18 | 16 |
| LoRA v3 | Qwen2.5-1.5B | 0.72 | 0.28 | 0.04 | 0.2108 | 14 | 22 | 14 |
| LoRA v3 | Qwen2.5-7B | 0.72 | 0.24 | 0.00 | 0.1569 | 14 | 24 | 12 |
| Cross-Encoder | Qwen2.5-1.5B | 0.88 | 0.28 | 0.00 | 0.2058 | 6 | 30 | 14 |
| Cross-Encoder | Qwen2.5-7B | 0.88 | 0.30 | 0.00 | 0.1635 | 6 | 29 | 15 |

## Refusal Phrase Diagnostics

The legacy `insufficient_information_rate` is preserved as an exact-match metric. The updated aggregator also reports three explicit sample-level diagnostics:

- `exact_insufficient_information_rate`: normalized answer, after trimming spaces and trailing ` .:`, equals `insufficient information`.
- `starts_with_insufficient_information_rate`: normalized answer starts with `insufficient information`.
- `contains_insufficient_information_rate`: normalized answer contains `insufficient information` anywhere.

| Generator | Reranker | Contains | Starts With | Exact |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-1.5B | BM25 | 6/50 = 12% | 2/50 = 4% | 2/50 = 4% |
| Qwen2.5-1.5B | LoRA v3 | 2/50 = 4% | 2/50 = 4% | 2/50 = 4% |
| Qwen2.5-1.5B | Cross-Encoder | 3/50 = 6% | 1/50 = 2% | 1/50 = 2% |
| Qwen2.5-7B | BM25 | 43/50 = 86% | 16/50 = 32% | 0/50 = 0% |
| Qwen2.5-7B | LoRA v3 | 41/50 = 82% | 16/50 = 32% | 0/50 = 0% |
| Qwen2.5-7B | Cross-Encoder | 48/50 = 96% | 18/50 = 36% | 0/50 = 0% |

## Observations

Qwen2.5-7B-Instruct does not steadily outperform Qwen2.5-1.5B-Instruct in this fixed 50-question protocol.

BM25 improves in Answer Hit from 0.22 to 0.32, and generation failures decrease from 23 to 18. However, LoRA v3 drops from 0.28 to 0.24 in Answer Hit, and its Token F1 drops from 0.2108 to 0.1569. Cross-Encoder obtains only a small Answer Hit gain from 0.28 to 0.30, while Token F1 drops from 0.2058 to 0.1635.

Cross-Encoder still provides the strongest evidence availability: Gold@3 is 0.88 and retrieval failures fall to 6. The 7B generator nevertheless reaches only 0.30 Answer Hit with Cross-Encoder contexts, which shows that stronger evidence coverage alone does not automatically solve the evidence-use and answer-formatting bottleneck.

The 7B outputs include the phrase `Insufficient information` in 82%-96% of samples, much more often than the 1.5B outputs. A common pattern is to give an answer, append `Insufficient information`, and then provide a longer explanation. This conservative and verbose behavior hurts short-answer metrics such as Exact Match and Token F1.

The result should not be interpreted as "7B is worse" in general. A more precise interpretation is that the current prompt and open-form generation format are not fully aligned with short-answer MS MARCO automatic evaluation. Model scale, prompt constraints, conflicting evidence handling, output formatting, and post-processing jointly determine final answer quality.

## Conclusion Boundary

In the current fixed 50-question protocol, with identical top-3 contexts and the same prompt, Qwen2.5-7B-Instruct did not steadily outperform Qwen2.5-1.5B-Instruct. Scaling the general generator alone is not sufficient to reliably convert better retrieval coverage into higher short-answer quality.

## Relation to RankRAG

This is a downstream diagnostic extension, not a same-protocol reproduction of the original RankRAG answer-generation experiment. The original RankRAG work uses a jointly trained ranking and generation model. This repository uses fixed rerankers plus a general Qwen Instruct generator, so the experiment should be used to analyze the retrieval-to-generation interface in this lightweight reproduction system rather than to compare final QA metrics directly with the RankRAG paper.
