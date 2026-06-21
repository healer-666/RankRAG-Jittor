# Qwen2.5-1.5B Downstream RAG Prompt Ablation

## 1. Research Question

This extension experiment asks whether a strict short-answer prompt changes answer quality when the generator, rankings, top-3 contexts, context order, question set, decoding settings, and evaluation metrics are fixed.

The only intended main variable is the prompt style. The protocol consistency check confirms whether the fixed inputs match before interpreting the results.

## 2. Prompt Comparison

The original prompt asks for a short, direct answer and uses `Insufficient information` when the contexts are insufficient.

The strict prompt adds explicit output-format constraints:

- output only the final answer
- one line
- no explanation, reasoning, notes, or citations
- shortest answer span
- use a directly answering context rather than refusing when at least one exists
- resolve conflicts by choosing the most specific context matching the entities and intent
- output the refusal phrase only when no context contains enough information
- never append the refusal phrase after giving an answer

## 3. Complete Results

| Reranker | Prompt | Gold@3 | Answer Hit | EM | Token F1 | Avg. Length | Generation Fail | Success | Contains Refusal |
| -------- | ------ | -----: | ---------: | -: | -------: | ----------: | --------------: | ------: | ---------------: |
| BM25 | original | 0.6800 | 0.2200 | 0.0000 | 0.1646 | 38.40 | 23 | 11 | 0.1200 |
| BM25 | strict | 0.6800 | 0.3200 | 0.0600 | 0.2684 | 32.08 | 18 | 16 | 0.0000 |
| LoRA v3 | original | 0.7200 | 0.2800 | 0.0400 | 0.2108 | 37.94 | 22 | 14 | 0.0400 |
| LoRA v3 | strict | 0.7200 | 0.2400 | 0.1000 | 0.3335 | 24.72 | 24 | 12 | 0.0200 |
| Cross-Encoder | original | 0.8800 | 0.2800 | 0.0000 | 0.2058 | 38.98 | 30 | 14 | 0.0600 |
| Cross-Encoder | strict | 0.8800 | 0.3200 | 0.1000 | 0.3249 | 28.98 | 28 | 16 | 0.0000 |

## 4. Main Observations

- BM25: Token F1 increased by +0.1037; Answer Hit increased by +0.1000; average answer length decreased by -6.32 words; contains-refusal rate decreased by -0.1200; generation failures decreased by -5.
- LoRA v3: Token F1 increased by +0.1227; Answer Hit decreased by -0.0400; average answer length decreased by -13.22 words; contains-refusal rate decreased by -0.0200; generation failures increased by +2.
- Cross-Encoder: Token F1 increased by +0.1191; Answer Hit increased by +0.0400; average answer length decreased by -10.00 words; contains-refusal rate decreased by -0.0600; generation failures decreased by -2.

Across the three rerankers, 2/3 show higher Token F1 without lower Answer Hit under the strict prompt.
Gold@3 remains unchanged because rankings and top-3 contexts are fixed. Any metric movement after the protocol check is attributable to generation behavior under the changed prompt.

## 5. Conclusion Boundary

Within this fixed 50-question protocol, the strict short-answer prompt has a measurable effect on output length, refusal behavior, and automatic answer metrics.

This is a downstream prompt ablation and generation-format sensitivity check. It is an extension experiment, not a reranking main result and not a same-protocol reproduction of RankRAG answer generation.

The run only evaluates Qwen2.5-1.5B-Instruct. It does not establish a Qwen2.5-7B strict-prompt conclusion.
