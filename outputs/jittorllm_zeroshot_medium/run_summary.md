# JittorLLM Zero-shot Reranking Run Summary

- Status: blocked
- Model/backend: jittorllm-unconfigured
- Dataset: `data/processed/msmarco_medium/test.jsonl`
- Cache path: `outputs/jittorllm_zeroshot_medium/score_cache.jsonl`
- Num queries: 0
- Num pairs: 0
- Inference time sec: 0.019509315490722656
- Pairs per second: 0.0
- Logprob scoring available: False
- Fallback generated-label scoring: False
- Parse fail count: 0

## Metrics

| Metric | Value |
| --- | ---: |
| recall@1 | N/A |
| recall@3 | N/A |
| recall@5 | N/A |
| recall@10 | N/A |
| ndcg@1 | N/A |
| ndcg@3 | N/A |
| ndcg@5 | N/A |
| ndcg@10 | N/A |
| mrr | N/A |
| pairwise_accuracy | N/A |

## Prompt Template

```text
Given a search query and a candidate passage, determine whether the passage is relevant to the query.

Query:
{query}

Passage:
{passage}

Answer with exactly one label:
Relevant or Irrelevant.

```

## Limitations

JittorLLM inference is blocked because the local JittorLLM package/model environment is unavailable. No relevance scores were fabricated.
