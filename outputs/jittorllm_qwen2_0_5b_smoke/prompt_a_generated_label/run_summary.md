# Qwen2.5-0.5B JittorLLM Reranking Run

- Status: ready
- Backend: qwen2_jittorllms
- Model: Qwen2.5-0.5B-Instruct
- Dataset: `data/processed/msmarco_medium/test.jsonl`
- Num queries: 5
- Num candidate pairs: 42
- Scoring method: greedy_generated_label
- Parse failures: 0 (0.00%)
- Model inference time: 23.526s
- Pairs per second: 1.7853
- Logprob scoring: unavailable
- Generated-label fallback: used

## Metrics

| Metric | Value |
| --- | ---: |
| recall@1 | 0.6000 |
| recall@3 | 0.6000 |
| recall@5 | 0.8000 |
| recall@10 | 1.0000 |
| ndcg@1 | 0.6000 |
| ndcg@3 | 0.6000 |
| ndcg@5 | 0.6774 |
| ndcg@10 | 0.7405 |
| mrr | 0.6650 |
| pairwise_accuracy | 0.0000 |

## Prompt

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

This is generated-label zero-shot scoring on a small subset. Binary 0/1 scores create ties and are coarser than label-token log probabilities or a trained cross-encoder.
