# Qwen2.5-0.5B JittorLLM Reranking Run

- Status: ready
- Backend: qwen2_jittorllms
- Model: Qwen2.5-0.5B-Instruct
- Dataset: `data/processed/msmarco_medium/test.jsonl`
- Num queries: 5
- Num candidate pairs: 42
- Scoring method: label_token_logit_margin
- Parse failures: 0 (0.00%)
- Model inference time: 4.644s
- Pairs per second: 9.0447
- Label-token logit scoring: available
- Generated-label fallback: not used

## Metrics

| Metric | Value |
| --- | ---: |
| recall@1 | 0.0000 |
| recall@3 | 0.6000 |
| recall@5 | 0.8000 |
| recall@10 | 1.0000 |
| ndcg@1 | 0.0000 |
| ndcg@3 | 0.3524 |
| ndcg@5 | 0.4297 |
| ndcg@10 | 0.5010 |
| mrr | 0.3400 |
| pairwise_accuracy | 0.6194 |

## Prompt

```text
Decide whether the passage directly answers the search query.
Return only one digit:
1 = relevant and directly answers the query
0 = irrelevant or does not directly answer the query

Query: {query}
Passage: {passage}
Label:

```

## Limitations

This is label-token logit-margin zero-shot scoring on a small subset. It is a Jittor LLM inference proof of concept, not a replacement for a trained cross-encoder.
