# Downstream RAG Data Audit

This audit checks whether the existing MS MARCO medium candidate pool and reranker rankings can support downstream answer-generation evaluation.

## Processed Test File

Path: `data/processed/msmarco_medium/test.jsonl`

| Item | Value |
| --- | --- |
| Queries | 500 |
| Candidates | 4044 |
| Top-level fields | `candidates, query, query_id` |
| Candidate fields | `doc_id, label, text, title` |
| Has `answer` | False |
| Has `answers` | False |
| Has `wellFormedAnswers` | False |

The processed subset has query, candidate passage, candidate id, and label fields, but it does not retain gold answer fields.

## Original Answer Source

Original `microsoft/ms_marco:v1.1` validation is available in the current environment and contains answer fields.

| Field | Available |
| --- | --- |
| `answers` | True |
| `wellFormedAnswers` | True |

Sample query: `walgreens store sales average`
Sample answers: `['Approximately $15,000 per year.']`

## Rankings Alignment

| Method | Exists | Status | Queries | Missing | Extra | Candidate aligned | Path |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| bm25 | True | ready | 500 | 0 | 0 | True | `outputs/msmarco_medium_retrieval_baseline_rankings.json` |
| lora_v3 | True | ready | 500 | 0 | 0 | True | `outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_rankings.json` |
| cross_encoder | True | ready | 500 | 0 | 0 | True | `outputs/msmarco_medium_cross_encoder_rankings.json` |
| jittor_qwen2_1_5b_zero_shot | True | ready | 500 | 0 | 0 | True | `outputs/jittorllm_qwen2_1_5b_full/rankings.json` |
| jittor_qwen2_0_5b_zero_shot | True | ready | 500 | 0 | 0 | True | `outputs/jittorllm_qwen2_0_5b_full/rankings.json` |
| jittor_mlp | True | ready | 500 | 0 | 0 | True | `outputs/msmarco_medium_jittor_test_rankings.json` |
| jittor_textcnn | True | ready | 500 | 0 | 0 | True | `outputs/msmarco_medium_textcnn_jittor_test_rankings.json` |

## Audit Conclusion

Methods suitable for the first downstream experiment: `bm25, lora_v3, cross_encoder`.
Additional aligned methods available for optional expansion: `jittor_qwen2_1_5b_zero_shot, jittor_qwen2_0_5b_zero_shot, jittor_mlp, jittor_textcnn`.

The project can build unified top-k contexts because the first-pass rankings cover the same 500 query ids and candidate ids as `data/processed/msmarco_medium/test.jsonl`.
