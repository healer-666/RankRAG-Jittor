# Final Results

This file is generated from committed result artifacts by `scripts/build_final_project_summary.py`. The machine-readable source of truth is `outputs/final_results_summary.json`.

## 1. Main Reranking Results

The main reranking table uses the same MS MARCO medium subset: 500 queries, 4044 query-passage pairs, the same candidate pool, and the same gold definition. The LoRA row uses the Stage E `10k-rerun` result, not the historical 10k run.

| Method | R@1 | R@3 | R@5 | NDCG@5 | MRR | Pairwise Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.230 | 0.554 | 0.784 | 0.5074 | 0.4476 | 0.6253 |
| Jittor MLP | 0.228 | 0.506 | 0.712 | 0.4698 | 0.4318 | 0.5901 |
| Jittor TextCNN | 0.180 | 0.450 | 0.678 | 0.4270 | 0.3912 | 0.5484 |
| Qwen2.5-1.5B zero-shot reranker | 0.236 | 0.552 | 0.812 | 0.5210 | 0.4525 | 0.6342 |
| Qwen2.5-1.5B LoRA 10k-rerun | 0.356 | 0.696 | 0.866 | 0.6236 | 0.5633 | 0.7343 |
| Cross-Encoder | 0.434 | 0.808 | 0.934 | 0.7019 | 0.6341 | 0.8049 |

## 2. PyTorch/Jittor Alignment

MLP and TextCNN are lightweight alignment baselines. They are not core models from the original RankRAG paper, and exact number-by-number equality is not expected.

| Framework | Model | R@1 | R@3 | R@5 | NDCG@5 | MRR | Delta R@1 vs PyTorch | Delta NDCG@5 vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | MLP | 0.192 | 0.478 | 0.700 | 0.4475 | 0.4079 | baseline | baseline |
| Jittor | MLP | 0.228 | 0.506 | 0.712 | 0.4698 | 0.4318 | 0.0360 | 0.0222 |
| PyTorch | TextCNN | 0.172 | 0.496 | 0.722 | 0.4463 | 0.3953 | baseline | baseline |
| Jittor | TextCNN | 0.180 | 0.450 | 0.678 | 0.4270 | 0.3912 | 0.0080 | -0.0194 |

## 3. LoRA Data-Size Ablation

E1 fixes the optimizer-step budget at 800 steps. It measures the effect of adding training-data diversity, not a fixed-epoch comparison. The 1k, 3k, and 10k-rerun subsets are nested.

| Run | Training pairs | Effective epochs | Max steps | R@1 | R@3 | R@5 | NDCG@5 | MRR | Pairwise Acc | Train runtime sec | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 1000 | 6.4000 | 800 | 0.288 | 0.668 | 0.852 | 0.5808 | 0.5125 | 0.6934 | 1332.110 | 4679.4 |
| 3k | 3000 | 2.1333 | 800 | 0.362 | 0.692 | 0.854 | 0.6187 | 0.5627 | 0.7240 | 1271.951 | 4743.5 |
| 10k-rerun | 10000 | 0.6400 | 800 | 0.356 | 0.696 | 0.866 | 0.6236 | 0.5633 | 0.7343 | 187.895 | 16509.3 |

## 4. Scoring-Method Ablation

E2 compares scoring rules for the same 10k-rerun LoRA adapter. These are not three different models.

| Scoring method | R@1 | R@3 | R@5 | NDCG@5 | MRR | Runtime sec | Parse failure rate | Query tie rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generate_parse | 0.128 | 0.424 | 0.646 | 0.3848 | 0.3490 | 904.754 | 0.0010 | 1.0000 |
| relevant_logprob | 0.356 | 0.698 | 0.870 | 0.6254 | 0.5637 | 227.470 | 0.0000 | 0.0080 |
| logprob_margin | 0.356 | 0.696 | 0.866 | 0.6236 | 0.5633 | 454.050 | 0.0000 | 0.0100 |

`generate_parse` is much slower and produces many ties. `relevant_logprob` and `logprob_margin` have nearly identical R@1 but differ slightly across the full ranking metrics.

## 5. Downstream RAG Results

Downstream RAG uses 50 questions and top-3 contexts. Higher evidence availability does not guarantee the generator will use that evidence correctly.

| Generator / Prompt | Reranker | Gold@3 | Answer Hit | EM | Token F1 | Successes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B + Original Prompt | bm25 | 0.680 | 0.220 | 0.000 | 0.1646 | 11 |
| Qwen2.5-1.5B + Original Prompt | lora_v3 | 0.720 | 0.280 | 0.040 | 0.2108 | 14 |
| Qwen2.5-1.5B + Original Prompt | cross_encoder | 0.880 | 0.280 | 0.000 | 0.2058 | 14 |
| Qwen2.5-1.5B + Strict Short-Answer Prompt | bm25 | 0.680 | 0.320 | 0.060 | 0.2684 | 16 |
| Qwen2.5-1.5B + Strict Short-Answer Prompt | lora_v3 | 0.720 | 0.240 | 0.100 | 0.3335 | 12 |
| Qwen2.5-1.5B + Strict Short-Answer Prompt | cross_encoder | 0.880 | 0.320 | 0.100 | 0.3249 | 16 |
| Qwen2.5-7B + Original Prompt | bm25 | 0.680 | 0.320 | 0.000 | 0.1641 | 16 |
| Qwen2.5-7B + Original Prompt | lora_v3 | 0.720 | 0.240 | 0.000 | 0.1569 | 12 |
| Qwen2.5-7B + Original Prompt | cross_encoder | 0.880 | 0.300 | 0.000 | 0.1635 | 15 |
| Qwen2.5-7B + Strict Short-Answer Prompt | bm25 | 0.680 | 0.340 | 0.020 | 0.2072 | 17 |
| Qwen2.5-7B + Strict Short-Answer Prompt | lora_v3 | 0.720 | 0.340 | 0.020 | 0.2078 | 17 |
| Qwen2.5-7B + Strict Short-Answer Prompt | cross_encoder | 0.880 | 0.400 | 0.000 | 0.1800 | 20 |

## 6. Error Analysis Summary

Stage G uses 30 unique stratified diagnostic queries: six cases from each A/B/C/D/E selection bucket. This is not an unbiased estimate over all 500 queries.

| Primary error type | Count |
| --- | ---: |
| ambiguous_query | 3 |
| background_only | 2 |
| candidate_or_label_issue | 3 |
| evidence_utilization_failure | 6 |
| lexical_trap | 4 |
| llm_overjudgment | 4 |
| multi_evidence_confusion | 2 |
| small_model_semantic_limit | 6 |

- High-confidence cases: 16
- Medium-confidence cases: 14
- Self-consistency sample size: 10
- Primary label agreement: 0.70
- Secondary label agreement: 1.00
- Note: self-consistency is single-annotator repeated annotation, not inter-annotator agreement.

## 7. Resource and Effectiveness Profile

The effectiveness metrics are comparable across the six main methods. Resource records are heterogeneous, so they are a profile rather than a strict speed benchmark.

| Method | R@1 | NDCG@5 | Training | Pretrained semantics | Train runtime | Eval runtime | Peak VRAM | Hardware | Strict resource comparable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.230 | 0.5074 | none | no | not recorded | not recorded | not recorded | not recorded | false |
| Jittor MLP | 0.228 | 0.4698 | from_scratch | no | not recorded | not recorded | not recorded | not recorded | false |
| Jittor TextCNN | 0.180 | 0.4270 | from_scratch | no | not recorded | not recorded | not recorded | not recorded | false |
| Qwen2.5-1.5B zero-shot reranker | 0.236 | 0.5210 | none | yes | not recorded | 114.582 | 3608.0 | not recorded | false |
| Qwen2.5-1.5B LoRA 10k-rerun | 0.356 | 0.6236 | lora_finetuning | yes | 187.895 | 444.287 | 16509.3 | NVIDIA GeForce RTX 4090 D | false |
| Cross-Encoder | 0.434 | 0.7019 | none_in_this_project | yes | not recorded | 5.955 | not recorded | cuda | false |

Strictly resource-comparable method count: 0.

## 8. Source Files

The main source files used to build this summary are recorded in `outputs/final_results_summary.json` under `source_files`.
