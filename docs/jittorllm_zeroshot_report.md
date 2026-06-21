# JittorLLM Qwen2.5 Zero-shot Reranking Report

## Motivation

This experiment adds a real Jittor-side LLM inference path to the RankRAG-style selector. It tests query-passage relevance without training a new reranker. The formal zero-shot comparison now includes Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct on the MS MARCO medium subset.

The experiment is separate from LoRA training results. LoRA is used only as a comparison point where explicit LoRA metrics are already present in the repository.

## Relationship to Other Rerankers

- BM25 and TF-IDF are lexical baselines.
- MLP and TextCNN Jittor are trained lightweight rankers.
- Cross-Encoder and LoRA rerankers are supervised or pretrained semantic rankers.
- Qwen2.5 JittorLLM zero-shot is an inference-only relevance scorer.

This module demonstrates Jittor LLM inference and ranking integration. It is not intended to be the strongest reranker in the project.

## Environment and Implementation

- OS: Ubuntu 22.04
- Local GPU used for 0.5B: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB
- Jittor: 1.3.11.0 locally; 1.3.10.0 in the remote 1.5B run log
- Jittor CUDA: enabled (`has_cuda=1`, `use_cuda=1`)
- Backend: JittorLLMs Qwen2
- Models: Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct

The implementation uses `external/JittorLLMs/models/qwen2` through a project adapter rather than changing the JittorLLMs registry:

```text
src/jittorllm_reranker/backend_qwen2_jittor.py
```

The model class is `Qwen2ForCausalLM`; tokenizer loading uses the JittorLLMs `Qwen2TokenizerFast`. Model tensors, forward passes, attention, and logits use `jt.Var`, `jittor.nn`, and Jittor CUDA.

The JittorLLMs implementation cannot directly load Hugging Face safetensors. The local `model.safetensors` is converted to an FP16 PyTorch-compatible state dict, which `jt.load()` reads. Hugging Face marks the input embedding and LM head as tied and omits `lm_head.weight`; conversion and loading explicitly restore that tie. PyTorch is used only for checkpoint serialization during conversion, not for inference.

## FP32 Attention Compatibility Note

In the tested environment and JittorLLMs Qwen2 implementation, the original FP16 attention path produced NaN logits for Qwen2.5-1.5B. A narrow compatibility patch temporarily casts query/key/value tensors to FP32 for scaled-dot-product attention and casts the attention output back to the original dtype afterward. The rest of the model remains FP16.

After this patch, diagnostic relevant and irrelevant samples produced finite logits in the expected direction, and the full 500-query run passed independent validation. This is reported as an observed compatibility issue in the current environment and implementation, not as a general defect in official Jittor or Qwen models.

Patch artifact:

```text
docs/qwen2_1_5b_fp32_attention.patch
```

Repeatable patch helper:

```bash
python scripts/patch_jittorllms_qwen2_fp32_attention.py external/JittorLLMs
```

The helper accepts either a JittorLLMs root directory or a direct `modeling_qwen2.py` path, checks the target snippet, creates a backup, and exits safely if the patch is already present.

## Prompt and Scoring

Free generated labels with the original `Relevant/Irrelevant` prompt collapsed to `Relevant` for all 42 smoke pairs in the 0.5B diagnostic run. That behavior is retained as a diagnostic result only. It is not used for the formal ranking experiments.

The formal runs use a stricter numeric prompt:

```text
Decide whether the passage directly answers the search query.
Return only one digit:
1 = relevant and directly answers the query
0 = irrelevant or does not directly answer the query

Query: {query}
Passage: {passage}
Label:
```

Ranking uses the first-token logit margin:

```text
score = logit("1") - logit("0")
```

This token-logit margin produces continuous ranking scores and avoids the single-label collapse observed with free generation. The selected raw label, both logits, score, runtime, query, passage, and gold label are retained in the local JSONL cache. The large score cache is not committed.

## Formal 500-query Results

Both JittorLLM zero-shot runs use the MS MARCO medium test subset: 500 queries and 4044 query-passage pairs.

| Metric | Qwen2.5-0.5B Zero-shot | Qwen2.5-1.5B Zero-shot |
| --- | ---: | ---: |
| Recall@1 | 0.1660 | 0.2360 |
| Recall@3 | 0.4540 | 0.5520 |
| Recall@5 | 0.6760 | 0.8120 |
| Recall@10 | 1.0000 | 1.0000 |
| NDCG@1 | 0.1660 | 0.2360 |
| NDCG@3 | 0.3281 | 0.4139 |
| NDCG@5 | 0.4189 | 0.5210 |
| NDCG@10 | 0.5258 | 0.5832 |
| MRR | 0.3804 | 0.4525 |
| Pairwise Accuracy | 0.5367 | 0.6342 |

### Qwen2.5-1.5B Run Details

| Item | Value |
| --- | ---: |
| Model | Qwen2.5-1.5B-Instruct |
| Backend | JittorLLMs Qwen2 |
| Dataset | MS MARCO medium subset |
| Queries | 500 |
| Candidate pairs | 4044 |
| Cache reused pairs | 0 |
| Newly inferred pairs | 4044 |
| Parse failures | 0 |
| Pure inference time | 114.5817s |
| Wall time | 147.7716s |
| Throughput | 35.2936 pairs/s |
| Peak observed GPU memory | 3608 MiB |

### Qwen2.5-1.5B Score Statistics

| Statistic | Value |
| --- | ---: |
| Min score | -8.156250 |
| Max score | 10.062500 |
| Mean score | 1.410160 |
| Score std | 2.897395 |
| Positive mean | 2.381719 |
| Negative mean | 1.273089 |
| Unique score values | 831 |
| Tie groups | 683 |
| Tied items | 3896 |
| Tied pairs | 12094 |

## Independent Validation

Validation output:

```text
outputs/jittorllm_qwen2_1_5b_full/validation.json
```

| Check | Result |
| --- | ---: |
| Status | passed |
| Recomputed queries | 500 / 500 |
| Recomputed pairs | 4044 / 4044 |
| Duplicate pairs | 0 |
| Missing pairs | 0 |
| Extra pairs | 0 |
| Parse failures | 0 |
| Rankings consistent with score-descending order | true |
| Metrics consistent with `metrics.json` | true |
| Max metric absolute difference | 0.0 |

The 0.5B full run also passed the same independent validation checks.

## Comparison With Other Medium-subset Rerankers

| Method | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | MRR | Pairwise Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.2300 | 0.5540 | 0.7840 | 0.5074 | 0.4476 | 0.6253 |
| Qwen2.5-0.5B Jittor zero-shot | 0.1660 | 0.4540 | 0.6760 | 0.4189 | 0.3804 | 0.5367 |
| Qwen2.5-1.5B Jittor zero-shot | 0.2360 | 0.5520 | 0.8120 | 0.5210 | 0.4525 | 0.6342 |
| Qwen2.5-1.5B LoRA v3 | 0.3580 | 0.6980 | 0.8720 | 0.6266 | 0.5642 | 0.7345 |
| Cross-Encoder | 0.4340 | 0.8080 | 0.9340 | 0.7019 | 0.6341 | not reported |

The 1.5B zero-shot model clearly improves over the 0.5B zero-shot model. Against BM25, it is broadly comparable: Recall@1, Recall@5, NDCG@5, MRR, and pairwise accuracy are slightly higher, while Recall@3 is essentially tied and slightly lower. This should not be described as a dramatic or definitive win over BM25.

The LoRA v3 result is much stronger than 1.5B zero-shot at the same base scale, which shows that task-specific training remains important. The external Cross-Encoder remains the strongest reported reranker among these references.

## Earlier Small Runs

The 5-query and 20-query 0.5B runs were used as integration checks before the full run.

| Metric | 5 queries | 20 queries |
| --- | ---: | ---: |
| recall@1 | 0.0000 | 0.3000 |
| recall@3 | 0.6000 | 0.6000 |
| recall@5 | 0.8000 | 0.7000 |
| recall@10 | 1.0000 | 1.0000 |
| ndcg@1 | 0.0000 | 0.3000 |
| ndcg@3 | 0.3524 | 0.4827 |
| ndcg@5 | 0.4297 | 0.5214 |
| ndcg@10 | 0.5010 | 0.6201 |
| mrr | 0.3400 | 0.5032 |
| pairwise_accuracy | 0.6194 | 0.6395 |
| candidate pairs | 42 | 158 |
| parse failures | 0 | 0 |

## Case Study

Detailed top-1 correct and incorrect examples for 0.5B and 1.5B are in:

```text
docs/jittorllm_qwen2_full_case_study.md
```

The correct 1.5B examples are mostly direct factoid matches where the selected passage contains the requested value or definition. The incorrect examples often involve strong topical overlap, alternative answer formulations, or passages marked as negative that can still directly answer the query. These should be described as potential false negatives or incomplete labels, not automatically as dataset annotation errors.

## Commands and Outputs

0.5B commands:

```bash
bash scripts/run_jittorllm_qwen2_0_5b_smoke.sh
bash scripts/run_jittorllm_qwen2_0_5b_20q.sh
bash scripts/run_jittorllm_qwen2_0_5b_full.sh
```

1.5B configs:

```text
configs/jittorllm_qwen2_1_5b_smoke.yaml
configs/jittorllm_qwen2_1_5b_full.yaml
```

Validation command:

```bash
python scripts/validate_jittorllm_qwen2_results.py --output-dir outputs/jittorllm_qwen2_1_5b_full
```

Committed 1.5B outputs:

```text
outputs/jittorllm_qwen2_1_5b_full/metrics.json
outputs/jittorllm_qwen2_1_5b_full/rankings.json
outputs/jittorllm_qwen2_1_5b_full/run_summary.md
outputs/jittorllm_qwen2_1_5b_full/validation.json
```

`score_cache.jsonl`, model weights, Hugging Face cache, JittorLLMs checkout, Jittor compile artifacts, virtual environments, AutoDL raw logs, and AutoDL environment scripts are intentionally not committed.

## Interpretation and Limitations

This experiment validates a real Jittor CUDA LLM-style zero-shot reranking path, including weight conversion, Qwen2 inference, token-logit scoring, cache/resume behavior, full-run metrics, FP32-attention compatibility handling for 1.5B, and independent result validation.

Scaling from 0.5B to 1.5B materially improves zero-shot reranking. The 1.5B zero-shot result is close to BM25 overall and slightly higher on several metrics, but it remains below LoRA and Cross-Encoder results. The main conclusion is therefore infrastructure plus a useful zero-shot baseline, not replacement of task-specific training.
