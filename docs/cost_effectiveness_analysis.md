# Resource and Effectiveness Profile

## 1. Purpose

Stage F builds a resource and effectiveness profile for six reranking methods in this repository. It is not a strict hardware-normalized benchmark and does not claim a full RankRAG reproduction. The project remains a Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.

## 2. Methods

- BM25: lexical retrieval baseline without neural model training.
- Jittor MLP: lightweight Jittor neural ranking baseline trained from scratch on handcrafted features.
- Jittor TextCNN: lightweight Jittor neural ranking baseline trained from scratch on local text patterns.
- Qwen2.5-1.5B zero-shot reranker: pretrained LLM relevance scoring without local task training.
- Qwen2.5-1.5B LoRA 10k-rerun: the Stage E 10k-rerun LoRA adapter using `logprob_margin` scoring.
- Cross-Encoder: external pretrained semantic reranker reference, not the Jittor reproduction body.

## 3. Measurement Protocol

The effectiveness metrics come from existing committed metrics files. The six main methods all contain 500 queries and 4044 query-passage pairs, and the candidate-pool signature matches across methods. The MS MARCO medium test split SHA256 is `497e99a2acd46c1336bcdc0ab755100d0fb43f2124f88029c1a0dbc3bf4d7504`. No model was retrained, no model inference was rerun, and missing resource fields are recorded as `not recorded`.

Resource records come from heterogeneous artifacts: local Jittor logs, JittorLLM metrics, Cross-Encoder metrics, and AutoDL RTX 4090 D LoRA logs. Because hardware, software, batching, and runtime scopes differ, resource fields form a profile rather than a fair speed ranking.

## 4. Effectiveness Profile

| Method | R@1 | MRR | NDCG@5 | Training | Pretrained semantics | Model scale | Eval runtime | Peak VRAM | Hardware | Resource comparability |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| BM25 | 0.2300 | 0.4476 | 0.5074 | none | no | N/A | not recorded | not recorded | not recorded | cpu_local_retrieval; strict=false |
| Jittor MLP | 0.2280 | 0.4318 | 0.4698 | from_scratch | no | 0.394M static | not recorded | not recorded | not recorded | jittor_local_training; strict=false |
| Jittor TextCNN | 0.1800 | 0.3912 | 0.4270 | from_scratch | no | 3.899M static | not recorded | not recorded | not recorded | jittor_local_training; strict=false |
| Qwen2.5-1.5B zero-shot reranker | 0.2360 | 0.4525 | 0.5210 | none | yes | 1.5B class | 114.582s (scoring_only) | 3608.0 MiB | not recorded | qwen_zero_shot_recorded_environment; strict=false |
| Qwen2.5-1.5B LoRA 10k-rerun | 0.3560 | 0.5633 | 0.6236 | lora_finetuning | yes | 1.5B class | 444.287s (scoring_only) | 16509.3 MiB | NVIDIA GeForce RTX 4090 D | autodl_4090d_lora; strict=false |
| Cross-Encoder | 0.4340 | 0.6341 | 0.7019 | none_in_this_project | yes | MiniLM-L6 cross-encoder class | 5.955s (scoring_only) | not recorded | cuda | pretrained_cross_encoder_unknown_hardware; strict=false |

BM25 reaches R@1=0.2300 and NDCG@5=0.5074. The lightweight Jittor MLP is close on R@1 but lower on NDCG@5, while Jittor TextCNN is weaker on this candidate pool. Qwen2.5-1.5B zero-shot improves NDCG@5 over the lightweight Jittor models, showing the value of pretrained semantic ability even without task training. The Stage E LoRA 10k-rerun raises R@1 to 0.3560 and NDCG@5 to 0.6236. The Cross-Encoder remains the strongest ranking reference in this set, with R@1=0.4340 and NDCG@5=0.7019.

## 5. Resource Profile

BM25 has no neural parameter count and no recorded runtime or hardware in the committed metrics. Jittor MLP and Jittor TextCNN have static parameter counts derived from source/configuration, but their training and evaluation runtime summaries, hardware, and peak memory were not recorded. Qwen zero-shot records scoring runtime and observed GPU memory in its metrics, but no explicit GPU model or software environment. The LoRA 10k-rerun records AutoDL NVIDIA GeForce RTX 4090 D hardware, Python/Torch/CUDA environment, training runtime, scoring runtime, and peak GPU memory. Cross-Encoder records CUDA device, batch size, evaluation runtime, query count, and pair count, but not the exact GPU model or peak memory.

No pair of methods is marked as strictly resource comparable in the main table. The strict flag requires the same hardware, software environment, query and pair count, runtime scope, batch or serial path, and loader scope. The current artifacts satisfy the effectiveness comparability requirement, but not the strict resource comparability requirement.

## 6. Internal LoRA Scoring Cost

The E2 scoring ablation uses the same 10k-rerun LoRA adapter. These rows are not separate main methods; they isolate scoring-rule cost and behavior.

| Scoring method | R@1 | NDCG@5 | MRR | Runtime sec | Parse failure rate | Query tie rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate_parse | 0.1280 | 0.3848 | 0.3490 | 904.754 | 0.0010 | 1.0000 |
| relevant_logprob | 0.3560 | 0.6254 | 0.5637 | 227.470 | 0.0000 | 0.0080 |
| logprob_margin | 0.3560 | 0.6236 | 0.5633 | 454.050 | 0.0000 | 0.0100 |

`generate_parse` calls generation for each pair and is much slower while producing many ties. `relevant_logprob` scores the `Relevant` label sequence only. `logprob_margin` scores both `Relevant` and `Irrelevant` label sequences and uses their difference; it is the Stage E default used in the main table.

## 7. Main Trade-off

The practical ladder is lexical low-resource retrieval, lightweight Jittor neural baselines, zero-shot LLM reranking, task-adapted LoRA reranking, and a mature Cross-Encoder reference. BM25 is cheap and simple but semantically limited. Jittor MLP/TextCNN are useful for framework alignment and lightweight neural comparisons, not for maximum ranking quality. Qwen zero-shot adds pretrained semantic ability without task training. LoRA adds task-specific adaptation and produces a clear effectiveness gain over zero-shot in this setup. Cross-Encoder gives the strongest ranking metrics, while LoRA is most relevant to the RankRAG-style LLM reranking reproduction objective.

## 8. Limitations

- Runtime and memory were collected under different hardware and implementation paths.
- Some methods lack recorded training runtime, evaluation runtime, GPU model, or peak memory.
- Runtime scopes differ and may or may not include loader overhead.
- Batch sizes and serial/parallel scoring paths differ.
- The current table is not a strict speed benchmark or a hardware-normalized efficiency ranking.
- Parameter scale and observed latency are not linearly related.
- The conclusions are limited to the current MS MARCO medium subset.

## Historical 10k Reproducibility Reference

The historical 10k LoRA directory is retained only as a reproducibility reference: `outputs/lora_qwen_1_5b_10k_lr1e4_s800/`. The main Stage F table uses `outputs/lora_ablation_data_10k_rerun/` instead. Historical metrics are R@1=0.3580, R@3=0.6980, R@5=0.8720, NDCG@5=0.6266, MRR=0.5642, pairwise accuracy=0.7345. Training runtime and memory should not be strictly compared with the rerun because the exact environment and code version are not fully identical.

## Figure

`docs/figures/cost_effectiveness_tradeoff.png` is a non-strict resource profile figure because fewer than three methods satisfy strict resource comparability. It plots NDCG@5 and annotates training/pretraining/hardware context rather than drawing a runtime scatter.
