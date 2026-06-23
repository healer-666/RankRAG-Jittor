# Reproduction Guide

This guide lists the verified entry points used by this repository. It separates CPU-only checks from commands that train models or run model inference. Model weights and LoRA adapters are not stored in Git.

Use `python` in the commands below. On this Windows workstation, the equivalent interpreter is:

```powershell
& "D:\anaconda\envs\pytorch\python.exe" <script> <args>
```

## 1. Environment Matrix

| Component | Recommended OS | Framework | GPU required |
| --- | --- | --- | --- |
| Result aggregation and repository checks | Windows or Linux | Python | No |
| TF-IDF / BM25 | Windows or Linux | scikit-learn, rank_bm25 | No |
| PyTorch MLP/TextCNN baselines | Windows or Linux | PyTorch | No for current configs |
| Jittor MLP/TextCNN baselines | Ubuntu preferred | Jittor | No for current CPU configs |
| Qwen zero-shot reranking | Linux preferred | JittorLLM | Yes for practical full run |
| Qwen LoRA reranker | Linux + CUDA | PyTorch, Transformers, PEFT | Yes |
| Downstream generation | Linux + CUDA | Transformers | Yes |

## 2. Quick Repository Check

These commands do not train models or run model inference.

```bash
python -m py_compile scripts/build_lora_data_size_ablation.py
python -m py_compile scripts/check_lora_data_ablation.py
python -m py_compile scripts/aggregate_lora_data_size_ablation.py
python -m py_compile scripts/aggregate_lora_scoring_ablation.py
python -m py_compile scripts/aggregate_error_taxonomy.py
python -m py_compile scripts/build_cost_effectiveness_profile.py
python -m py_compile scripts/check_cost_effectiveness_outputs.py
python -m py_compile scripts/build_final_project_summary.py
python -m py_compile scripts/check_final_repository.py

python scripts/check_lora_data_ablation.py
python scripts/check_cost_effectiveness_outputs.py
python scripts/build_final_project_summary.py
python scripts/check_final_repository.py
```

## 3. Data Preparation

Prepare the synthetic smoke data:

```bash
python scripts/prepare_data.py
```

Build the MS MARCO medium subset:

```bash
python scripts/prepare_msmarco_subset.py \
  --max_train_queries 5000 \
  --max_valid_queries 500 \
  --max_test_queries 500 \
  --candidates_per_query 10 \
  --output_dir data/processed/msmarco_medium \
  --run_name msmarco_medium \
  --seed 42
```

Build nested LoRA E1 training subsets and validate them:

```bash
python scripts/build_lora_data_size_ablation.py
python scripts/check_lora_data_ablation.py
```

## 4. Lightweight PyTorch/Jittor Alignment

The MLP and TextCNN models are lightweight PyTorch/Jittor alignment baselines. They are not the core models from the original RankRAG paper.

```bash
python src/train_torch.py --config configs/msmarco_medium.yaml
python src/eval_torch.py --config configs/msmarco_medium.yaml

python src/train_jittor.py --config configs/msmarco_medium.yaml
python src/eval_jittor.py --config configs/msmarco_medium.yaml

python src/train_textcnn_torch.py --config configs/msmarco_medium_textcnn.yaml
python src/eval_textcnn_torch.py --config configs/msmarco_medium_textcnn.yaml

python src/train_textcnn_jittor.py --config configs/msmarco_medium_textcnn.yaml
python src/eval_textcnn_jittor.py --config configs/msmarco_medium_textcnn.yaml
```

Aggregate the medium-subset alignment results:

```bash
python src/aggregate_l2_results.py --run_name msmarco_medium
```

## 5. Retrieval and Reranking Baselines

TF-IDF and BM25:

```bash
python src/baseline_retrieval.py \
  --data_path data/processed/msmarco_medium/test.jsonl \
  --output_path outputs/msmarco_medium_retrieval_baseline_metrics.json \
  --rankings_path outputs/msmarco_medium_retrieval_baseline_rankings.json \
  --run_name msmarco_medium
```

Qwen2.5-1.5B zero-shot reranking uses `configs/jittorllm_qwen2_1_5b_full.yaml` and requires a local model path outside the repository:

```bash
python src/jittorllm_reranker/evaluate_qwen2_jittor.py \
  --config configs/jittorllm_qwen2_1_5b_full.yaml
```

Cross-Encoder reference:

```bash
python src/pretrained_cross_encoder_reference.py \
  --data_path data/processed/msmarco_medium/test.jsonl \
  --model_name cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output_metrics outputs/msmarco_medium_cross_encoder_metrics.json \
  --output_rankings outputs/msmarco_medium_cross_encoder_rankings.json \
  --batch_size 32 \
  --max_queries 500
```

## 6. LoRA Reranker

Set the base Qwen path outside the repository before training or evaluation:

```bash
export QWEN_LORA_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
```

The Stage E1 configs are:

```bash
python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_1k_lr1e4_s800.yaml
python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_3k_lr1e4_s800.yaml
python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml
```

Evaluate each trained adapter with the matching config and output directory:

```bash
python src/lora_reranker/evaluate_lora_reranker.py --config configs/lora_qwen_1_5b_1k_lr1e4_s800.yaml
python src/lora_reranker/evaluate_lora_reranker.py --config configs/lora_qwen_1_5b_3k_lr1e4_s800.yaml
python src/lora_reranker/evaluate_lora_reranker.py --config configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml
```

Aggregate E1 and E2 outputs:

```bash
python scripts/aggregate_lora_data_size_ablation.py
python scripts/aggregate_lora_scoring_ablation.py
```

The historical directory `outputs/lora_qwen_1_5b_10k_lr1e4_s800/` is retained as a reproducibility reference. Do not overwrite it when reproducing E1; use the `10k-rerun` config.

## 7. Downstream RAG

Set the generator model path outside the repository:

```bash
export QWEN_GENERATOR_MODEL_PATH=/path/to/Qwen2.5-Instruct
```

Qwen2.5-1.5B original prompt:

```bash
python src/audit_downstream_rag_data.py --config configs/downstream_rag_50q.yaml
python src/build_downstream_qa_subset.py --config configs/downstream_rag_50q.yaml
python scripts/run_downstream_rag_eval.py --config configs/downstream_rag_50q.yaml --methods bm25,lora_v3,cross_encoder --top-k 3
python src/aggregate_downstream_rag_results.py --config configs/downstream_rag_50q.yaml --input-dir outputs/downstream_rag_eval --top-k 3
python scripts/validate_downstream_rag_results.py --config configs/downstream_rag_50q.yaml --output-dir outputs/downstream_rag_eval --top-k 3
```

Strict prompt and 7B variants use:

```bash
python scripts/run_downstream_rag_eval.py --config configs/downstream_rag_50q_qwen2_1_5b_strict_prompt.yaml --methods bm25,lora_v3,cross_encoder --top-k 3
python scripts/run_downstream_rag_eval.py --config configs/downstream_rag_50q_qwen2_7b.yaml --methods bm25,lora_v3,cross_encoder --top-k 3
python scripts/run_downstream_rag_eval.py --config configs/downstream_rag_50q_qwen2_7b_strict_prompt.yaml --methods bm25,lora_v3,cross_encoder --top-k 3
```

## 8. Analysis-Only Stages

These commands are CPU-only and read existing artifacts:

```bash
python scripts/aggregate_lora_data_size_ablation.py
python scripts/aggregate_lora_scoring_ablation.py
python scripts/aggregate_error_taxonomy.py
python scripts/build_cost_effectiveness_profile.py
python scripts/check_cost_effectiveness_outputs.py
python scripts/build_final_project_summary.py
python scripts/check_final_repository.py
```

## 9. Expected Outputs

| Stage | Key outputs |
| --- | --- |
| Main ranking | `outputs/cost_effectiveness_table.json`, `docs/final_results.md` |
| PyTorch/Jittor alignment | `outputs/l2_msmarco_medium_results.json` |
| LoRA E1 | `outputs/lora_ablation_results.json` |
| LoRA E2 | `outputs/lora_scoring_ablation_results.json` |
| Downstream RAG | `outputs/downstream_rag_eval*/downstream_rag_eval_results.json` |
| Error taxonomy | `outputs/error_taxonomy_summary.json`, `docs/error_taxonomy.md` |
| Resource profile | `outputs/cost_effectiveness_table.json`, `docs/cost_effectiveness_analysis.md` |
| Final summary | `outputs/final_results_summary.json`, `docs/final_results.md` |

## 10. Reproducibility Notes

- Main ranking uses the MS MARCO medium test split with 500 queries and 4044 query-passage pairs.
- E1 uses nested 1k / 3k / 10k training subsets with fixed 800 optimizer steps.
- Smaller E1 subsets therefore see more effective epochs: 1k = 6.4, 3k = 2.1333, 10k-rerun = 0.64.
- The LoRA main result is `10k-rerun`; the historical 10k output is only a reproducibility reference.
- Downstream RAG uses 50 questions and top-3 contexts.
- Error taxonomy uses 30 stratified diagnostic cases, not the full 500-query distribution.
- Resource measurements come from heterogeneous environments and should not be read as a strict speed benchmark.
- Adapter files and base model weights are intentionally not tracked in Git.
