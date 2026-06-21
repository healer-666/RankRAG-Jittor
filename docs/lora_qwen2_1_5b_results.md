# Qwen2.5-1.5B LoRA Reranker Results

This document records the formal rented-GPU LoRA reranker experiment for the RankRAG-style context ranking reproduction. The experiment keeps the project scope focused on evidence selection / reranking and does not reproduce full RankRAG answer generation.

## Task

The LoRA reranker scores each query-passage pair with a Qwen2.5 instruction model. The evaluation score is:

```text
score = log P("Relevant") - log P("Irrelevant")
```

Candidates are sorted by this score, and ranking metrics are computed on the same MS MARCO medium test candidate set used by the BM25, Jittor MLP/TextCNN, JittorLLM zero-shot, and Cross-Encoder reference runs.

## Formal v3 Setup

| Item | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Framework | PyTorch + PEFT LoRA |
| Train pairs | 10,000 |
| Valid pairs | 1,000 |
| Test queries | 500 |
| Test query-passage pairs | 4,044 |
| Max length | 256 |
| Max train steps | 800 |
| Learning rate | 1e-4 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Runtime | 1124.75 s |

The v3 training loss decreased from 3.5866 to 0.0878. This is useful as an optimization sanity check, but the final claim is based on test-set ranking metrics.

## Tuning Summary

The project keeps all three small tuning runs to show the path to the formal result.

| Run | Train pairs | Valid pairs | Steps | LR | LoRA r | Loss start | Loss end | R@1 | R@5 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B LoRA v1 | 5000 | 500 | 600 | 1e-4 | 8 | 0.0665 | 0.6983 | 0.3440 | 0.8600 | 0.6151 | 0.5549 |
| Qwen2.5-1.5B LoRA v2 | 5000 | 500 | 500 | 5e-5 | 8 | 0.0665 | 0.1202 | 0.2920 | 0.8440 | 0.5812 | 0.5171 |
| Qwen2.5-1.5B LoRA v3 | 10000 | 1000 | 800 | 1e-4 | 8 | 3.5866 | 0.0878 | 0.3580 | 0.8720 | 0.6266 | 0.5642 |

The original v1 config records 600 train steps. If older notes mention 500 steps for v1, the committed config and training summary should be treated as the source of truth.

The tuning result is restrained: simply lowering the learning rate did not improve the model; expanding the training pairs and keeping the stronger learning rate produced the most stable overall result.

## Main Comparison

The following table is generated from the committed JSON metrics by `python src/aggregate_lora_results.py`.

| Method | Framework | Training | R@1 | R@3 | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR | Pairwise Acc. |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | rank_bm25 | none | 0.2300 | 0.5540 | 0.7840 | 1.0000 | 0.5074 | 0.5791 | 0.4476 | 0.6253 |
| Jittor MLP | Jittor | from scratch | 0.2280 | 0.5060 | 0.7120 | 1.0000 | 0.4698 | 0.5657 | 0.4318 | 0.5901 |
| Jittor TextCNN | Jittor | from scratch | 0.1800 | 0.4500 | 0.6780 | 1.0000 | 0.4270 | 0.5341 | 0.3912 | 0.5484 |
| Qwen2.5-0.5B Jittor zero-shot | JittorLLM | zero-shot | 0.1660 | 0.4540 | 0.6760 | 1.0000 | 0.4189 | 0.5258 | 0.3804 | 0.5367 |
| Qwen2.5-1.5B Jittor zero-shot | JittorLLM | zero-shot | 0.2360 | 0.5520 | 0.8120 | 1.0000 | 0.5210 | 0.5832 | 0.4525 | 0.6342 |
| Qwen2.5-1.5B LoRA v3 | PyTorch | LoRA, 10k pairs, lr=1e-4 | 0.3580 | 0.6980 | 0.8720 | 1.0000 | 0.6266 | 0.6698 | 0.5642 | 0.7345 |
| Cross-Encoder reference | sentence-transformers | external pretrained | 0.4340 | 0.8080 | 0.9340 | 1.0000 | 0.7019 | 0.7242 | 0.6341 | 0.8049 |

## Interpretation

Qwen2.5-1.5B LoRA v3 clearly improves over the same-size JittorLLM zero-shot reranker, indicating that task-specific reranker training provides important gains. It also improves over BM25 and the lightweight Jittor MLP/TextCNN models.

The result remains below the external pretrained Cross-Encoder reference. This should be reported as narrowing the gap, not matching or exceeding the Cross-Encoder.

The 1.5B zero-shot result is close to BM25 overall: some metrics are slightly higher, while Recall@3 is essentially tied and slightly lower. The stronger improvement appears after LoRA task training.

## Reproducibility Artifacts

Committed:

- LoRA training configs under `configs/lora_qwen_1_5b_*.yaml`
- Data cards under `data/processed/lora_qwen_1_5b_*/data_card.md`
- Metrics, tuning summaries, train logs, loss curves, and rankings under `outputs/lora_qwen_1_5b_*`
- Aggregated comparison under `outputs/lora_qwen2_1_5b_comparison.md`

Not committed:

- Base model weights
- LoRA adapter weights
- Hugging Face cache
- Full checkpoint directories
- Optimizer, scheduler, and RNG states
- Large score caches

Model weights and LoRA adapters are excluded from version control. The repository retains reproducible training configurations, scripts, metrics, and validation artifacts.

## Commands

```bash
bash scripts/run_lora_qwen_1_5b_v3_data.sh
bash scripts/run_lora_qwen_1_5b_v3_train.sh
bash scripts/run_lora_qwen_1_5b_v3_eval.sh
python src/aggregate_lora_results.py
```

Equivalent expanded commands:

```bash
python src/lora_reranker/build_lora_data.py \
  --train_in data/processed/msmarco_medium/train.jsonl \
  --valid_in data/processed/msmarco_medium/valid.jsonl \
  --test_in data/processed/msmarco_medium/test.jsonl \
  --out_dir data/processed/lora_qwen_1_5b_10k \
  --max_train_pairs 10000 \
  --max_valid_pairs 1000 \
  --max_test_queries 500 \
  --negatives_per_query 4 \
  --seed 42

python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_10k_lr1e4_s800.yaml

python src/lora_reranker/evaluate_lora_reranker.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/lora_adapters/qwen_1_5b_10k_lr1e4_s800 \
  --test_path data/processed/lora_qwen_1_5b_10k/test_queries.jsonl \
  --output_metrics outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_metrics.json \
  --output_rankings outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_rankings.json \
  --max_queries 500 \
  --max_length 256

python src/aggregate_lora_results.py
```
