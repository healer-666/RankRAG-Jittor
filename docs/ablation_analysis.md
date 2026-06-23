# Stage E1: LoRA Training Data-Size Ablation

E1 fixes the optimizer-step budget at 800 steps. Smaller datasets are revisited more often, while larger datasets expose the model to more distinct training pairs.

All runs keep the same global batch size and the same 6400 training samples seen at 800 optimizer steps. The 10k-rerun uses a larger micro-batch on the AutoDL RTX 4090 D (`per_device_train_batch_size=8`, `gradient_accumulation_steps=1`) to reduce rented-GPU time; 1k and 3k use the original `1 x 8` accumulation schedule.

## Marginal Changes

- 1k to 3k R@1: +0.0740; MRR: +0.0502; NDCG@5: +0.0379.
- 3k to 10k-rerun R@1: -0.0060; MRR: +0.0006; NDCG@5: +0.0050.

Pairwise accuracy and ranking metrics should be read together; disagreement indicates that better pair-level separation did not necessarily improve top-ranked evidence.

## E3 Scope Control

Stage E deliberately skips a new top-k downstream RAG ablation. The project already includes BM25 / LoRA / Cross-Encoder downstream answer comparisons and a Qwen2.5-1.5B / 7B by Original / Strict Prompt 2x2 experiment. Adding more top-k sweeps on the same 50 questions would increase repeated test-set use with limited incremental value, so Stage E focuses on reranker-side E1 and E2.

## Historical 10k Reference

The historical 10k result is shown only as a reproducibility reference. It does not replace the 10k-rerun point in the E1 trend line, and runtime is not directly comparable because the historical adapter metadata and environment are incomplete.
