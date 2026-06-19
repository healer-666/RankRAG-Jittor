# LoRA Reranker Plan

## Purpose

The LoRA reranker is a RankRAG-mini supplementary experiment. It is added to test an LLM-style relevance judgment path:

```text
question + passage -> Relevant / Irrelevant
```

This is useful because the original RankRAG work uses language-model relevance judgment as part of its ranking-generation design. The current project does not reproduce full RankRAG generation, so the LoRA reranker is only a small engineering preview of that direction.

## Relation to the Jittor Reproduction

The LoRA reranker is a RankRAG-mini supplementary experiment.
It does not replace the Jittor reproduction body.
The Jittor reproduction body remains the MLP/TextCNN rerankers.

The Jittor reproduction body is still:

- MLP reranker in PyTorch and Jittor
- TextCNN reranker in PyTorch and Jittor
- MS MARCO small/medium ranking evaluation
- PyTorch/Jittor alignment checks

## Why Debug Only

This stage is a Windows small-scale debug run. It validates the closed loop:

```text
MS MARCO medium processed data
-> instruction pair data
-> Qwen2.5-0.5B-Instruct + LoRA training
-> adapter save/load
-> Relevant / Irrelevant log-prob scoring
-> candidate reranking metrics
```

The target is code correctness and workflow completeness, not final ranking quality.

## Why PyTorch / PEFT

The LoRA experiment uses PyTorch, Transformers, and PEFT because the current LoRA ecosystem for Qwen instruction models is mature there. This does not change the main Jittor reproduction scope. It is a supplementary RankRAG-mini path for future larger experiments.

## Formal Experiment Plan

A later formal run should use a rented GPU environment, preferably 24GB VRAM or more, and run Qwen2.5-1.5B-Instruct or a larger model with a larger MS MARCO subset. The formal run should report metrics separately from the Jittor MLP/TextCNN reproduction results.
