# Stage E2: LoRA Scoring-Method Ablation

E2 compares inference-time scoring rules for the same 10k-rerun LoRA adapter. It is not a new model-training ablation.

## Methods

- `generate_parse`: deterministic generation of Relevant / Irrelevant, parsed into binary scores.
- `relevant_logprob`: score = log P(Relevant).
- `logprob_margin`: score = log P(Relevant) - log P(Irrelevant), the backward-compatible default.

## Observations

- generate_parse tie rate: 0.9990; parse failure rate: 0.0010.
- relevant_logprob R@1 delta vs margin: +0.0000.
- logprob_margin R@1: 0.3560; MRR: 0.5633; NDCG@5: 0.6236.

The log-probability methods compute the full label token sequence probability rather than only the first token.

## Label Tokenization

- Relevant: ids=[693, 8367], length=2
- Irrelevant: ids=[48113, 97573], length=2
