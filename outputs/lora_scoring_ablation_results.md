# LoRA Scoring-Method Ablation Results

Status: ready

| Method | R@1 | R@3 | R@5 | NDCG@5 | MRR | Pairwise Acc. | Runtime s | Parse fail % | Tie rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generate_parse | 0.1280 | 0.4240 | 0.6460 | 0.3848 | 0.3490 | 0.0012 | 904.7539 | 0.0010 | 0.9990 |
| relevant_logprob | 0.3560 | 0.6980 | 0.8700 | 0.6254 | 0.5637 | 0.7358 | 227.4696 | 0.0000 | 0.0023 |
| logprob_margin | 0.3560 | 0.6960 | 0.8660 | 0.6236 | 0.5633 | 0.7343 | 454.0497 | 0.0000 | 0.0030 |
