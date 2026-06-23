# LoRA Data-Size Ablation Results

Status: ready

| Run | Pairs | Micro batch | Grad accum | Global batch | Eff. epochs | R@1 | R@3 | R@5 | NDCG@5 | MRR | Pairwise Acc. | Train s | Eval s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 1000 | 1 | 8 | 8 | 6.4000 | 0.2880 | 0.6680 | 0.8520 | 0.5808 | 0.5125 | 0.6934 | 1332.1103 | 460.5405 |
| 3k | 3000 | 1 | 8 | 8 | 2.1333 | 0.3620 | 0.6920 | 0.8540 | 0.6187 | 0.5627 | 0.7240 | 1271.9508 | 453.9075 |
| 10k-rerun | 10000 | 8 | 1 | 8 | 0.6400 | 0.3560 | 0.6960 | 0.8660 | 0.6236 | 0.5633 | 0.7343 | 187.8953 | 444.2874 |

Historical 10k reference, not part of the E1 trend line:

- R@1: 0.3580; R@5: 0.8720; NDCG@5: 0.6266; MRR: 0.5642
