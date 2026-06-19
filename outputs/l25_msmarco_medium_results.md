# L2.5 MS MARCO Medium Results

Cross-Encoder is an external pretrained semantic reranker reference, not the Jittor reproduction body.

| Method | Framework | Training | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@3 | NDCG@5 | Time / throughput | Role |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TF-IDF | sklearn | none | 0.2220 | 0.5700 | 0.7880 | 1.0000 | 0.4465 | 0.4188 | 0.5084 | N/A | lexical baseline |
| BM25 | rank_bm25 | none | 0.2300 | 0.5540 | 0.7840 | 1.0000 | 0.4476 | 0.4127 | 0.5074 | N/A | lexical baseline |
| MLP | PyTorch | from scratch | 0.1920 | 0.4780 | 0.7000 | 1.0000 | 0.4079 | 0.3559 | 0.4475 | N/A | PyTorch lightweight reranker |
| MLP | Jittor | from scratch | 0.2280 | 0.5060 | 0.7120 | 1.0000 | 0.4318 | 0.3853 | 0.4698 | N/A | Jittor lightweight reranker |
| TextCNN | PyTorch | from scratch | 0.1720 | 0.4960 | 0.7220 | 1.0000 | 0.3953 | 0.3534 | 0.4463 | N/A | PyTorch neural reranker |
| TextCNN | Jittor | from scratch | 0.1800 | 0.4500 | 0.6780 | 1.0000 | 0.3912 | 0.3341 | 0.4270 | N/A | Jittor neural reranker |
| Cross-Encoder | sentence-transformers | external pretrained | 0.4340 | 0.8080 | 0.9340 | 1.0000 | 0.6341 | 0.6495 | 0.7019 | 679.06 pairs/s | external pretrained semantic reranker reference |
