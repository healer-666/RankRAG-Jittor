# Qwen2.5-1.5B LoRA Reranker Comparison

Metrics are read from the committed JSON result files. LoRA adapter weights and base model weights are excluded from version control.

| Method | Framework | Training | R@1 | R@3 | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR | Pairwise Acc. | Throughput | Role |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BM25 | rank_bm25 | none | 0.2300 | 0.5540 | 0.7840 | 1.0000 | 0.5074 | 0.5791 | 0.4476 | 0.6253 | N/A | lexical baseline |
| Jittor MLP | Jittor | from scratch | 0.2280 | 0.5060 | 0.7120 | 1.0000 | 0.4698 | 0.5657 | 0.4318 | 0.5901 | N/A | lightweight Jittor reranker |
| Jittor TextCNN | Jittor | from scratch | 0.1800 | 0.4500 | 0.6780 | 1.0000 | 0.4270 | 0.5341 | 0.3912 | 0.5484 | N/A | lightweight Jittor reranker |
| Qwen2.5-0.5B Jittor zero-shot | JittorLLM | zero-shot | 0.1660 | 0.4540 | 0.6760 | 1.0000 | 0.4189 | 0.5258 | 0.3804 | 0.5367 | 31.06 pairs/s | LLM zero-shot reranker |
| Qwen2.5-1.5B Jittor zero-shot | JittorLLM | zero-shot | 0.2360 | 0.5520 | 0.8120 | 1.0000 | 0.5210 | 0.5832 | 0.4525 | 0.6342 | 35.29 pairs/s | LLM zero-shot reranker |
| Qwen2.5-1.5B LoRA v3 | PyTorch | LoRA, 10k pairs, lr=1e-4 | 0.3580 | 0.6980 | 0.8720 | 1.0000 | 0.6266 | 0.6698 | 0.5642 | 0.7345 | 11.92 pairs/s | formal result |
| Cross-Encoder reference | sentence-transformers | external pretrained | 0.4340 | 0.8080 | 0.9340 | 1.0000 | 0.7019 | 0.7242 | 0.6341 | 0.8049 | 679.06 pairs/s | external semantic reference |
