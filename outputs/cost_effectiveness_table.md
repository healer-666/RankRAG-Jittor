# Resource and Effectiveness Profile

| Method | R@1 | MRR | NDCG@5 | Training | Pretrained semantics | Model scale | Eval runtime | Peak VRAM | Hardware | Resource comparability |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| BM25 | 0.2300 | 0.4476 | 0.5074 | none | no | N/A | not recorded | not recorded | not recorded | cpu_local_retrieval; strict=false |
| Jittor MLP | 0.2280 | 0.4318 | 0.4698 | from_scratch | no | 0.394M static | not recorded | not recorded | not recorded | jittor_local_training; strict=false |
| Jittor TextCNN | 0.1800 | 0.3912 | 0.4270 | from_scratch | no | 3.899M static | not recorded | not recorded | not recorded | jittor_local_training; strict=false |
| Qwen2.5-1.5B zero-shot reranker | 0.2360 | 0.4525 | 0.5210 | none | yes | 1.5B class | 114.582s (scoring_only) | 3608.0 MiB | not recorded | qwen_zero_shot_recorded_environment; strict=false |
| Qwen2.5-1.5B LoRA 10k-rerun | 0.3560 | 0.5633 | 0.6236 | lora_finetuning | yes | 1.5B class | 444.287s (scoring_only) | 16509.3 MiB | NVIDIA GeForce RTX 4090 D | autodl_4090d_lora; strict=false |
| Cross-Encoder | 0.4340 | 0.6341 | 0.7019 | none_in_this_project | yes | MiniLM-L6 cross-encoder class | 5.955s (scoring_only) | not recorded | cuda | pretrained_cross_encoder_unknown_hardware; strict=false |
