# LoRA Reranker Debug Report

Status: completed Windows small-scale debug run.

## Environment

| Item | Value |
| --- | --- |
| Python | 3.10.18 |
| torch | 2.8.0+cu126 |
| CUDA available | True |
| CUDA version | 12.6 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.00 GB |

## Model

| Item | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Adapter | outputs/lora_adapters/qwen_0_5b_debug |
| Trainable parameters | 2,199,552 |

## Data

| Split | Count |
| --- | ---: |
| Train pairs | 200 |
| Valid pairs | 100 |
| Test queries | 20 |
| Test candidate pairs | 158 |

## Training

| Item | Value |
| --- | --- |
| Epochs | 1 debug pass |
| Steps | 20 |
| Max length | 192 |
| LoRA r | 4 |
| Loss start | 0.156637 |
| Loss end | 0.058918 |
| Loss decreased | True |

## Evaluation

| Metric | Value |
| --- | ---: |
| R@1 | 0.150000 |
| R@3 | 0.550000 |
| R@5 | 0.900000 |
| R@10 | 1.000000 |
| MRR | 0.400417 |
| NDCG@3 | 0.369639 |
| NDCG@5 | 0.511612 |
| NDCG@10 | 0.545195 |
| Pairwise accuracy | 0.646726 |
| Inference time sec | 57.662612 |
| Pairs per second | 2.740077 |

## Conclusion

The LoRA reranker debug pipeline ran end to end on Windows with the 6GB RTX 3060 Laptop GPU. The run validated data construction, LoRA training, adapter save/load, and log-probability reranking evaluation.

This result is not a formal performance claim. The dataset, step count, sequence length, and LoRA rank were reduced to fit the local Windows debug environment. A formal RankRAG-mini LoRA experiment should be run later on a rented 24GB GPU with Qwen2.5-1.5B-Instruct or a larger model.
