<div align="center">

# RankRAG-Jittor

**A Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.**

[English](README.md) · [简体中文](README.zh-CN.md) · [Results](docs/final_results.md) · [Reproduction](docs/reproduction.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Jittor](https://img.shields.io/badge/Jittor-alignment-2563EB)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA%20reranking-EE4C2C)
[![RankRAG Paper](https://img.shields.io/badge/RankRAG-NeurIPS%202024-7C3AED)](https://arxiv.org/abs/2407.02485)

<img src="docs/figures/rankrag_jittor_overview.svg" alt="RankRAG-Jittor overview" width="940">

</div>

## Why RankRAG-Jittor?

RankRAG-style systems retrieve a candidate pool, rerank passages by relevance to the question, pass the strongest evidence to an LLM, and then evaluate whether better evidence actually improves answers. This repository studies that chain on an MS MARCO medium subset with lightweight Jittor baselines, Qwen2.5 LLM rerankers, LoRA adaptation, downstream RAG validation, error analysis, and resource profiling.

The project is intentionally scoped: it is a reproducible empirical reproduction path, not a full reimplementation of every component or benchmark in the original RankRAG paper.

## At a Glance

| Item | Scope |
| --- | --- |
| Evaluation set | 500 queries, 4,044 query-passage pairs |
| Main methods | BM25, Jittor MLP, Jittor TextCNN, Qwen zero-shot, Qwen LoRA, Cross-Encoder reference |
| Framework alignment | PyTorch and Jittor MLP/TextCNN rerankers on the same data and metrics |
| LLM reranking | Qwen2.5-1.5B zero-shot scoring and Qwen2.5-1.5B LoRA relevance scoring |
| Ablations | 1k / 3k / 10k LoRA data-size ablation and scoring-method ablation |
| Diagnostics | 50-question downstream RAG checks, 30-case error taxonomy, resource-effectiveness profile |

## Main Results

<img src="docs/figures/main_reranking_results.svg" alt="Reranking effectiveness on MS MARCO Medium" width="940">

| Method | R@1 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.230 | 0.5074 | 0.4476 |
| Jittor MLP | 0.228 | 0.4698 | 0.4318 |
| Jittor TextCNN | 0.180 | 0.4270 | 0.3912 |
| Qwen2.5-1.5B Zero-shot | 0.236 | 0.5210 | 0.4525 |
| Qwen2.5-1.5B LoRA (10k pairs) | 0.356 | 0.6236 | 0.5633 |
| Cross-Encoder Reference | 0.434 | 0.7019 | 0.6341 |

The LoRA row uses the formal same-environment 10k LoRA rerun. Full R@3, R@5, pairwise accuracy, runtime metadata, and source paths are in [docs/final_results.md](docs/final_results.md).

Three takeaways matter most:

- BM25 remains a strong lexical baseline on this subset.
- Qwen LoRA substantially improves over Qwen zero-shot, showing the value of task-adapted RankRAG-style relevance judgment.
- The Cross-Encoder remains the strongest external effectiveness reference; LoRA is a reproduction path for LLM-based reranking, not a claim of beating a specialized reranker.

## What Is Reproduced?

### PyTorch to Jittor Alignment

MLP and TextCNN are lightweight alignment baselines. They test whether the PyTorch and Jittor paths behave coherently on the same reranking data, losses, and metrics. Their role is framework migration and sanity checking, not matching the RankRAG paper's core LLM reranker.

### RankRAG-Style LLM Reranking

The LLM track uses Qwen2.5-1.5B to score query-passage relevance, first as a zero-shot reranker and then through LoRA fine-tuning. The output is a passage ranking used to choose top-k evidence for downstream generation.

### End-to-End Validation

The repository also checks whether ranking improvements survive contact with downstream answer generation, scoring choices, data-size changes, resource measurements, and qualitative error modes.

| Component | Runtime / framework | Role |
| --- | --- | --- |
| MLP / TextCNN | PyTorch + Jittor | Framework alignment baselines |
| Qwen zero-shot | JittorLLM | Semantic reranking without local task training |
| Qwen LoRA | PyTorch + Transformers + PEFT | RankRAG-style LLM reranking target |
| Cross-Encoder | sentence-transformers / PyTorch | External pretrained effectiveness reference |

## PyTorch/Jittor Alignment

<img src="docs/figures/pytorch_jittor_alignment.svg" alt="PyTorch and Jittor alignment baselines" width="940">

The alignment figure is not a leaderboard. It confirms that the lightweight Jittor implementations are in the same empirical neighborhood as their PyTorch counterparts under the same data split and metric definitions.

## Beyond Headline Metrics

<img src="docs/figures/readme_lora_ablation.svg" alt="LoRA data-size ablation" width="900">

Data-size ablation compares nested 1k, 3k, and 10k LoRA training pairs under a fixed 800-step budget.

<img src="docs/figures/readme_error_taxonomy.svg" alt="Error taxonomy" width="900">

Error analysis separates lexical traps, semantic limits, evidence-use failures, label issues, and ambiguous queries.

<img src="docs/figures/readme_resource_profile.svg" alt="Resource-effectiveness profile" width="900">

Resource profiling records effectiveness and runtime metadata without treating heterogeneous hardware as a strict speed benchmark.

The deeper analyses show where the main ranking table is too compressed:

- Scoring ablation separates model quality from the rule used to turn LLM outputs into ranking scores.
- Downstream RAG checks whether stronger top-k evidence actually improves generated answers.

## Key Findings

- Lexical matching is still competitive when candidate passages share surface terms with the query.
- Pretrained semantic ability helps: Qwen zero-shot and Cross-Encoder both outperform the lightweight from-scratch Jittor baselines.
- Task adaptation helps more: Qwen LoRA improves substantially over Qwen zero-shot on the same candidate pool.
- The Cross-Encoder is the strongest reference method, which is expected for a specialized pretrained reranker.
- Better ranking increases the chance that correct evidence reaches the generator, but answer generation can still fail.

## Quick Start

```bash
git clone https://github.com/healer-666/RankRAG-Jittor.git
cd RankRAG-Jittor
pip install -r requirements.txt
python scripts/build_final_project_summary.py
python scripts/build_readme_figures.py
python scripts/check_final_repository.py
```

The commands above rebuild summaries and figures from existing artifacts only. They do not train, run inference, download model weights, or regenerate rankings. Full environment notes and optional experiment commands are in [docs/reproduction.md](docs/reproduction.md).

## Repository Structure

| Path | Purpose |
| --- | --- |
| `configs/` | Experiment and evaluation configs |
| `data/processed/` | Processed MS MARCO subsets and LoRA pair files |
| `src/` | Retrieval, reranking, evaluation, and aggregation code |
| `scripts/` | Reproducibility checks, summaries, figures, and analysis scripts |
| `outputs/` | Committed metrics, rankings, summaries, and analysis figures |
| `logs/` | Historical run logs and hardware monitoring traces |
| `docs/` | Detailed reproduction, result, ablation, error, and resource reports |
| `docs/figures/` | README and report figures |

## Documentation

| Document | Use it for |
| --- | --- |
| [docs/final_results.md](docs/final_results.md) | Full metric tables and source-artifact references |
| [docs/reproduction.md](docs/reproduction.md) | Environment setup, data preparation, and reproducibility commands |
| [docs/ablation_analysis.md](docs/ablation_analysis.md) | LoRA data-size ablation |
| [docs/scoring_ablation_analysis.md](docs/scoring_ablation_analysis.md) | LoRA scoring-method ablation |
| [docs/downstream_rag_analysis.md](docs/downstream_rag_analysis.md) | Downstream RAG evaluation |
| [docs/error_taxonomy.md](docs/error_taxonomy.md) | Error-type definitions and diagnostic cases |
| [docs/cost_effectiveness_analysis.md](docs/cost_effectiveness_analysis.md) | Resource and effectiveness profile |
| [docs/final_repository_audit.md](docs/final_repository_audit.md) | Final repository integrity audit |

## Reproduction Boundary

This repository is a lightweight reproduction and empirical analysis. It does not include Llama 3 experiments, complete joint RankRAG training, all original paper benchmarks, or versioned model weights/adapters. The main ranking evaluation uses an MS MARCO medium subset with 500 queries and 4,044 query-passage pairs. The Cross-Encoder is an external pretrained reference. Resource measurements come from heterogeneous environments, so they should be read as provenance and profiling metadata rather than a strict speed benchmark.

## Citation

RankRAG-Jittor is based on the RankRAG paper:

- arXiv: [2407.02485](https://arxiv.org/abs/2407.02485)
- OpenReview: [RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs](https://openreview.net/forum?id=S1fc92uemC)

```bibtex
@inproceedings{yu2024rankrag,
  title = {RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs},
  author = {Yu, Yue and Ping, Wei and Liu, Zihan and Wang, Boxin and You, Jiaxuan and Zhang, Chao and Shoeybi, Mohammad and Catanzaro, Bryan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```

## License

No repository license file is currently included. Treat the code and artifacts according to the repository owner's terms until a license is added.
