<div align="center">

# RankRAG-Jittor

**A Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.**

[简体中文](README.md) · [English](README.en.md) · [Full Results](docs/final_results.md) · [Reproduction](docs/reproduction.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Jittor](https://img.shields.io/badge/Jittor-alignment-2563EB)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA%20reranking-EE4C2C)
[![RankRAG Paper](https://img.shields.io/badge/RankRAG-NeurIPS%202024-7C3AED)](https://arxiv.org/abs/2407.02485)

<img src="docs/figures/rankrag_jittor_overview.svg" alt="RankRAG-Jittor overview" width="940">

</div>

## Project Overview

This project studies a practical question: before giving passages to a large language model, how can we rank the genuinely useful evidence higher? A RankRAG-style pipeline retrieves candidate passages, reranks query-passage pairs by relevance, passes top evidence to a generator, and then checks whether better ranking improves final answers.

The repository has three layers. First, it reproduces lightweight PyTorch/Jittor MLP and TextCNN rerankers for framework alignment. Second, it evaluates Qwen2.5-1.5B zero-shot and LoRA rerankers as RankRAG-style LLM relevance judges. Third, it adds downstream RAG validation, LoRA data-size ablation, scoring-method ablation, error taxonomy, and a resource-effectiveness profile.

This is not a full reproduction of every component or benchmark in the RankRAG paper. It focuses on a reproducible empirical path: LLM relevance judgment -> reranking -> downstream validation.

## Why RankRAG, and Where Jittor Is Used

RankRAG is a useful target for a course-style reproduction project because it separates retrieval quality from evidence ranking quality. The key question is not only whether a relevant passage exists in the candidate pool, but whether it is ranked high enough and then used by the generator.

Jittor is used for lightweight reranker reproduction and framework alignment, not for training a full LLM from scratch:

| Track | Framework | Role |
| --- | --- | --- |
| MLP / TextCNN | PyTorch + Jittor | Confirm that the two implementation paths have aligned overall trends and similar result levels |
| Qwen zero-shot | JittorLLM | Validate LLM relevance judgment without local task training |
| Qwen LoRA | PyTorch + Transformers + PEFT | Main RankRAG-style task-adapted LLM reranker |
| Cross-Encoder | sentence-transformers / PyTorch | External pretrained effectiveness reference |

This split keeps the Jittor reproduction requirement visible while avoiding an overclaim that the project rewrites the full RankRAG system.

## Pipeline

The overview diagram represents a parallel comparison pool. Candidate passages enter the compared-reranker panel; BM25, PyTorch/Jittor lightweight baselines, Qwen zero-shot, Qwen LoRA, and the Cross-Encoder each produce their own ranking; all methods then output ranked passages. The diagram does not imply a serial dependency such as PyTorch MLP -> Jittor MLP or Qwen zero-shot -> LoRA -> Cross-Encoder.

## Completed Work

| Module | Completed work |
| --- | --- |
| Data and candidates | Built the MS MARCO medium subset with 500 queries and 4,044 query-passage pairs |
| Lightweight baselines | Evaluated BM25, MLP, and TextCNN rerankers |
| Jittor alignment | Ported MLP/TextCNN paths to Jittor and compared them with PyTorch under the same protocol |
| LLM reranking | Evaluated Qwen2.5-1.5B zero-shot relevance scoring and LoRA reranking |
| LoRA ablation | Prepared nested 1k / 3k / 10k training pairs with a unified 800-step setting |
| Scoring ablation | Compared generate_parse, relevant_logprob, and logprob_margin scoring |
| Downstream RAG | Checked whether stronger top-k evidence improves generated answers |
| Diagnostics | Built a 30-case error taxonomy and a resource-effectiveness profile |
| Release cleanup | Consolidated final summaries, README figures, reproduction docs, and repository checks |

## Experiment Environment

| Part | Environment note |
| --- | --- |
| Local development | Windows + Python for data processing, checks, documentation, and lightweight experiments |
| Jittor path | Jittor/JittorLLM for lightweight alignment and Qwen zero-shot validation |
| Formal LoRA rerun | 10k LoRA rerun completed in a unified RTX 4090 D environment with PyTorch + Transformers + PEFT |
| Resource metadata | 10k-rerun training time 187.8953 s, evaluation time 444.2874 s, peak GPU memory 16,509.3 MiB |

The resource profile does not treat heterogeneous hardware records as a strict speed benchmark. Effectiveness metrics are comparable; resource fields are provenance and profiling metadata.

## Dataset and Evaluation

| Item | Setting |
| --- | --- |
| Dataset | MS MARCO medium subset |
| Main evaluation size | 500 queries, 4,044 query-passage pairs |
| Candidate pool | Shared by all main reranking methods |
| Metrics | Recall@1/3/5, NDCG@5, MRR, pairwise accuracy |
| Downstream RAG | 50 questions comparing BM25, LoRA, and Cross-Encoder evidence |
| Error analysis | 30 stratified diagnostic queries, not an unbiased full-distribution estimate |

## PyTorch / Jittor Alignment

<img src="docs/figures/pytorch_jittor_alignment.svg" alt="PyTorch and Jittor alignment baselines" width="940">

MLP and TextCNN are lightweight alignment baselines. They confirm that the two implementation paths have aligned overall trends and similar result levels; they are not the core LLM reranker from the RankRAG paper.

| Model | PyTorch R@1 | Jittor R@1 | Reading |
| --- | ---: | ---: | --- |
| MLP | 0.192 | 0.228 | Similar trend, Jittor slightly higher |
| TextCNN | 0.172 | 0.180 | Similar trend, very small gap |

## LLM Reranking and LoRA Reproduction

The Qwen2.5-1.5B track has two stages. First, the model scores query-passage relevance in a zero-shot setting. Second, LoRA adapts the model on nested 1k, 3k, and 10k training pairs and evaluates the resulting rankings on the same candidate pool.

| Field | Setting |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA rank | 8 |
| Learning rate | 1e-4 |
| Max steps | 800 |
| Max length | 256 |
| Main scoring method | `logprob_margin` |
| Formal 10k output | New 10k-rerun output directory, without overwriting the historical 10k directory |

The historical directory `outputs/lora_qwen_1_5b_10k_lr1e4_s800/` is retained as a reference. The formal E1 comparison uses 1k, 3k, and 10k-rerun results from the unified RTX 4090 D environment.

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

Key conclusions:

- BM25 remains a strong lexical baseline on this subset.
- Qwen LoRA improves the same Qwen2.5-1.5B reranker from 0.236 R@1 in zero-shot mode to 0.356 R@1 after task adaptation.
- Cross-Encoder remains the strongest external effectiveness reference in this test, reaching 0.434 R@1.
- LoRA validates the LLM reranking path; it is not a claim of surpassing a specialized pretrained Cross-Encoder.

Full R@3, R@5, pairwise accuracy, runtime metadata, and source artifact paths are in [docs/final_results.md](docs/final_results.md).

## Ablations, Downstream RAG, Error Analysis, and Resources

<img src="docs/figures/readme_lora_ablation.svg" alt="LoRA data-size ablation" width="900">

The data-size ablation compares nested 1k, 3k, and 10k training pairs under a fixed 800-step budget. The 10k-rerun has the highest NDCG@5 and pairwise accuracy, while the 3k run is slightly higher on R@1.

<img src="docs/figures/readme_error_taxonomy.svg" alt="Error taxonomy" width="900">

The error taxonomy separates lexical traps, small-model semantic limits, evidence-utilization failures, candidate or label issues, and ambiguous queries. It is a 30-case diagnostic analysis, not an unbiased estimate over all 500 queries.

<img src="docs/figures/readme_resource_profile.svg" alt="Resource-effectiveness profile" width="900">

The resource profile keeps effectiveness comparison and hardware provenance separate. The candidate pool and metrics are shared, but training strategy, hardware records, and runtime environments are heterogeneous.

Detailed reports:

| Analysis | Document |
| --- | --- |
| LoRA data-size ablation | [docs/ablation_analysis.md](docs/ablation_analysis.md) |
| Scoring-method ablation | [docs/scoring_ablation_analysis.md](docs/scoring_ablation_analysis.md) |
| Downstream RAG | [docs/downstream_rag_analysis.md](docs/downstream_rag_analysis.md) |
| Error taxonomy | [docs/error_taxonomy.md](docs/error_taxonomy.md) |
| Resource-effectiveness profile | [docs/cost_effectiveness_analysis.md](docs/cost_effectiveness_analysis.md) |

## Quick Start

The following commands rebuild summaries and README figures from existing artifacts only. They do not train, run inference, download model weights, or regenerate rankings.

```bash
git clone https://github.com/healer-666/RankRAG-Jittor.git
cd RankRAG-Jittor
pip install -r requirements.txt
python scripts/build_final_project_summary.py
python scripts/build_readme_figures.py
python scripts/check_final_repository.py
```

Common entry points:

| Goal | Command or document |
| --- | --- |
| Rebuild final result summary | `python scripts/build_final_project_summary.py` |
| Rebuild README figures | `python scripts/build_readme_figures.py` |
| Check final repository state | `python scripts/check_final_repository.py` |
| See full reproduction commands | [docs/reproduction.md](docs/reproduction.md) |

## Repository Structure and Documentation

| Path | Purpose |
| --- | --- |
| `configs/` | Experiment and evaluation configs |
| `data/processed/` | Processed MS MARCO subsets and LoRA pair files |
| `src/` | Retrieval, reranking, evaluation, and aggregation code |
| `scripts/` | Data builders, checks, summaries, figures, and analysis scripts |
| `outputs/` | Committed metrics, rankings, summaries, and analysis figures |
| `logs/` | Historical run logs and hardware monitoring traces |
| `docs/` | Reproduction, result, ablation, error, and resource reports |
| `docs/figures/` | README and report figures |

| Document | Contents |
| --- | --- |
| [docs/final_results.md](docs/final_results.md) | Final metric tables and evidence paths |
| [docs/reproduction.md](docs/reproduction.md) | Environment, data preparation, and reproduction commands |
| [docs/final_repository_audit.md](docs/final_repository_audit.md) | Repository integrity audit |

## Reproduction Boundary and Limitations

- This is a lightweight reproduction and empirical analysis, not a complete RankRAG reproduction.
- The main ranking evaluation uses a 500-query MS MARCO medium subset.
- Downstream RAG uses 50 questions.
- Error analysis uses 30 stratified diagnostic cases.
- Resource records come from heterogeneous environments and should not be read as a strict speed benchmark.
- Model weights, LoRA adapters, checkpoints, and local caches are not versioned in Git.

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

## License Status

No `LICENSE` file is currently included. Until a formal license is added, use the code and artifacts according to the repository owner's terms.
