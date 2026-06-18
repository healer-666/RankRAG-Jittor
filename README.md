# RankRAG-style Context Ranking in Jittor

## Project Overview

This repository is a lightweight Jittor reproduction of the context ranking / selector idea from **RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs**. It focuses on evidence filtering: given a query and several candidate passages, the model scores and ranks candidate evidence.

This is not a full reproduction of RankRAG. It does not reproduce LLM instruction tuning or answer generation. The goal is to validate a compact context ranking pipeline, implement it in Jittor, keep a PyTorch baseline, and compare PyTorch/Jittor results on both a smoke-test synthetic benchmark and a public MS MARCO small subset.

## Paper Information

**Paper:** RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs  
**Authors:** Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, Bryan Catanzaro  
**Conference:** NeurIPS 2024  
**Core idea:** the original RankRAG framework unifies context ranking and retrieval-augmented generation so that one model can rank useful context and generate answers.

## Reproduction Scope

- Reproduced: RankRAG-style context ranking / evidence filtering.
- Reproduced: PyTorch baseline and Jittor implementation.
- Reproduced: training, evaluation, visualization, and PyTorch/Jittor alignment.
- Not reproduced: full LLM instruction tuning.
- Not reproduced: answer generation.
- Not equivalent to: RankRAG's full LLM-based generation pipeline.

The current scorer is deliberately lightweight, so its results should not be over-claimed as full RankRAG performance.

## Method

The implemented ranking pipeline is:

```text
query + candidate context
-> fixed text features
-> MLP scorer
-> pairwise ranking loss
-> relevance score
```

Feature dimension is `1537`: `q_emb, c_emb, abs(q-c), q*c, cosine`, where the HashingVectorizer embedding dimension is `384`, so `384 * 4 + 1 = 1537`.

## Environment

Verified on Ubuntu 22.04 with CPU mode:

```text
Python: 3.10.20
PyTorch: 2.12.1+cpu
Jittor: 1.3.11.0
Jittor mode: CPU, use_cuda=0
g++: 11.4.0
```

The Jittor scripts default to CPU mode to avoid automatic CUDA/cuDNN downloads. On this machine, Jittor can detect a CUDA driver and may otherwise try to download a 5.61GB CUDA/cuDNN package, which is unnecessary for this lightweight reproduction.

## Installation

```bash
conda create -p .venv-jittor python=3.10 -y
conda activate ./.venv-jittor
pip install -r requirements.txt
python -c "import jittor as jt; print(jt.__version__)"
```

If the environment uses a SOCKS proxy for Hugging Face downloads, `socksio` is included in `requirements.txt`.

## Experiments

### 1. Synthetic hard-negative benchmark

The synthetic benchmark is kept as a smoke test and pipeline correctness check. It verifies data generation, feature extraction, pairwise ranking loss, PyTorch/Jittor training, evaluation, and visualization.

It is no longer the main experiment because the data is template-generated and contains lexical/template regularities.

```bash
python scripts/prepare_data.py
bash scripts/run_train_torch.sh
bash scripts/run_eval_torch.sh
bash scripts/run_train_jittor.sh
bash scripts/run_eval_jittor.sh
python src/compare_results.py
python src/plot_results.py
```

### 2. MS MARCO small subset

MS MARCO small subset is the current main public ranking-data experiment. It is closer to the RankRAG context ranking setup than the synthetic benchmark because it uses real query-passage data.

Prepare data:

```bash
python scripts/prepare_msmarco_subset.py \
  --max_train_queries 1000 \
  --max_valid_queries 200 \
  --max_test_queries 200 \
  --candidates_per_query 5 \
  --seed 42
```

Run experiments:

```bash
bash scripts/run_train_torch_msmarco.sh
bash scripts/run_eval_torch_msmarco.sh
bash scripts/run_train_jittor_msmarco.sh
bash scripts/run_eval_jittor_msmarco.sh
python src/compare_results.py --run_name msmarco
python src/plot_results.py --run_name msmarco
```

Readiness check:

```bash
python scripts/check_project_ready.py
```

### 3. Academic demo

`data/academic_demo.json` is kept as a small Paper-Skill academic search demo. It shows how candidate evidence passages can be ranked for a specific academic query. It is for demonstration, not a public benchmark.

## MS MARCO Small Subset

Source: `microsoft/ms_marco`, config `v1.1`.

Subset size:

| Split | Queries | Avg candidates | Positives | Negatives |
| --- | ---: | ---: | ---: | ---: |
| train | 1000 | 4.97 | 1000 | 3968 |
| valid | 200 | 5.00 | 200 | 799 |
| test | 200 | 4.99 | 200 | 798 |

The subset is small by design. It is not the full MS MARCO dataset and is not a leaderboard submission setting.

## Current Main Results: MS MARCO

| Metric | PyTorch | Jittor | Diff |
| --- | ---: | ---: | ---: |
| recall@1 | 0.2600 | 0.2450 | -0.0150 |
| ndcg@1 | 0.2600 | 0.2450 | -0.0150 |
| recall@3 | 0.6500 | 0.6700 | 0.0200 |
| ndcg@3 | 0.4812 | 0.4863 | 0.0051 |
| recall@5 | 1.0000 | 1.0000 | 0.0000 |
| ndcg@5 | 0.6251 | 0.6214 | -0.0037 |
| mrr | 0.5031 | 0.4978 | -0.0053 |
| pairwise_accuracy | 0.5529 | 0.5546 | 0.0017 |

The MS MARCO scores are much lower than the synthetic scores, which is expected: real query-passage ranking is harder and less template-like. The PyTorch/Jittor metrics are close, indicating that the two implementations are reasonably aligned.

## Synthetic Smoke-Test Results

| Metric | PyTorch | Jittor | Diff |
| --- | ---: | ---: | ---: |
| recall@1 | 1.0000 | 1.0000 | 0.0000 |
| ndcg@1 | 1.0000 | 1.0000 | 0.0000 |
| recall@3 | 1.0000 | 1.0000 | 0.0000 |
| ndcg@3 | 1.0000 | 1.0000 | 0.0000 |
| recall@5 | 1.0000 | 1.0000 | 0.0000 |
| ndcg@5 | 1.0000 | 1.0000 | 0.0000 |
| mrr | 1.0000 | 1.0000 | 0.0000 |
| pairwise_accuracy | 1.0000 | 1.0000 | 0.0000 |

These synthetic metrics mainly prove that the training, ranking, evaluation, and PyTorch/Jittor alignment workflow is correct. They should not be interpreted as real open academic search generalization.

## Outputs

MS MARCO outputs:

- `data/processed/msmarco/dataset_card.md`
- `logs/msmarco_torch_train.log`
- `logs/msmarco_jittor_train.log`
- `outputs/msmarco_torch_metrics.json`
- `outputs/msmarco_jittor_metrics.json`
- `outputs/msmarco_metrics_compare.json`
- `outputs/msmarco_metrics_compare.md`
- `outputs/msmarco_loss_curve.png`
- `outputs/msmarco_metrics_compare.png`
- `outputs/msmarco_demo_ranking_result_torch.json`
- `outputs/msmarco_demo_ranking_result_jittor.json`

Synthetic outputs:

- `logs/torch_train.log`
- `logs/jittor_train.log`
- `outputs/torch_metrics.json`
- `outputs/jittor_metrics.json`
- `outputs/metrics_compare.json`
- `outputs/metrics_compare.md`
- `outputs/loss_curve.png`
- `outputs/metrics_compare.png`
- `outputs/demo_ranking_result_torch.json`
- `outputs/demo_ranking_result_jittor.json`

## Result Boundary

This project should be read as a lightweight RankRAG-style context ranking reproduction in Jittor. MS MARCO small subset results are more meaningful than the synthetic smoke-test, but they still use a small subset and a compact MLP scorer. The project does not claim full RankRAG reproduction, LLM generation quality, or real academic search generalization.

## File Structure

```text
configs/   Configuration files
data/      Synthetic data, MS MARCO small subset, and academic demo
scripts/   Data preparation, training, evaluation, and readiness scripts
src/       PyTorch/Jittor models, training, evaluation, metrics, and plotting code
docs/      Method summary, result analysis, and Jittor setup notes
logs/      Training logs
outputs/   Metrics, figures, and demo ranking results
```

## Future Work

- Add a more standard MS MARCO passage ranking evaluation setup.
- Add BEIR-SciFact for scientific fact/paper relevance evaluation.
- Replace the fixed-feature MLP with a stronger neural reranker.
- Evaluate larger subsets while keeping storage and runtime manageable.
- Keep the scope focused on RankRAG-style context ranking unless full LLM instruction tuning is explicitly added.
