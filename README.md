# RankRAG-style Context Ranking in Jittor

## Project Overview

This repository is a lightweight Jittor reproduction of the context ranking / selector idea from **RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs**. It focuses on evidence filtering for academic search enhancement: given a query and a set of candidate paper contexts, the model scores and ranks candidate evidence.

This project does not reproduce full LLM instruction tuning or answer generation. Its goal is to verify a compact ranking pipeline, provide a Jittor implementation, keep a PyTorch baseline, and compare PyTorch/Jittor outputs under the same synthetic benchmark.

## Paper Information

**Paper:** RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs  
**Authors:** Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, Bryan Catanzaro  
**Conference:** NeurIPS 2024  
**Core idea:** the original RankRAG framework unifies context ranking and retrieval-augmented generation so that a model can both rank useful context and generate answers.

## Reproduction Scope

This repository reproduces the context ranking / evidence filtering idea in RankRAG.

- Reproduced: context ranking / selector pipeline.
- Reproduced: PyTorch baseline and Jittor implementation.
- Reproduced: training, evaluation, visualization, and PyTorch/Jittor alignment.
- Not reproduced: full LLM instruction tuning.
- Not reproduced: answer generation.
- Goal: verify ranking pipeline correctness, Jittor implementation, and PyTorch/Jittor metric alignment.

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

The project was completed and verified on Ubuntu 22.04 with CPU mode:

```text
Python: 3.10.20
PyTorch: 2.12.1+cpu
Jittor: 1.3.11.0
Jittor mode: CPU, use_cuda=0
g++: 11.4.0
```

The scripts default to CPU mode for Jittor to avoid automatic CUDA/cuDNN downloads. On this machine, Jittor detects a CUDA driver and may otherwise try to download a 5.61GB CUDA/cuDNN package, which is unnecessary for this lightweight reproduction.

## Installation

Recommended Conda setup:

```bash
conda create -p .venv-jittor python=3.10 -y
conda activate ./.venv-jittor
pip install -r requirements.txt
python -c "import jittor as jt; print(jt.__version__)"
```

If using a plain virtual environment on a machine with `python3-venv` installed:

```bash
python3 -m venv .venv-jittor
source .venv-jittor/bin/activate
pip install -r requirements.txt
python -c "import jittor as jt; print(jt.__version__)"
```

## Data Preparation

```bash
python scripts/prepare_data.py
```

This generates:

- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`
- `data/academic_demo.json`

The current benchmark is a synthetic dataset with hard negatives. It is designed to validate the workflow, not to measure real academic search generalization.

## Training

```bash
bash scripts/run_train_torch.sh
bash scripts/run_train_jittor.sh
```

The Jittor script exports CPU defaults:

```bash
use_cuda=0
nvcc_path=''
```

## Evaluation

```bash
bash scripts/run_eval_torch.sh
bash scripts/run_eval_jittor.sh
```

## Alignment and Visualization

```bash
python src/compare_results.py
python src/plot_results.py
python scripts/check_project_ready.py
```

Expected ready status:

```text
PyTorch baseline: ready
Jittor skeleton: ready
Jittor training: ready
Visualization: ready
README: ready
```

## Results

Current PyTorch/Jittor alignment results:

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

Training logs:

- `logs/torch_train.log`
- `logs/jittor_train.log`

Performance and alignment outputs:

- `outputs/torch_metrics.json`
- `outputs/jittor_metrics.json`
- `outputs/metrics_compare.json`
- `outputs/metrics_compare.md`
- `outputs/demo_ranking_result_torch.json`
- `outputs/demo_ranking_result_jittor.json`

Visualizations:

- `outputs/loss_curve.png`
- `outputs/metrics_compare.png`

## Result Boundary

Current metrics are all 1.0 mainly because the synthetic benchmark validates training, ranking, evaluation, and PyTorch/Jittor alignment. Since the data is template-generated, even with hard negatives it still contains lexical and structural regularities. Therefore, these results should not be interpreted as generalization ability on real open academic search tasks.

## File Structure

```text
configs/   Configuration files
data/      Synthetic data and academic demo
scripts/   Data preparation, training, evaluation, and readiness scripts
src/       PyTorch/Jittor models, training, evaluation, metrics, and plotting code
docs/      Method summary, result analysis, and Jittor setup notes
logs/      Training logs
outputs/   Metrics, figures, and demo ranking results
```

## Main Artifacts

- `configs/default.yaml`
- `scripts/prepare_data.py`
- `scripts/run_train_torch.sh`
- `scripts/run_train_jittor.sh`
- `scripts/run_eval_torch.sh`
- `scripts/run_eval_jittor.sh`
- `scripts/check_project_ready.py`
- `src/model_torch.py`
- `src/model_jittor.py`
- `src/train_torch.py`
- `src/train_jittor.py`
- `src/eval_torch.py`
- `src/eval_jittor.py`
- `src/metrics.py`
- `src/compare_results.py`
- `src/plot_results.py`
- `docs/method_summary.md`
- `docs/result_analysis.md`
- `docs/jittor_setup.md`

## Future Work

- Add a more authoritative public benchmark, such as an MS MARCO passage ranking small subset.
- Add BEIR-SciFact for scientific fact/paper relevance evaluation.
- Reduce template overlap between train/valid/test.
- Add richer academic query types and cross-topic hard negatives.
- Compare CPU/GPU Jittor behavior if a controlled CUDA environment is available.
