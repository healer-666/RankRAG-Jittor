# RankRAG-style Context Ranking in Jittor

## Project Overview

This repository is a lightweight Jittor reproduction of the context ranking / selector idea from **RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs**. It focuses on evidence filtering: given a query and several candidate passages, the model scores and ranks candidate evidence.

This is not a full reproduction of RankRAG. It does not reproduce LLM instruction tuning or answer generation. The goal is to validate a compact context ranking pipeline, implement it in Jittor, keep a PyTorch baseline, and compare PyTorch/Jittor results on smoke-test synthetic data plus public MS MARCO small and medium subsets.

## Paper Information

**Paper:** RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs  
**Authors:** Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, Bryan Catanzaro  
**Conference:** NeurIPS 2024  
**Core idea:** the original RankRAG framework unifies context ranking and retrieval-augmented generation so that one model can rank useful context and generate answers.

## Reproduction Scope

- Reproduced: RankRAG-style context ranking / evidence filtering.
- Reproduced: PyTorch baseline and Jittor implementation.
- Reproduced: training, evaluation, visualization, and PyTorch/Jittor alignment.
- Reproduced: L2 multi-model ranking comparison on MS MARCO small and medium subsets, including TF-IDF, BM25, MLP, and TextCNN.
- Not reproduced: full LLM instruction tuning.
- Not reproduced: answer generation.
- Not equivalent to: RankRAG's full LLM-based generation pipeline.

The current scorer is deliberately lightweight, so its results should not be over-claimed as full RankRAG performance.

## Project Scope

This project is a lightweight reproduction of RankRAG-style context ranking in Jittor. It does not reproduce full Llama3 instruction tuning or answer generation.

## Experiment Levels

### L0 Synthetic smoke test

Used to validate pipeline correctness.

### L1 MS MARCO MLP reranker

Uses an MS MARCO small subset to validate the MLP scorer and PyTorch/Jittor alignment.

### L2 Multi-model context ranking

Adds traditional retrieval baselines and a TextCNN reranker:

- TF-IDF / BM25 baseline
- MLP scorer, PyTorch + Jittor
- TextCNN reranker, PyTorch + Jittor
- case study and error analysis

### Medium-scale public ranking evaluation

Uses a larger MS MARCO medium subset as the stronger public-data experiment while keeping the full workflow CPU-runnable.

## Method

The L1 MLP ranking pipeline is:

```text
query + candidate context
-> fixed text features
-> MLP scorer
-> pairwise ranking loss
-> relevance score
```

Feature dimension is `1537`: `q_emb, c_emb, abs(q-c), q*c, cosine`, where the HashingVectorizer embedding dimension is `384`, so `384 * 4 + 1 = 1537`.

The L2 TextCNN reranker uses:

```text
query <sep> candidate passage
-> token ids
-> embedding
-> Conv1d kernels 3/4/5
-> max-over-time pooling
-> linear scorer
-> pairwise ranking loss
```

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

MS MARCO small subset is retained for quick reproduction and debugging. It is closer to the RankRAG context ranking setup than the synthetic benchmark because it uses real query-passage data.

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

Run L2 multi-model experiments:

```bash
bash scripts/run_retrieval_baselines_msmarco.sh
bash scripts/run_train_textcnn_torch_msmarco.sh
bash scripts/run_eval_textcnn_torch_msmarco.sh
bash scripts/run_train_textcnn_jittor_msmarco.sh
bash scripts/run_eval_textcnn_jittor_msmarco.sh
python src/aggregate_l2_results.py
python src/case_study_msmarco.py
```

### 3. MS MARCO medium subset

To make the evaluation more stable than the initial small subset, we further construct a medium subset:

- train: 5000 queries
- valid: 500 queries
- test: 500 queries
- candidates per query: 10

The medium subset is used as the stronger public-data experiment. The previous small subset is retained for quick reproduction and debugging. It is not the full MS MARCO dataset and is not a leaderboard setup, but increasing `candidates_per_query` from 5 to 10 makes Recall@5 and NDCG@5 more discriminative.

```bash
python scripts/prepare_msmarco_subset.py \
  --max_train_queries 5000 \
  --max_valid_queries 500 \
  --max_test_queries 500 \
  --candidates_per_query 10 \
  --output_dir data/processed/msmarco_medium \
  --run_name msmarco_medium \
  --seed 42

bash scripts/run_retrieval_baselines_msmarco_medium.sh
bash scripts/run_train_torch_msmarco_medium.sh
bash scripts/run_eval_torch_msmarco_medium.sh
bash scripts/run_train_jittor_msmarco_medium.sh
bash scripts/run_eval_jittor_msmarco_medium.sh
bash scripts/run_train_textcnn_torch_msmarco_medium.sh
bash scripts/run_eval_textcnn_torch_msmarco_medium.sh
bash scripts/run_train_textcnn_jittor_msmarco_medium.sh
bash scripts/run_eval_textcnn_jittor_msmarco_medium.sh
python src/compare_results.py --run_name msmarco_medium
python src/plot_results.py --run_name msmarco_medium
python src/aggregate_l2_results.py --run_name msmarco_medium
python src/case_study_msmarco.py --run_name msmarco_medium
```

Readiness check:

```bash
python scripts/check_project_ready.py
```

### 4. Academic demo

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

## MS MARCO Medium Subset

Source: `microsoft/ms_marco`, config `v1.1`.

| Split | Queries | Avg candidates | Positives | Negatives |
| --- | ---: | ---: | ---: | ---: |
| train | 5000 | 8.11 | 5000 | 35567 |
| valid | 500 | 8.16 | 500 | 3580 |
| test | 500 | 8.09 | 500 | 3544 |

The script requests at most 10 candidates per query. The observed average is about 8.1 because some original MS MARCO rows have fewer than 10 usable labeled passages after requiring at least one positive and one negative.

## L2 Multi-model MS MARCO Medium Results

| Method | Framework | Status | Recall@1 | Recall@3 | Recall@5 | Recall@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MRR | Pairwise Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TFIDF | sklearn/rank_bm25 | ready | 0.2220 | 0.5700 | 0.7880 | 1.0000 | 0.2220 | 0.4188 | 0.5084 | 0.5785 | 0.4465 | 0.6327 |
| BM25 | sklearn/rank_bm25 | ready | 0.2300 | 0.5540 | 0.7840 | 1.0000 | 0.2300 | 0.4127 | 0.5074 | 0.5791 | 0.4476 | 0.6253 |
| MLP | PyTorch | ready | 0.1920 | 0.4780 | 0.7000 | 1.0000 | 0.1920 | 0.3559 | 0.4475 | 0.5476 | 0.4079 | 0.5751 |
| MLP | Jittor | ready | 0.2280 | 0.5060 | 0.7120 | 1.0000 | 0.2280 | 0.3853 | 0.4698 | 0.5657 | 0.4318 | 0.5901 |
| TextCNN | PyTorch | ready | 0.1720 | 0.4960 | 0.7220 | 1.0000 | 0.1720 | 0.3534 | 0.4463 | 0.5383 | 0.3953 | 0.5708 |
| TextCNN | Jittor | ready | 0.1800 | 0.4500 | 0.6780 | 1.0000 | 0.1800 | 0.3341 | 0.4270 | 0.5341 | 0.3912 | 0.5484 |

Medium results are more discriminative than small results: Recall@5 is no longer naturally 1.0 because each query can have up to 10 candidates. BM25/TF-IDF remain strong because MS MARCO query-passage relevance often has substantial lexical overlap. The MLP and TextCNN models here are trained from scratch and do not use pretrained semantic encoders, which helps explain why they do not consistently exceed lexical baselines.

## L2 Multi-model MS MARCO Results

| Method | Framework | Status | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR | Pairwise Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TFIDF | sklearn/rank_bm25 | ready | 0.2950 | 0.7500 | 1.0000 | 0.2950 | 0.5579 | 0.6598 | 0.5477 | 0.6129 |
| BM25 | sklearn/rank_bm25 | ready | 0.3100 | 0.7600 | 1.0000 | 0.3100 | 0.5664 | 0.6656 | 0.5552 | 0.6242 |
| MLP | PyTorch | ready | 0.2600 | 0.6500 | 1.0000 | 0.2600 | 0.4812 | 0.6251 | 0.5031 | 0.5529 |
| MLP | Jittor | ready | 0.2450 | 0.6700 | 1.0000 | 0.2450 | 0.4863 | 0.6214 | 0.4978 | 0.5546 |
| TextCNN | PyTorch | ready | 0.2400 | 0.6350 | 1.0000 | 0.2400 | 0.4670 | 0.6165 | 0.4917 | 0.5396 |
| TextCNN | Jittor | ready | 0.2400 | 0.6250 | 1.0000 | 0.2400 | 0.4567 | 0.6106 | 0.4842 | 0.5267 |

On this small subset, BM25 is the strongest L2 method among the tested lightweight rankers. TextCNN does not clearly improve over MLP or BM25 under the current short training setup. Its value here is that it adds a Jittor-native neural reranker and makes the comparison more complete. Stronger semantic ranking would likely require a pretrained encoder or LLM-style reranker, which is outside this project's current L2 scope.

## L1 MLP Alignment Results: MS MARCO Small

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
- `outputs/msmarco_retrieval_baseline_metrics.json`
- `outputs/msmarco_retrieval_baseline_metrics.md`
- `outputs/msmarco_retrieval_baseline_rankings.json`
- `logs/msmarco_textcnn_torch_train.log`
- `logs/msmarco_textcnn_jittor_train.log`
- `outputs/msmarco_textcnn_torch_metrics.json`
- `outputs/msmarco_textcnn_jittor_metrics.json`
- `outputs/msmarco_textcnn_demo_ranking_result_torch.json`
- `outputs/msmarco_textcnn_demo_ranking_result_jittor.json`
- `outputs/l2_msmarco_results.json`
- `outputs/l2_msmarco_results.md`
- `outputs/l2_msmarco_results.png`
- `outputs/msmarco_case_study.json`
- `docs/msmarco_case_study.md`
- `docs/msmarco_medium_case_study.md`
- `docs/hardware_report.md`

Medium-specific outputs:

- `data/processed/msmarco_medium/dataset_card.md`
- `logs/msmarco_medium_torch_train.log`
- `logs/msmarco_medium_jittor_train.log`
- `logs/msmarco_medium_textcnn_torch_train.log`
- `logs/msmarco_medium_textcnn_jittor_train.log`
- `outputs/msmarco_medium_torch_metrics.json`
- `outputs/msmarco_medium_jittor_metrics.json`
- `outputs/msmarco_medium_textcnn_torch_metrics.json`
- `outputs/msmarco_medium_textcnn_jittor_metrics.json`
- `outputs/msmarco_medium_retrieval_baseline_metrics.json`
- `outputs/msmarco_medium_metrics_compare.md`
- `outputs/msmarco_medium_loss_curve.png`
- `outputs/msmarco_medium_metrics_compare.png`
- `outputs/l2_msmarco_medium_results.md`
- `outputs/l2_msmarco_medium_results.png`
- `outputs/msmarco_medium_case_study.json`
- `docs/msmarco_medium_case_study.md`

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

This project should be read as a lightweight RankRAG-style context ranking reproduction in Jittor. MS MARCO medium results are more meaningful than the synthetic smoke-test and more stable than the initial small subset, but they still use a subset and compact rankers. The project does not claim full RankRAG reproduction, LLM generation quality, or real academic search generalization.

L2 results are intended to compare lightweight context ranking choices, not to exceed RankRAG. The synthetic metrics only validate smoke-test correctness. MS MARCO small and medium subsets are public ranking-data experiments, but neither is a leaderboard setup. Lightweight MLP/TextCNN rankers lack the deep semantic judgment of RankRAG's LLM reranking setup.

Case studies:

- Small subset case study: `docs/msmarco_case_study.md`
- Medium subset case study: `docs/msmarco_medium_case_study.md`

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
- Add stronger non-LLM rerankers before attempting L3.
- Evaluate larger subsets while keeping storage and runtime manageable.
- Keep the scope focused on RankRAG-style context ranking unless full LLM instruction tuning is explicitly added.

## L2.5 External Pretrained Reranker Reference

This cross-encoder experiment is an external pretrained semantic reranker reference. It is not the Jittor reproduction body. The Jittor reproduction body remains the MLP/TextCNN rerankers implemented in Jittor. This reference is used to analyze why RankRAG-style LLM reranking matters.

The experiment uses `cross-encoder/ms-marco-MiniLM-L6-v2` through `sentence-transformers` as an external pretrained semantic reranker. It is not trained by this repository and should not be described as a Jittor implementation.

Purpose:

- BM25 / TF-IDF: lexical matching baselines.
- Jittor MLP/TextCNN: lightweight trainable rerankers and the reproduction body.
- Cross-Encoder: pretrained semantic reranker reference.

If Cross-Encoder is clearly stronger, the result should be interpreted as the effect of pretrained semantic relevance modeling. If it is not clearly stronger, report that honestly and consider subset size, candidate construction, model size, and MS MARCO matching patterns.

Run on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_cross_encoder_msmarco_medium.ps1
python src/aggregate_l25_results.py
python src/case_study_cross_encoder.py
```

Run on bash:

```bash
bash scripts/run_cross_encoder_msmarco_medium.sh
python src/aggregate_l25_results.py
python src/case_study_cross_encoder.py
```

Expected outputs:

- `outputs/msmarco_medium_cross_encoder_metrics.json`
- `outputs/msmarco_medium_cross_encoder_metrics.md`
- `outputs/msmarco_medium_cross_encoder_rankings.json`
- `outputs/l25_msmarco_medium_results.md`
- `outputs/l25_msmarco_medium_results.png`
- `docs/msmarco_medium_cross_encoder_case_study.md`

The L2.5 result table is written to `outputs/l25_msmarco_medium_results.md`.

### L2.5 MS MARCO medium result table

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

