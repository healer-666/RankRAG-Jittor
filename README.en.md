<div align="center">

# RankRAG-Jittor

**A Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.**

[简体中文](README.md) · [English](README.en.md) · [Full Results](docs/final_results.md) · [Reproduction](docs/reproduction.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Jittor](https://img.shields.io/badge/Jittor-Alignment-2563EB)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA-EE4C2C)
[![RankRAG](https://img.shields.io/badge/RankRAG-NeurIPS%202024-7C3AED)](https://arxiv.org/abs/2407.02485)

<img src="docs/figures/rankrag_jittor_overview_final.png" alt="RankRAG-Jittor overview" width="920">

</div>

## 📌 Project Overview

RAG systems first retrieve passages that appear relevant to a question, but lexical similarity does not guarantee that a passage can directly answer the question. This project reproduces the core RankRAG idea: before answer generation, judge the relevance between the query and candidate passages again, then rank the more useful evidence higher.

```text
Query
-> Candidate passages
-> Reranking
-> Top-k evidence
-> Answer generation
```

This project uses RankRAG (NeurIPS 2024) as the reproduction target and completes:

* PyTorch/Jittor alignment for lightweight rerankers;
* JittorLLM Qwen2.5-1.5B zero-shot reranking;
* Qwen2.5-1.5B LoRA reranker training;
* comparison among BM25, lightweight models, LLM rerankers, and Cross-Encoder;
* data-size ablation, scoring-method ablation, and downstream RAG validation.

The middle area in the overview figure is a **parallel comparison pool**: BM25, PyTorch/Jittor lightweight baselines, Qwen zero-shot, Qwen LoRA, and Cross-Encoder independently rerank the same candidate pool. They are not serial modules.

---

## 🧭 Why Jittor, JittorLLM, and PyTorch Are Used Together

RankRAG focuses on using large language models for context ranking and generation, not training a complete large model from scratch. This project does not force every module into Jittor. Instead, each framework is used where it is most appropriate: Jittor for controllable alignment, JittorLLM for zero-shot LLM reranking, and PyTorch for the more mature LoRA training stack.

| Experiment path | Framework | Role |
| --- | --- | --- |
| MLP / TextCNN | PyTorch + Jittor | Validate lightweight reranker migration and result alignment |
| Qwen zero-shot | JittorLLM | Test LLM reranking ability without task-specific training |
| Qwen LoRA | PyTorch + Transformers + PEFT | Perform RankRAG-style relevance-task adaptation |
| Cross-Encoder | PyTorch | Mature specialized reranker reference |

**Why not align every module in both PyTorch and Jittor?**
MLP and TextCNN are small and stable enough for framework migration checks. Qwen2.5-1.5B LoRA, however, requires a full training stack: tokenizer behavior, mixed precision, LoRA adapters, GPU memory management, adapter saving/loading, and log-probability scoring. Re-implementing that entire stack in Jittor would move the focus away from RankRAG-style reranking and toward LLM training-framework migration.

**Why is Qwen LoRA not trained in Jittor?**
The formal LoRA experiments need stable Transformers, PEFT, adapter handling, and scoring behavior. The PyTorch ecosystem is more mature for this path, reduces rented-GPU failure risk, and makes the final artifacts easier to reproduce.

The resulting design keeps the Jittor reproduction requirement visible while still covering the key RankRAG experimental path:

```text
Relevance judgment -> Candidate reranking -> Downstream QA validation
```

---

## 🧪 Experiment Environment

| Part | Environment |
| --- | --- |
| Local development and result organization | Windows, Python 3.10, RTX 3060 Laptop GPU |
| PyTorch lightweight baselines | Windows or Ubuntu, PyTorch, CPU-reproducible |
| Jittor lightweight baselines | Ubuntu, Jittor, CPU-reproducible |
| Qwen zero-shot | Ubuntu, JittorLLM, GPU |
| Formal LoRA experiments | Ubuntu, RTX 4090 D, PyTorch, Transformers, PEFT |

The LoRA row reports the **10k LoRA rerun completed in a unified RTX 4090 D environment**:

| Item | Value |
| --- | ---: |
| Training steps | 800 |
| Training time | 187.90 s |
| Evaluation time | 444.29 s |
| Peak memory | 16,509.3 MiB |

Full environment details are in [docs/reproduction.md](docs/reproduction.md).

---

## 📦 Data Preparation

The project uses a unified **MS MARCO medium subset**. It is a medium-scale subset built from MS MARCO passage ranking data to make the reranking workflow reproducible under controlled compute cost. The build fixes seed `42` and produces pairwise training data, validation data, and a fixed test candidate pool.

Main evaluation candidate pool:

| Item | Count |
| --- | ---: |
| Test questions | 500 queries |
| Query-passage pairs | 4,044 |
| Candidates per query | up to 10 |
| Random seed | 42 |

All main reranking methods are evaluated on the same 500 queries and 4,044 query-passage pairs, so differences in metrics come from reranking behavior rather than candidate-pool changes.

Build the MS MARCO medium subset:

```bash
python scripts/prepare_msmarco_subset.py \
  --max_train_queries 5000 \
  --max_valid_queries 500 \
  --max_test_queries 500 \
  --candidates_per_query 10 \
  --output_dir data/processed/msmarco_medium \
  --run_name msmarco_medium \
  --seed 42
```

Build nested LoRA 1k, 3k, and 10k training subsets:

```bash
python scripts/build_lora_data_size_ablation.py
python scripts/check_lora_data_ablation.py
```

---

## 🧩 Jittor Reproduction Steps

Run this section in an **Ubuntu + Jittor/JittorLLM** environment. MLP/TextCNN can be reproduced on CPU; Qwen zero-shot requires a GPU and a local Qwen2.5-1.5B model path.

### 1. Jittor MLP

```bash
python src/train_jittor.py --config configs/msmarco_medium.yaml
python src/eval_jittor.py --config configs/msmarco_medium.yaml
```

### 2. Jittor TextCNN

```bash
python src/train_textcnn_jittor.py \
  --config configs/msmarco_medium_textcnn.yaml

python src/eval_textcnn_jittor.py \
  --config configs/msmarco_medium_textcnn.yaml
```

### 3. JittorLLM Qwen zero-shot reranking

Set the local Qwen2.5-1.5B model path in the config first:

```bash
python src/jittorllm_reranker/evaluate_qwen2_jittor.py \
  --config configs/jittorllm_qwen2_1_5b_full.yaml
```

---

## ⚙️ PyTorch Reproduction Steps

PyTorch MLP/TextCNN can run on Windows or Linux CPU environments. Formal Qwen LoRA experiments should run in an **Ubuntu + RTX 4090 D + PyTorch + Transformers + PEFT** environment to avoid local VRAM limits and excessive training time.

### 1. PyTorch MLP

```bash
python src/train_torch.py --config configs/msmarco_medium.yaml
python src/eval_torch.py --config configs/msmarco_medium.yaml
```

### 2. PyTorch TextCNN

```bash
python src/train_textcnn_torch.py \
  --config configs/msmarco_medium_textcnn.yaml

python src/eval_textcnn_torch.py \
  --config configs/msmarco_medium_textcnn.yaml
```

### 3. Qwen LoRA reranking

Set the local model path in the rented-GPU environment:

```bash
export QWEN_LORA_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
```

Train the formal 10k model:

```bash
python src/lora_reranker/train_lora_reranker.py \
  --config configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml
```

Evaluate it:

```bash
python src/lora_reranker/evaluate_lora_reranker.py \
  --config configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml
```

Commands for 1k, 3k, and other experiments are in [docs/reproduction.md](docs/reproduction.md).

---

## 🧑‍💻 Main Work Completed

This is an individual reproduction project. The implementation, experiments, and analysis were completed as part of the project work. In addition to local development, the formal LoRA experiments used rented RTX 4090 D GPU resources:

1. Built unified MS MARCO train/validation/test data and candidate pools.
2. Implemented PyTorch/Jittor MLP and TextCNN and compared them under the same protocol.
3. Ran Qwen2.5-1.5B zero-shot reranking through JittorLLM.
4. Built LoRA relevance-training data and completed formal 1k, 3k, and 10k experiments.
5. Implemented unified evaluation for BM25, Cross-Encoder, and Qwen LoRA.
6. Completed data-size, scoring-method, and downstream RAG ablations.
7. Organized training logs, performance logs, GPU records, and result figures.
8. Wrote data checks, result aggregation, and repository audit scripts.

The most expensive part was Qwen LoRA training and evaluation. It required model upload/path setup, rented-GPU training, evaluation, log preservation, result download, and repeated consistency checks. The README reports formal results and reproducible commands rather than packaging intermediate failed or partial local runs as final experiments.

---

## 🔁 PyTorch / Jittor Alignment

<img src="docs/figures/pytorch_jittor_alignment.svg" alt="PyTorch and Jittor alignment" width="880">

| Model | PyTorch R@1 | Jittor R@1 |
| --- | ---: | ---: |
| MLP | 0.192 | 0.228 |
| TextCNN | 0.172 | 0.180 |

MLP and TextCNN are **lightweight alignment baselines**. They verify that the Jittor implementations stay in a similar trend range to the PyTorch counterparts. They are not the core LLM reranker from the RankRAG paper.

---

## 📉 Training Behavior and Loss Curves

<img src="docs/figures/05_training_curves.png" alt="Training and loss curves" width="880">

Main training logs:

| Experiment | PyTorch log | Jittor log |
| --- | --- | --- |
| MLP | [log](logs/msmarco_medium_torch_train.log) | [log](logs/msmarco_medium_jittor_train.log) |
| TextCNN | [log](logs/msmarco_medium_textcnn_torch_train.log) | [log](logs/msmarco_medium_textcnn_jittor_train.log) |

Formal LoRA logs:

* [1k train log](logs/e1_autodl_4090d/1k_train.log)
* [3k train log](logs/e1_autodl_4090d/3k_train.log)
* [10k train log](logs/e1_autodl_4090d/10k_rerun_train.log)
* [10k eval log](logs/e1_autodl_4090d/10k_rerun_eval.log)

---

## 📊 Experimental Results

<img src="docs/figures/main_reranking_results.svg" alt="Main reranking results" width="880">

| Method | R@1 | R@3 | R@5 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.230 | 0.554 | 0.784 | 0.5074 | 0.4476 |
| Jittor MLP | 0.228 | 0.506 | 0.712 | 0.4698 | 0.4318 |
| Jittor TextCNN | 0.180 | 0.450 | 0.678 | 0.4270 | 0.3912 |
| Qwen Zero-shot | 0.236 | 0.552 | 0.812 | 0.5210 | 0.4525 |
| Qwen2.5-1.5B LoRA (10k pairs) | 0.356 | 0.696 | 0.866 | 0.6236 | 0.5633 |
| Cross-Encoder | **0.434** | **0.808** | **0.934** | **0.7019** | **0.6341** |

Key conclusions:

* BM25 remains a strong lexical baseline on this subset.
* Qwen LoRA improves R@1 from the zero-shot `0.236` to `0.356`.
* LoRA training clearly improves the LLM relevance-judgment ability.
* Cross-Encoder remains the strongest external effectiveness reference in this test, reaching R@1 = 0.434.

Full metrics and result artifacts are in [docs/final_results.md](docs/final_results.md).

---

## 🧪 Additional Experiments

<img src="docs/figures/readme_lora_ablation.svg" alt="LoRA data-size ablation" width="850">

* [Data-size ablation](docs/ablation_analysis.md): compares 1k, 3k, and 10k training subsets under fixed 800 steps.
* [Scoring-method ablation](docs/scoring_ablation_analysis.md): compares `generate_parse`, `relevant_logprob`, and `logprob_margin`.
* [Downstream RAG](docs/downstream_rag_analysis.md): checks whether better ranking improves final answers.
* [Error taxonomy](docs/error_taxonomy.md): analyzes lexical traps, semantic limits, and evidence-use failures.
* [Resource profile](docs/cost_effectiveness_analysis.md): records runtime, memory, and hardware provenance.

<details>
<summary>More experiment figures</summary>

<img src="docs/figures/readme_error_taxonomy.svg" alt="Error taxonomy" width="820">

<img src="docs/figures/readme_resource_profile.svg" alt="Resource profile" width="820">

</details>

---

## 🧱 Trial, Error, and Takeaways

The main lessons came from three kinds of iteration:

* **Framework boundary**: it is tempting to interpret "Jittor reproduction" as rewriting every module in Jittor. The more practical choice was to use Jittor for lightweight alignment and JittorLLM zero-shot validation, while using PyTorch for the LoRA path where the training ecosystem is more stable.
* **Training cost**: the local 3060 environment was not suitable for formal LoRA ablation. The final 1k, 3k, and 10k-rerun experiments used rented RTX 4090 D resources, so data, configs, output paths, and overwrite guards had to be fixed before training to control cost.
* **Result interpretation**: better ranking metrics do not automatically mean better generated answers. The final report separates ranking, downstream RAG, error taxonomy, and resource profile to avoid over-reading a single metric.

The biggest takeaway is that reproduction is not just getting a model to run. It also means documenting data protocol, environment boundaries, failed attempts, comparability risks, and the final evidence behind each conclusion.

---

## 🧭 Reproduction Scope

This project reproduces the core RankRAG-style experimental path:

```text
LLM relevance judgment
-> Passage reranking
-> Downstream RAG validation
```

Due to compute and project-cycle limits, it does not reproduce Llama 3 8B/70B, large-scale joint training, or every benchmark from the original paper.

MLP/TextCNN are PyTorch/Jittor alignment baselines. Qwen LoRA is the main LLM reranking experiment. Cross-Encoder is an external effectiveness reference.

---

## 📚 Citation

* [RankRAG — arXiv](https://arxiv.org/abs/2407.02485)
* [RankRAG — OpenReview](https://openreview.net/forum?id=S1fc92uemC)

```bibtex
@inproceedings{yu2024rankrag,
  title = {RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs},
  author = {Yu, Yue and Ping, Wei and Liu, Zihan and Wang, Boxin and You, Jiaxuan and Zhang, Chao and Shoeybi, Mohammad and Catanzaro, Bryan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```
