# 结果分析

## MS MARCO Small Subset Results

### 数据来源

当前主要公开 ranking 实验使用 `microsoft/ms_marco` 的 `v1.1` 配置，通过 Hugging Face `datasets` streaming 方式抽取小规模 query-passage ranking subset。

### Subset 构造

- train: 1000 queries
- valid: 200 queries
- test: 200 queries
- 每个 query 最多保留 5 个 candidate passages
- 至少包含 1 个 positive 和 1 个 negative
- positive 来自 `is_selected == 1`
- negative 来自同一 query 下 `is_selected == 0`

该 subset 是 MS MARCO small subset，不是完整 MS MARCO，也不是 leaderboard 设置。详细统计见 `data/processed/msmarco/dataset_card.md`。

### PyTorch 结果

| Metric | PyTorch |
| --- | ---: |
| recall@1 | 0.2600 |
| ndcg@1 | 0.2600 |
| recall@3 | 0.6500 |
| ndcg@3 | 0.4812 |
| recall@5 | 1.0000 |
| ndcg@5 | 0.6251 |
| mrr | 0.5031 |
| pairwise_accuracy | 0.5529 |

PyTorch training loss 从 `0.657214` 降至 `0.140893`，说明 pairwise ranking objective 在训练集上可以学习到区分信号。

### Jittor 结果

| Metric | Jittor |
| --- | ---: |
| recall@1 | 0.2450 |
| ndcg@1 | 0.2450 |
| recall@3 | 0.6700 |
| ndcg@3 | 0.4863 |
| recall@5 | 1.0000 |
| ndcg@5 | 0.6214 |
| mrr | 0.4978 |
| pairwise_accuracy | 0.5546 |

Jittor CPU training loss 从 `0.659262` 降至 `0.149369`，走势与 PyTorch 类似。

### PyTorch / Jittor 差异

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

两套实现的指标接近但不完全相同，差异来自框架初始化、优化器数值细节和训练过程中的实现差别。整体看，Jittor 实现与 PyTorch baseline 已经达到轻量复现所需的对齐程度。

## 与 Synthetic 结果的对比

Synthetic hard-negative benchmark 上 PyTorch/Jittor 均达到 1.0。该结果只说明 pipeline 正确，因为数据由模板生成，即使包含 hard negatives，也存在明显的词面和结构规律。

MS MARCO small subset 使用真实 query-passage 数据，因此更能反映 ranking 任务难度。它比 synthetic 更有说服力，但当前仍然只是 small subset，不代表完整 MS MARCO 或真实开放学术搜索表现。

## Demo 排序解释

`data/academic_demo.json` 用于展示 Paper-Skill 学术搜索场景下的候选文档排序。它适合做 qualitative demo，但不能替代公开 benchmark。

## L2 Multi-model MS MARCO Results

L2 升级将单一 MLP scorer 扩展为多模型 context ranking 对比，包括传统检索 baseline、MLP reranker 和 TextCNN reranker。

| Method | Framework | Status | Recall@1 | Recall@3 | Recall@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR | Pairwise Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TFIDF | sklearn/rank_bm25 | ready | 0.2950 | 0.7500 | 1.0000 | 0.2950 | 0.5579 | 0.6598 | 0.5477 | 0.6129 |
| BM25 | sklearn/rank_bm25 | ready | 0.3100 | 0.7600 | 1.0000 | 0.3100 | 0.5664 | 0.6656 | 0.5552 | 0.6242 |
| MLP | PyTorch | ready | 0.2600 | 0.6500 | 1.0000 | 0.2600 | 0.4812 | 0.6251 | 0.5031 | 0.5529 |
| MLP | Jittor | ready | 0.2450 | 0.6700 | 1.0000 | 0.2450 | 0.4863 | 0.6214 | 0.4978 | 0.5546 |
| TextCNN | PyTorch | ready | 0.2400 | 0.6350 | 1.0000 | 0.2400 | 0.4670 | 0.6165 | 0.4917 | 0.5396 |
| TextCNN | Jittor | ready | 0.2400 | 0.6250 | 1.0000 | 0.2400 | 0.4567 | 0.6106 | 0.4842 | 0.5267 |

### 主要观察

- BM25 在当前 small subset 上表现最好，说明词面匹配和局部统计仍然是强 baseline。
- MLP PyTorch 与 Jittor 仍然基本对齐，差异处于轻量 CPU 训练和框架数值差异可接受范围内。
- TextCNN 在 small subset 和 5 epoch 短训练设置下没有稳定优于 MLP 或 BM25。
- TextCNN 的主要价值不是指标提升，而是补充了一个 Jittor 原生 neural reranker，使 L2 comparison 更完整。

### 失败案例总结

`docs/msmarco_case_study.md` 中的 case study 显示，失败或中间案例通常来自：

- query 与 hard negative 有较高关键词重合；
- negative passage 的 answer style 与 positive 很接近；
- 轻量模型缺少 pretrained encoder 或 LLM reranker 的深层语义判断；
- TextCNN 只从当前 small subset 学习 embedding 和卷积特征，泛化能力有限。

### 与完整 RankRAG 的差距

完整 RankRAG 使用 LLM instruction tuning 来统一 context ranking 和 answer generation。本项目 L2 仍然是轻量 context ranking reproduction，不包含 Llama3 微调、answer generation 或 LLM-style reranking。因此不能将当前结果解释为完整 RankRAG 性能。

## 与原 RankRAG 的关系

原 RankRAG 统一 context ranking 与 answer generation，通过 instruction tuning 让同一个 LLM 同时承担上下文排序和答案生成。

本项目主动收缩范围，只复现 context ranking / selector 模块，不复现 answer generation，也不做 LLM instruction tuning。该模块对应 Paper-Skill 学术搜索增强中的证据筛选能力。

## 当前局限

- 只使用 MS MARCO small subset。
- 只做轻量 TF-IDF/BM25、fixed-feature MLP 和 TextCNN reranker。
- 未复现 RankRAG 的 LLM instruction tuning。
- 未复现 answer generation。
- 未使用完整 MS MARCO passage ranking leaderboard 设置。
- Jittor 已在 Ubuntu 22.04 CPU 环境跑通，但尚未验证 GPU 路径。
