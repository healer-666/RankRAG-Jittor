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

## 与原 RankRAG 的关系

原 RankRAG 统一 context ranking 与 answer generation，通过 instruction tuning 让同一个 LLM 同时承担上下文排序和答案生成。

本项目主动收缩范围，只复现 context ranking / selector 模块，不复现 answer generation，也不做 LLM instruction tuning。该模块对应 Paper-Skill 学术搜索增强中的证据筛选能力。

## 当前局限

- 只使用 MS MARCO small subset。
- 只做轻量 fixed-feature MLP scorer。
- 未复现 RankRAG 的 LLM instruction tuning。
- 未复现 answer generation。
- 未使用完整 MS MARCO passage ranking leaderboard 设置。
- Jittor 已在 Ubuntu 22.04 CPU 环境跑通，但尚未验证 GPU 路径。
