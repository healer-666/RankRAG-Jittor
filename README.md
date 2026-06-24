<div align="center">

# RankRAG-Jittor

**基于 Jittor 的 RankRAG 风格大模型重排序轻量复现与实证分析**

[简体中文](README.md) · [English](README.en.md) · [完整结果](docs/final_results.md) · [复现说明](docs/reproduction.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Jittor](https://img.shields.io/badge/Jittor-alignment-2563EB)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA%20reranking-EE4C2C)
[![RankRAG Paper](https://img.shields.io/badge/RankRAG-NeurIPS%202024-7C3AED)](https://arxiv.org/abs/2407.02485)

<img src="docs/figures/rankrag_jittor_overview.svg" alt="RankRAG-Jittor overview" width="940">

</div>

## 项目简介

在 RAG 系统中，检索器通常会先找出一批与问题相关的资料，但“内容相关”并不等于“能够直接回答问题”。如果真正有用的证据排得太靠后，即使后面的生成模型能力很强，最终答案也可能出错。

本项目关注的就是生成之前的这一步：**重新判断每段候选资料与问题的相关程度，并把真正有用的证据排到前面。**

整体流程可以概括为：

```text
用户问题
→ 检索候选资料
→ 对候选资料重新排序
→ 选择排名靠前的证据
→ 大模型生成答案
→ 分析排序效果和回答质量
```

围绕这条流程，本仓库完成了两层主要复现工作：

1. 在 PyTorch 和 Jittor 中分别实现 MLP、TextCNN 排序器，检查相同模型迁移到 Jittor 后的结果趋势；
2. 使用 Qwen2.5-1.5B 完成 zero-shot 和 LoRA 重排序，验证 RankRAG“由语言模型判断资料相关性，再将排序结果用于下游生成”的核心思路。

在此基础上，项目还补充了下游 RAG、数据量消融、评分方式消融、错误类型分析和资源记录。

---

## 为什么选择 RankRAG

普通检索方法更擅长寻找关键词或主题相近的资料，但真实问答还要求模型判断：

> 这段资料是否真正包含回答当前问题所需的证据？

RankRAG 将上下文排序和检索增强生成联系起来，不只评价“有没有检索到正确资料”，还进一步检查：

* 正确资料能否进入前几名；
* 更好的排序能否改善最终答案；
* 正确证据已经进入上下文时，生成模型是否能够正确使用它。

这使项目能够形成一条比较完整的实验链：

```text
传统检索
→ 轻量神经排序
→ 大模型重排序
→ 下游回答
→ 消融与错误分析
```

---

## Jittor 在项目中的作用

本项目没有把所有模块强行改写成 Jittor，而是按照不同实验的目标选择对应框架。

| 路径             | 使用框架                            | 作用                       |
| -------------- | ------------------------------- | ------------------------ |
| MLP / TextCNN  | PyTorch + Jittor                | 在相同数据和评测协议下检查框架迁移后的结果趋势  |
| Qwen zero-shot | JittorLLM                       | 验证未经过本地任务训练的大模型相关性判断能力   |
| Qwen LoRA      | PyTorch + Transformers + PEFT   | 完成 RankRAG 风格的大模型重排序任务适配 |
| Cross-Encoder  | sentence-transformers / PyTorch | 作为成熟的专用语义排序参照            |

其中，MLP 和 TextCNN 主要用于完成可控的 PyTorch/Jittor 对齐；Qwen zero-shot 和 LoRA 则负责验证大模型重排序路线。

这种分层方式既保留了 Jittor 复现要求，也避免把项目描述成完整的大模型系统重写。

---

## 本仓库完成的主要工作

| 模块                | 完成内容                                              |
| ----------------- | ------------------------------------------------- |
| 数据与评测协议           | 构建 MS MARCO medium 子集，固定 500 个测试问题和 4,044 个问题—资料对 |
| 传统检索基线            | 完成 TF-IDF、BM25 等词面检索方法                            |
| PyTorch/Jittor 对齐 | 分别实现 MLP 和 TextCNN，并在相同数据与指标下进行对照                 |
| JittorLLM 推理      | 使用 Qwen2.5-1.5B 完成 zero-shot 相关性判断与全量重排序          |
| LoRA 重排序          | 构建相关性训练数据，完成 Qwen2.5-1.5B LoRA 训练和统一评估            |
| 数据量消融             | 在相同配置下比较 1k、3k、10k 训练数据                           |
| 评分方式消融            | 比较直接生成标签、相关标签概率和正负标签概率差                           |
| 下游 RAG            | 比较不同重排序方法对 top-k 证据和最终答案的影响                       |
| 错误分析              | 对 30 个分层选择的问题进行错误类型诊断                             |
| 资源分析              | 整理训练时间、评估时间、显存和硬件来源                               |
| 工程整理              | 提供结果汇总、配置检查、图表生成和仓库审计脚本                           |

---

## 实验环境

项目的不同部分运行在不同环境中，因此资源数据主要用于记录和溯源，不作为严格的统一硬件速度排名。

| 实验部分      | 环境与用途                                                      |
| --------- | ---------------------------------------------------------- |
| 本地开发与检查   | Windows + Python，用于数据处理、代码调试、结果汇总和文档生成                     |
| Jittor 路径 | Jittor / JittorLLM，用于轻量排序器对齐和 Qwen zero-shot 重排序           |
| LoRA 正式实验 | Ubuntu、NVIDIA GeForce RTX 4090 D、PyTorch、Transformers、PEFT |
| 结果分析      | Windows / Linux CPU 环境，直接读取已有实验产物                          |

LoRA 10k 正式重跑记录：

| 指标   |                        数值 |
| ---- | ------------------------: |
| 训练时间 |                187.8953 秒 |
| 评估时间 |                444.2874 秒 |
| 峰值显存 |              16,509.3 MiB |
| GPU  | NVIDIA GeForce RTX 4090 D |

不同方法的硬件、批处理方式和计时范围并不完全一致，因此仓库只提供资源画像，不给出“某方法比另一方法快多少倍”的严格结论。

---

## 数据集与评测设置

| 项目     | 设置                                        |
| ------ | ----------------------------------------- |
| 数据集    | MS MARCO medium subset                    |
| 主评测规模  | 500 queries，4,044 query-passage pairs     |
| 候选池    | 所有主要重排序方法使用同一候选池                          |
| 主指标    | Recall@1/3/5、NDCG@5、MRR、Pairwise Accuracy |
| 下游 RAG | 50 个问题，比较 BM25、LoRA、Cross-Encoder         |
| 错误分析   | 30 个分层选择的问题                               |

几个主要指标的含义：

* **Recall@1**：正确资料是否排在第一位；
* **Recall@3/5**：正确资料是否进入前三名或前五名；
* **MRR**：正确资料排名越靠前，分数越高；
* **NDCG@5**：综合评价前五名的排序质量；
* **Pairwise Accuracy**：模型能否把正确资料排在负样本之前。

---

## PyTorch / Jittor 对齐

<img src="docs/figures/pytorch_jittor_alignment.svg" alt="PyTorch and Jittor alignment baselines" width="940">

MLP 和 TextCNN 是用于框架对齐的轻量基线，并不是 RankRAG 原论文中的核心大模型排序器。

它们的作用是检查：

```text
相同数据和评测协议
→ 分别使用 PyTorch 与 Jittor 实现
→ 比较两条实现路径的结果趋势
```

| 模型      | PyTorch R@1 | Jittor R@1 | 观察                   |
| ------- | ----------: | ---------: | -------------------- |
| MLP     |       0.192 |      0.228 | 整体处于相近范围，Jittor 结果略高 |
| TextCNN |       0.172 |      0.180 | 两条实现路径差异较小           |

这里的目标不是让两个框架得到逐位相同的数值，而是确认 Jittor 实现没有出现明显偏离，能够复现相近的整体趋势。

完整训练与评估命令见 [复现说明](docs/reproduction.md)。

---

## 大模型重排序与 LoRA

Qwen2.5-1.5B 的重排序实验分为两个阶段。

### Zero-shot 重排序

首先直接使用预训练模型判断问题和候选资料是否相关，不进行本地任务训练。这个实验用于观察模型仅依靠预训练知识时具备多少相关性判断能力。

### LoRA 任务适配

随后使用相关性训练数据对同一个模型进行 LoRA 微调，让模型更适应当前重排序任务。

正式配置如下：

| 字段               | 设置                           |
| ---------------- | ---------------------------- |
| Base model       | `Qwen/Qwen2.5-1.5B-Instruct` |
| Training pairs   | 10,000                       |
| Validation pairs | 1,000                        |
| LoRA rank        | 8                            |
| Learning rate    | `1e-4`                       |
| Max steps        | 800                          |
| Max length       | 256                          |
| 主评分方式            | `logprob_margin`             |

历史 10k 结果保存在：

```text
outputs/lora_qwen_1_5b_10k_lr1e4_s800/
```

正式数据量消融中的 1k、3k、10k 三组实验则在同一 RTX 4090 D 环境中重新运行，并使用独立输出目录，避免覆盖历史结果。

---

## 主排序结果

<img src="docs/figures/main_reranking_results.svg" alt="Reranking effectiveness on MS MARCO Medium" width="940">

| 方法                     |   R@1 | NDCG@5 |    MRR |
| ---------------------- | ----: | -----: | -----: |
| BM25                   | 0.230 | 0.5074 | 0.4476 |
| Jittor MLP             | 0.228 | 0.4698 | 0.4318 |
| Jittor TextCNN         | 0.180 | 0.4270 | 0.3912 |
| Qwen2.5-1.5B Zero-shot | 0.236 | 0.5210 | 0.4525 |
| Qwen2.5-1.5B LoRA（10k） | 0.356 | 0.6236 | 0.5633 |
| Cross-Encoder          | 0.434 | 0.7019 | 0.6341 |

结果可以概括为三点：

1. **BM25 仍然是很强的基线。**
   当前 MS MARCO 子集具有明显的关键词匹配特征，因此 BM25 能够超过从零训练的 MLP 和 TextCNN。

2. **任务微调明显改善了 Qwen 的排序能力。**
   同一个 Qwen2.5-1.5B 模型经过 LoRA 训练后，R@1 从 `0.236` 提升到 `0.356`，说明相关性任务适配比单纯使用 zero-shot 更有效。

3. **Cross-Encoder 仍然是当前最强的效果参照。**
   Cross-Encoder 的 R@1 达到 `0.434`。LoRA 的价值在于验证 RankRAG 风格的大模型重排序路线，而不是证明生成式大模型已经超过成熟的专用排序器。

完整的 R@3、R@5、Pairwise Accuracy 和结果来源见 [完整结果](docs/final_results.md)。

---

## 数据量消融

<img src="docs/figures/readme_lora_ablation.svg" alt="LoRA data-size ablation" width="900">

数据量实验比较了嵌套的 1k、3k 和 10k 训练集，其他主要配置保持一致，三组均训练 800 个优化步骤。

对应的近似有效训练轮数为：

| 训练数据量 | 近似有效轮数 |
| ----- | -----: |
| 1k    |    6.4 |
| 3k    |   2.13 |
| 10k   |   0.64 |

因此，这项实验研究的是：

> 在固定训练步数下，增加训练数据的多样性是否能够改善模型表现。

10k 模型在 NDCG@5 和 Pairwise Accuracy 上表现最好，而 3k 模型的 R@1 略高。这说明数据量并不是唯一因素，小数据被重复训练的次数、训练步数和数据覆盖范围会共同影响最终结果。

详细分析见 [LoRA 数据量消融](docs/ablation_analysis.md)。

---

## 评分方式消融

同一个 LoRA 模型可以使用不同方式给候选资料打分：

| 评分方式               | 含义                                    |
| ------------------ | ------------------------------------- |
| `generate_parse`   | 直接生成 `Relevant` 或 `Irrelevant` 并解析    |
| `relevant_logprob` | 使用完整 `Relevant` 标签序列的对数概率             |
| `logprob_margin`   | 使用 `Relevant` 与 `Irrelevant` 标签对数概率之差 |

三种方法使用的是同一个 10k LoRA 模型，改变的只是推理时如何把模型输出转换为排序分数。

实验显示，直接生成标签不仅运行时间更长，还会因为大量候选得到相同的二值分数而产生较多并列；使用标签概率能够得到更细粒度、更稳定的排序分数。

详细结果见 [评分方式消融](docs/scoring_ablation_analysis.md)。

---

## 下游 RAG 验证

排序指标更高，并不意味着最终回答一定更准确。

本项目固定选择每个问题排名前 3 的资料，并比较：

* BM25、LoRA、Cross-Encoder 三种重排序方法；
* Qwen2.5-1.5B 和 Qwen2.5-7B 两种生成模型；
* 原始提示词和严格短答案提示词。

主要观察是：

```text
正确证据没有进入上下文
→ 属于检索或排序问题

正确证据已经进入上下文，但回答仍然错误
→ 属于生成或证据使用问题
```

Cross-Encoder 更容易把正确资料排入前 3 名，但最终 Answer Hit 并没有始终明显超过 LoRA。这说明生成模型还可能忽略正确证据、使用其他资料、生成冗余内容，或者输出不符合自动指标要求的格式。

详细结果见 [下游 RAG 分析](docs/downstream_rag_analysis.md)。

---

## 错误类型分析

<img src="docs/figures/readme_error_taxonomy.svg" alt="Error taxonomy" width="900">

为了进一步解释不同方法为什么失败，本项目从正式结果中分层选择了 30 个问题，分析了以下错误类型：

* 词面误导；
* 轻量模型语义能力不足；
* 大模型过度判断；
* 背景相关但不能直接回答；
* 多证据混淆；
* 候选池或标签问题；
* 问题歧义；
* 正确证据进入上下文后仍未被正确使用。

这些案例是为了覆盖不同失败模式而选择的诊断样本，不能用来估计全部 500 个问题中的错误比例。

标注还包含 10 个案例的单标注者重复检查，主要错误类型一致率为 `0.70`。该数值只表示同一标注者重复判断时的稳定性，不代表跨标注者一致性。

详细案例见 [错误类型分析](docs/error_taxonomy.md)。

---

## 资源与效果画像

<img src="docs/figures/readme_resource_profile.svg" alt="Resource-effectiveness profile" width="900">

六种主要方法使用相同测试问题、候选池和评价指标，因此排序效果可以直接比较；但它们的运行硬件、软件环境、批处理方式和计时范围并不完全相同。

因此，仓库中的资源数据主要用于回答：

* 是否需要训练；
* 是否使用预训练语义模型；
* 是否记录了训练和评估时间；
* 显存和硬件信息是否可追溯。

这些数据不构成统一硬件下的公平速度排行榜。

详细说明见 [资源与效果分析](docs/cost_effectiveness_analysis.md)。

---

## 快速检查

下面的命令只会读取仓库中已有的实验产物，重新生成总结果、README 图表并检查仓库完整性。

它们不会训练模型、运行推理或下载 Qwen 权重。

```bash
git clone https://github.com/healer-666/RankRAG-Jittor.git
cd RankRAG-Jittor

pip install -r requirements.txt

python scripts/build_final_project_summary.py
python scripts/build_readme_figures.py
python scripts/check_final_repository.py
```

完整的数据准备、PyTorch/Jittor 训练、Qwen 重排序、LoRA 和下游 RAG 命令见：

* [完整复现说明](docs/reproduction.md)

---

## 仓库结构

```text
RankRAG-Jittor/
├── configs/          # 实验和评估配置
├── data/processed/   # 处理后的数据与 LoRA 训练对
├── docs/             # 结果、复现、消融和分析文档
├── logs/             # 训练、评估和硬件监控日志
├── outputs/          # 指标、排序结果和汇总产物
├── scripts/          # 数据构建、检查、汇总与制图脚本
└── src/              # 检索、排序、训练和评估代码
```

主要文档：

| 文档                                          | 内容                |
| ------------------------------------------- | ----------------- |
| [完整结果](docs/final_results.md)               | 全部正式指标和结果来源       |
| [复现说明](docs/reproduction.md)                | 环境、数据准备和完整运行命令    |
| [数据量消融](docs/ablation_analysis.md)          | 1k、3k、10k LoRA 对比 |
| [评分方式消融](docs/scoring_ablation_analysis.md) | 三种相关性评分方式         |
| [下游 RAG](docs/downstream_rag_analysis.md)   | 排序结果对最终回答的影响      |
| [错误分析](docs/error_taxonomy.md)              | 错误类型定义和代表性案例      |
| [资源分析](docs/cost_effectiveness_analysis.md) | 资源与效果画像           |
| [仓库审计](docs/final_repository_audit.md)      | 文件、链接和结果一致性检查     |

---

## 复现范围与局限

本项目复现的是 RankRAG 的核心思想链：

```text
语言模型判断资料相关性
→ 对候选资料重新排序
→ 验证排序对下游回答的影响
```

当前仓库没有覆盖：

* Llama 3 8B/70B；
* 原论文的大规模联合训练；
* 原论文全部公开和私有数据；
* 原论文全部 benchmark；
* 完整排序—生成统一模型训练。

其他限制包括：

* 主排序实验使用 MS MARCO medium subset；
* 下游 RAG 只评估了 50 个问题；
* 错误分析使用 30 个分层诊断案例；
* 资源数据来自不同运行环境，不能严格横向比较；
* Git 仓库不包含基础模型、LoRA adapter 和 checkpoint。

因此，本项目的准确定位是：

> **基于 Jittor 的 RankRAG 风格大模型重排序轻量复现与实证分析。**

---

## 引用

本项目基于 RankRAG 的核心思想：

* [arXiv: 2407.02485](https://arxiv.org/abs/2407.02485)
* [OpenReview: RankRAG](https://openreview.net/forum?id=S1fc92uemC)

```bibtex
@inproceedings{yu2024rankrag,
  title = {RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs},
  author = {Yu, Yue and Ping, Wei and Liu, Zihan and Wang, Boxin and You, Jiaxuan and Zhang, Chao and Shoeybi, Mohammad and Catanzaro, Bryan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```

---

## License

当前仓库尚未包含 `LICENSE` 文件。在正式许可证发布前，代码和实验产物的使用方式以仓库所有者的说明为准。
