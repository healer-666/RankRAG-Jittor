<div align="center">

# RankRAG-Jittor

**基于 Jittor 的 RankRAG 风格大模型重排序轻量复现与实证分析。**

[简体中文](README.md) · [English](README.en.md) · [完整结果](docs/final_results.md) · [复现说明](docs/reproduction.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Jittor](https://img.shields.io/badge/Jittor-alignment-2563EB)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA%20reranking-EE4C2C)
[![RankRAG Paper](https://img.shields.io/badge/RankRAG-NeurIPS%202024-7C3AED)](https://arxiv.org/abs/2407.02485)

<img src="docs/figures/rankrag_jittor_overview.svg" alt="RankRAG-Jittor overview" width="940">

</div>

## 项目简介

本项目研究的问题是：在把资料交给大模型回答问题之前，怎样把真正有用的证据排到前面。一个 RankRAG 风格流程通常先检索 candidate passages，再让不同重排序方法判断 query-passage relevance，最后把排序靠前的 evidence 交给生成模型，并检查更好的排序是否真的带来更可靠的回答。

仓库围绕这条链路做轻量复现和实证分析。第一层是 PyTorch/Jittor MLP 与 TextCNN 的框架对齐复现；第二层是 Qwen2.5-1.5B zero-shot 与 LoRA 的 RankRAG 风格大模型重排序；第三层继续做下游 RAG、数据量消融、打分方式消融、错误类型分析和资源画像。

本仓库不是 RankRAG 原论文的完整复现，不包含 Llama 3、完整联合训练或原论文全部 benchmark。它聚焦的是可复现的 “LLM relevance judgment -> rerank -> downstream validation” 实验路径。

## 为什么选择 RankRAG，以及 Jittor 如何参与复现

RankRAG 的核心价值在于把检索增强生成中的 evidence ranking 单独建模：不是只问“能不能检索到相关 passage”，而是进一步问“相关 passage 能否被排到足够靠前，并被生成模型用上”。这很适合做一个考核型复现项目，因为它同时包含检索、重排序、框架迁移、大模型打分、微调、评测和误差分析。

Jittor 在本项目中的作用不是从零训练完整大模型，而是承担轻量重排序器复现与框架对齐验证：

| 路径 | 使用框架 | 在项目中的作用 |
| --- | --- | --- |
| MLP / TextCNN | PyTorch + Jittor | 在相同数据和指标下验证两条实现路径的整体趋势一致、结果处于相近水平 |
| Qwen zero-shot | JittorLLM | 验证不做本地任务训练时的大模型 relevance judgment 能力 |
| Qwen LoRA | PyTorch + Transformers + PEFT | 完成 RankRAG 风格 LLM reranker 的任务适配实验 |
| Cross-Encoder | sentence-transformers / PyTorch | 作为外部预训练重排序效果参照 |

这种拆分保留了 Jittor 复现要求，同时避免把项目表述成“完整 RankRAG 系统重写”。项目重点是重排序行为、可比评测和结果解释。

## 项目整体流程

当前 overview 图已经把中间部分画成并列方法池：Candidate passages 进入 Compared rerankers，BM25、PyTorch/Jittor 轻量基线、Qwen zero-shot、Qwen LoRA 和 Cross-Encoder 并行产生各自排序，再统一输出 Ranked passages。图中不存在 PyTorch MLP -> Jittor MLP 或 Qwen zero-shot -> LoRA -> Cross-Encoder 的串行依赖。

## 本仓库完成的主要工作

| 模块 | 完成内容 |
| --- | --- |
| 数据与候选池 | 构建 MS MARCO medium 子集，统一 500 个 query 和 4,044 个 query-passage pair |
| 轻量基线 | 实现并评估 BM25、MLP、TextCNN 等基础重排序方法 |
| Jittor 对齐 | 将 MLP/TextCNN 路径迁移到 Jittor，并与 PyTorch 路径在相同协议下对照 |
| LLM 重排序 | 完成 Qwen2.5-1.5B zero-shot relevance scoring 与 LoRA reranker 评测 |
| LoRA 消融 | 准备并完成 1k / 3k / 10k 嵌套训练数据量消融，统一 800 steps 设置 |
| Scoring 消融 | 比较 generate_parse、relevant_logprob、logprob_margin 三种打分方式 |
| 下游 RAG | 检查更强重排序是否提升 top-k evidence 与最终答案质量 |
| 诊断分析 | 完成 30 个分层诊断样例的错误类型分析和资源-效果画像 |
| 工程收尾 | 固化结果汇总、README 图表、复现文档和仓库完整性检查脚本 |

## 实验环境

| 实验部分 | 环境说明 |
| --- | --- |
| 本地开发与检查 | Windows + Python，用于数据处理、脚本检查、文档整理和部分轻量实验 |
| Jittor 路径 | Jittor/JittorLLM 用于轻量框架对齐与 Qwen zero-shot 路径验证 |
| LoRA 正式重跑 | 在统一的 RTX 4090 D 环境下完成的 10k LoRA 重跑结果，使用 PyTorch + Transformers + PEFT |
| 资源记录 | LoRA 10k-rerun 记录训练时间 187.8953 秒，评估时间 444.2874 秒，峰值显存 16,509.3 MiB |

资源画像不把异构硬件结果包装成严格速度 benchmark。严格资源可比的方法数为 0；效果指标可比，资源字段用于 provenance 和 profile。

## 数据集与评测设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | MS MARCO medium subset |
| 主评测规模 | 500 queries, 4,044 query-passage pairs |
| 候选池 | 所有主重排序方法共用同一候选池 |
| 主指标 | Recall@1/3/5, NDCG@5, MRR, pairwise accuracy |
| 下游 RAG | 50 个问题，比较 BM25、LoRA、Cross-Encoder 的 top-k evidence 与生成答案 |
| 错误分析 | 30 个分层诊断 query，不作为 500-query 全量分布估计 |

## PyTorch / Jittor 对齐复现

<img src="docs/figures/pytorch_jittor_alignment.svg" alt="PyTorch and Jittor alignment baselines" width="940">

MLP 和 TextCNN 是轻量对齐基线，不是 RankRAG 原论文的核心 LLM reranker。它们用于确认两条实现路径的整体趋势一致、结果处于相近水平。

| 模型 | PyTorch R@1 | Jittor R@1 | 结论 |
| --- | ---: | ---: | --- |
| MLP | 0.192 | 0.228 | 趋势接近，Jittor 略高 |
| TextCNN | 0.172 | 0.180 | 趋势接近，差异很小 |

这部分说明 Jittor 轻量实现具备可用的框架迁移基础，但不把它解读为严格逐项一致。

## 大模型重排序与 LoRA 复现

Qwen2.5-1.5B 路径分为两步。第一步使用 zero-shot relevance judgment 直接给 query-passage 对打分；第二步使用 1k、3k、10k 嵌套训练对进行 LoRA 任务适配，并在统一评测集上比较排序效果。

LoRA 主配置如下：

| 字段 | 设置 |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA rank | 8 |
| Learning rate | 1e-4 |
| Max steps | 800 |
| Max length | 256 |
| 主 scoring method | `logprob_margin` |
| 正式 10k 输出 | 新 10k-rerun 输出目录，不覆盖历史 10k 目录 |

历史目录 `outputs/lora_qwen_1_5b_10k_lr1e4_s800/` 保留为历史参照；正式 E1 比较使用统一 RTX 4090 D 环境下的 1k、3k、10k-rerun 结果。

## 实验结果

<img src="docs/figures/main_reranking_results.svg" alt="Reranking effectiveness on MS MARCO Medium" width="940">

| 方法 | R@1 | NDCG@5 | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.230 | 0.5074 | 0.4476 |
| Jittor MLP | 0.228 | 0.4698 | 0.4318 |
| Jittor TextCNN | 0.180 | 0.4270 | 0.3912 |
| Qwen2.5-1.5B Zero-shot | 0.236 | 0.5210 | 0.4525 |
| Qwen2.5-1.5B LoRA (10k pairs) | 0.356 | 0.6236 | 0.5633 |
| Cross-Encoder Reference | 0.434 | 0.7019 | 0.6341 |

核心结论：

- BM25 在这个子集上仍然是很强的词面匹配基线。
- Qwen LoRA 将同一 Qwen2.5-1.5B reranker 的 R@1 从 zero-shot 的 0.236 提升到 0.356，说明任务适配对 RankRAG 风格 relevance judgment 有明显帮助。
- Cross-Encoder R@1 = 0.434，是当前测试中效果最强的外部参照方法。
- LoRA 的意义是复现并验证 LLM reranking 路径，而不是宣称超过专门预训练的 Cross-Encoder。

完整 R@3、R@5、pairwise accuracy、runtime metadata 和 source artifact 路径见 [docs/final_results.md](docs/final_results.md)。

## 消融、下游 RAG、错误分析与资源分析

<img src="docs/figures/readme_lora_ablation.svg" alt="LoRA data-size ablation" width="900">

数据量消融比较嵌套的 1k、3k、10k training pairs，统一 800 steps。10k-rerun 在 NDCG@5 和 pairwise accuracy 上最高，3k 在 R@1 上略高；这说明训练数据量、有效 epoch 和固定 step 预算共同影响结果。

<img src="docs/figures/readme_error_taxonomy.svg" alt="Error taxonomy" width="900">

错误分析把失败样例分为 lexical trap、small model semantic limit、evidence utilization failure、candidate/label issue、ambiguous query 等类型。该分析是 30 个分层诊断样例，不是全量 500-query 的无偏统计。

<img src="docs/figures/readme_resource_profile.svg" alt="Resource-effectiveness profile" width="900">

资源画像说明：效果可以在同一候选池和指标下比较，但不同方法的训练方式、硬件记录和运行环境不同，因此资源字段只作为说明性 profile。

更多细节见：

| 分析 | 文档 |
| --- | --- |
| LoRA 数据量消融 | [docs/ablation_analysis.md](docs/ablation_analysis.md) |
| Scoring method 消融 | [docs/scoring_ablation_analysis.md](docs/scoring_ablation_analysis.md) |
| 下游 RAG | [docs/downstream_rag_analysis.md](docs/downstream_rag_analysis.md) |
| 错误类型分析 | [docs/error_taxonomy.md](docs/error_taxonomy.md) |
| 资源-效果画像 | [docs/cost_effectiveness_analysis.md](docs/cost_effectiveness_analysis.md) |

## 快速运行

以下命令只基于已有 artifact 重建汇总和 README 图表，不训练、不推理、不下载模型权重，也不重新生成 rankings。

```bash
git clone https://github.com/healer-666/RankRAG-Jittor.git
cd RankRAG-Jittor
pip install -r requirements.txt
python scripts/build_final_project_summary.py
python scripts/build_readme_figures.py
python scripts/check_final_repository.py
```

常用入口：

| 目的 | 命令或文档 |
| --- | --- |
| 重新生成最终结果汇总 | `python scripts/build_final_project_summary.py` |
| 重新生成 README 图表 | `python scripts/build_readme_figures.py` |
| 检查最终仓库状态 | `python scripts/check_final_repository.py` |
| 查看完整复现命令 | [docs/reproduction.md](docs/reproduction.md) |

## 仓库结构与详细文档

| 路径 | 用途 |
| --- | --- |
| `configs/` | 实验与评估配置 |
| `data/processed/` | 已处理 MS MARCO 子集和 LoRA pair 文件 |
| `src/` | 检索、重排序、评估与聚合代码 |
| `scripts/` | 数据构建、结果检查、汇总、制图和分析脚本 |
| `outputs/` | 已提交的 metrics、rankings、summary 和分析图 |
| `logs/` | 历史运行日志与硬件监控记录 |
| `docs/` | 复现、结果、消融、错误和资源报告 |
| `docs/figures/` | README 与报告图 |

| 文档 | 内容 |
| --- | --- |
| [docs/final_results.md](docs/final_results.md) | 最终指标总表与证据路径 |
| [docs/reproduction.md](docs/reproduction.md) | 环境、数据准备与复现命令 |
| [docs/final_repository_audit.md](docs/final_repository_audit.md) | 仓库完整性检查 |

## 复现范围与局限

- 本项目是轻量复现和实证分析，不是完整 RankRAG reproduction。
- 主排序评测使用 500-query MS MARCO medium subset。
- 下游 RAG 使用 50 个问题。
- 错误分析使用 30 个分层诊断样例。
- 资源记录来自不同方法和运行环境，不能作为严格速度 benchmark。
- Git 仓库不纳入模型权重、LoRA adapter、checkpoint 或本地缓存。

## 引用

RankRAG-Jittor 基于 RankRAG 论文：

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

## License 状态

当前仓库未包含 `LICENSE` 文件。在正式 license 添加前，请按仓库所有者条款使用代码与 artifact。
