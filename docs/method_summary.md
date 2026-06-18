# 方法说明

## 1. RankRAG 的核心思想

RankRAG 的核心思想是将 context ranking 与 answer generation 统一到 LLM fine-tuning 中：同一个模型既能判断候选上下文是否值得使用，也能基于筛选后的上下文生成答案。

## 2. 本复现的收缩范围

本项目只复现 context ranking / selector，不复现 answer generation，也不进行大模型 instruction tuning。输入是一个 query 与若干 candidate context，输出是候选证据的相关性排序。

## 3. 为什么这样做

完整 LLM 微调成本高、工程复杂，不适合短期 Jittor 复现。Context ranking 更贴合 Paper-Skill 学术搜索中的证据筛选任务，可以作为检索增强流程中轻量、可解释、可替换的 selector 模块。

## 4. 本项目模型

模型使用固定文本特征 + MLP scorer：

- `HashingVectorizer` 将 query 与 candidate text 映射到 384 维向量。
- 拼接 `q_emb`、`c_emb`、`abs(q_emb - c_emb)`、`q_emb * c_emb`、`cosine_similarity`。
- 得到 1537 维 pair/query-candidate 特征。
- MLP 输出单个相关性分数。

## 5. 本项目训练目标

训练使用 pairwise ranking loss：

```text
loss = -log(sigmoid(score_pos - score_neg))
```

该目标鼓励正样本分数高于同一 query 下的负样本分数。

## 6. 本项目评估

评估指标包括：

- Recall@K
- MRR
- NDCG@K
- Pairwise Accuracy

这些指标从候选排序角度衡量 selector 是否能把相关证据排在前面。

## 7. 下一步

## Dataset Upgrade

项目初始阶段使用 synthetic hard-negative 数据验证 pipeline，包括数据读取、固定文本特征、pairwise ranking loss、PyTorch/Jittor 训练、评估和可视化。

进一步升级后，项目引入 MS MARCO small subset 作为当前主要公开 ranking 数据实验。MS MARCO 使用真实 query-passage 数据，比 synthetic 模板数据更接近 RankRAG context ranking 的公开评测设置，也更能暴露轻量模型的真实局限。

当前项目仍然是 lightweight reproduction，不是 full RankRAG reproduction：

- 不做 LLM instruction tuning。
- 不做 answer generation。
- 不使用完整 MS MARCO leaderboard 设置。
- 重点验证 Jittor context ranking 实现及其与 PyTorch baseline 的对齐。

## L2 Upgrade: From Pipeline Reproduction to Multi-model Ranking

L2 升级的目标是从单一 pipeline reproduction 扩展到更完整的 context ranking 对比，而不是进入 L3 LoRA 或完整 LLM reranking。

### 为什么需要传统 baseline

TF-IDF 和 BM25 是 ranking 任务中必须报告的强 lexical baseline。它们训练成本低、可解释性强，能帮助判断神经 reranker 是否真的学到超过词面匹配的信号。在当前 MS MARCO small subset 上，BM25 反而是最强方法，说明轻量神经模型不能绕过传统 baseline 的比较。

### 为什么需要 TextCNN

MLP scorer 使用固定 HashingVectorizer 特征，验证了 Jittor/PyTorch pairwise ranking pipeline。TextCNN 进一步引入 token ids、embedding、Conv1d 和 max-over-time pooling，是一个更接近 neural reranker 的 Jittor 原生实现。

### TextCNN 和 MLP 的区别

- MLP 依赖固定文本特征，包括 query/candidate 向量交互和 cosine。
- TextCNN 从 `query <sep> candidate` token 序列学习局部 n-gram 模式。
- MLP 更稳定、训练更轻；TextCNN 更接近端到端神经 reranking，但 small subset 下容易过拟合或不稳定。

### 当前仍然不是完整 RankRAG

L2 仍然只覆盖 context ranking / evidence filtering。它不包含 Llama3 instruction tuning，不生成答案，也不具备完整 RankRAG 中 LLM reranker 的深层语义能力。

### 为什么不把 L3 LoRA 作为当前主线

`docs/hardware_report.md` 显示本机 NVIDIA 驱动当前无法通过 `nvidia-smi` 正常通信，PyTorch 是 CPU-only，LoRA 依赖也未准备好。因此 L3 small-model LoRA 不适合作为当前主线。当前更稳妥的升级路线是继续完善 L2：BM25 / TF-IDF baseline、TextCNN reranker、case study 和 error analysis。

## 8. 下一步

- 扩展到更标准的 MS MARCO passage ranking 设置。
- 引入 BEIR-SciFact 等科学文献相关公开评测集。
- 尝试更强的 neural reranker，同时保持 Jittor/PyTorch 对齐。
- 制作方法说明 PPT 和录屏材料。
