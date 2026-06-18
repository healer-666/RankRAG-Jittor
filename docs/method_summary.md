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

- 接入真实学术 paper abstract 数据。
- 绘制训练与验证 loss 曲线。
- 输出 PyTorch/Jittor 指标对齐表。
- 制作方法说明 PPT 和录屏材料。
