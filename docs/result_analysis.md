# 结果分析

## 当前 PyTorch / Jittor 结果

当前 `outputs/torch_metrics.json` 与 `outputs/jittor_metrics.json` 中的测试指标如下：

| Metric | PyTorch | Jittor |
| --- | ---: | ---: |
| recall@1 | 1.0000 | 1.0000 |
| recall@3 | 1.0000 | 1.0000 |
| recall@5 | 1.0000 | 1.0000 |
| ndcg@1 | 1.0000 | 1.0000 |
| ndcg@3 | 1.0000 | 1.0000 |
| ndcg@5 | 1.0000 | 1.0000 |
| mrr | 1.0000 | 1.0000 |
| pairwise_accuracy | 1.0000 | 1.0000 |

PyTorch 训练 5 epoch 后 loss 从 `0.353935` 降至 `0.001904`。Jittor CPU 训练 5 epoch 后 loss 从 `0.334018` 降至 `0.001747`。两者在测试集上的 ranking 指标一致，demo top5 排序也一致；分数数值不完全相同，属于不同框架初始化、优化器实现和数值细节造成的正常差异。

## 为什么指标全为 1.0

当前 synthetic 数据主要用于流程验证：数据生成、固定文本特征、pairwise ranking loss、训练日志、评估指标、demo 排序和结果可视化都已经跑通。

即使已经加入 hard negatives，数据仍然存在模板化和词面规律。模型可以学习到这些合成模板中的稳定差异，因此 PyTorch 与 Jittor 在当前测试集上都达到满分。这个结果不能证明模型在真实学术搜索场景下具备泛化能力，只能证明轻量 ranking pipeline 的训练、评估、排序流程是正确的。

后续若要评估真实能力，需要接入真实 paper abstract、query 和 relevance label，或者显著扩大 synthetic 模板空间并降低 train/test 模板重叠。

## Demo 排序解释

当前 demo 查询为：

```text
非平稳强化学习中，value-based 方法如何使用 UCB 思想进行探索？
```

排序结果中，PyTorch 与 Jittor 都将 positive 候选排第 1；hard negatives 紧随其后；easy negatives 靠后。这说明当前轻量模型能在 synthetic demo 中区分强相关证据、近邻干扰项和明显无关项。

这适合作为 Paper-Skill 学术搜索增强中“证据筛选 / Selector Skill”的流程展示，但不应被解释为真实检索质量结论。

## 与原 RankRAG 的关系

原 RankRAG 统一 context ranking 与 answer generation，通过 instruction tuning 让同一个 LLM 同时承担上下文排序和答案生成。

本项目主动收缩范围，只复现 context ranking / selector 模块，不复现 answer generation，也不做 LLM instruction tuning。该模块对应 Paper-Skill 学术搜索增强中的证据筛选能力：从多个候选论文片段中筛选更适合作为回答依据的上下文。

## 当前局限

- 未复现完整 LLM instruction tuning。
- 未复现 answer generation。
- 未接入真实大规模学术论文数据。
- Jittor 已在 Ubuntu 22.04 CPU 环境跑通，但尚未验证 GPU 路径。
- Synthetic 数据仍有模板化特征，泛化性有限。
