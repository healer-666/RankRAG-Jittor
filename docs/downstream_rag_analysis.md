# Downstream RAG Answer Generation Analysis

## Motivation

Stage D verifies the downstream chain:

```text
better reranking -> better top-k context -> better final answer
```

This experiment reuses the existing MS MARCO medium test candidate pool and committed reranker rankings. It does not redesign a new retrieval dataset.

## Dataset Construction

The processed file `data/processed/msmarco_medium/test.jsonl` contains 500 queries and 4,044 candidate passages, but it does not retain `answer`, `answers`, or `wellFormedAnswers`. Gold answers are recovered from `microsoft/ms_marco:v1.1` validation by matching normalized query text.

The downstream QA subset is saved at `data/processed/msmarco_downstream_qa_50.jsonl`.

Selection rules:

- Keep 50 questions.
- Keep the original MS MARCO medium candidate pool for each selected query.
- Require non-empty short gold answers.
- Require at least 3 candidate passages.
- Require at least one candidate passage to contain the normalized gold answer.
- Do not modify `data/processed/msmarco_medium/test.jsonl`.

The subset manifest is `outputs/downstream_rag_eval/qa_subset_manifest.json`.

## Compared Rerankers

The first-pass downstream evaluation compares:

| Method | Ranking source |
| --- | --- |
| BM25 | `outputs/msmarco_medium_retrieval_baseline_rankings.json` |
| LoRA v3 | `outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_rankings.json` |
| Cross-Encoder | `outputs/msmarco_medium_cross_encoder_rankings.json` |

All three methods use the same 50 questions, the same candidate pool, and top-k = 3.

## Fixed Answer Generator

| Item | Value |
| --- | --- |
| Generator | Qwen/Qwen2.5-1.5B-Instruct |
| Backend | PyTorch Transformers |
| Device | NVIDIA GeForce RTX 3060 Laptop GPU |
| dtype | torch.float16 |
| Batch size | 1 |
| top-k contexts | 3 |
| max context tokens | 2048 |
| max new tokens | 64 |
| do_sample | false |
| temperature | 0.0 |

The model is loaded from a local directory outside the Git repository through `QWEN_GENERATOR_MODEL_PATH` or the `--generator-model-path` CLI option. The repository does not store model weights or the local model path.

A stronger-generator sensitivity check with Qwen2.5-7B-Instruct is reported separately in [downstream_rag_generator_comparison.md](downstream_rag_generator_comparison.md). It uses the same 50 questions, rankings, top-3 contexts, prompt, and generation settings, and only changes the generator scale.

A downstream prompt ablation / generation-format sensitivity check for Qwen2.5-1.5B-Instruct is reported in [downstream_rag_prompt_ablation_1_5b.md](downstream_rag_prompt_ablation_1_5b.md). It keeps the generator, rankings, top-3 contexts, question set, context order, and decoding settings fixed, and only changes the prompt style. The Qwen2.5-7B strict-prompt ablation has not been run.

## Prompt

```text
Answer the question using only the provided contexts.

Give a short and direct answer.
Do not add information that is not supported by the contexts.
If the contexts do not contain enough information, answer exactly:
Insufficient information

Question:
{question}

Contexts:
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}

Answer:
```

The prompt never includes passage labels.

## Evaluation Protocol

Answer metrics use SQuAD-style normalization:

- lowercase
- remove punctuation
- remove English articles
- normalize whitespace
- support multiple gold answers by taking the best score

Reported answer metrics:

- Answer Hit Rate: exact match or normalized gold containment
- Exact Match
- token-level F1
- gold containment
- empty-answer rate
- insufficient-information rate
- average answer length

Reported context metrics:

- labeled-positive-in-top-3 rate
- gold-answer-in-context rate
- average gold-answer context rank

Failure attribution:

- retrieval failure: gold answer is not in top-3 context
- generation failure: gold answer is in top-3 context but answer hit fails
- success: answer hit succeeds

## Main Results

| Method | Questions | Top-k | Positive in Context | Gold in Context | Answer Hit | EM | Token F1 | Generation Failure |
| ------ | --------: | ----: | ------------------: | --------------: | ---------: | -: | -------: | -----------------: |
| BM25 | 50 | 3 | 0.6400 | 0.6800 | 0.2200 | 0.0000 | 0.1646 | 0.6765 |
| LoRA v3 | 50 | 3 | 0.6200 | 0.7200 | 0.2800 | 0.0400 | 0.2108 | 0.6111 |
| Cross-Encoder | 50 | 3 | 0.8200 | 0.8800 | 0.2800 | 0.0000 | 0.2058 | 0.6818 |

## Reranking-to-Answer Relationship

| Method | R@1 | R@3 | R@5 | MRR | Gold in Context@3 | Answer Hit |
| ------ | --: | --: | --: | --: | ----------------: | ---------: |
| BM25 | 0.2300 | 0.5540 | 0.7840 | 0.4476 | 0.6800 | 0.2200 |
| LoRA v3 | 0.3580 | 0.6980 | 0.8720 | 0.5642 | 0.7200 | 0.2800 |
| Cross-Encoder | 0.4340 | 0.8080 | 0.9340 | 0.6341 | 0.8800 | 0.2800 |

The stronger rerankers generally improve evidence availability. LoRA v3 improves over BM25 on gold-in-context and answer hit. Cross-Encoder gives the best evidence availability, but its final answer hit is tied with LoRA v3 in this 50-question setting. This indicates that the fixed generator and strict automatic answer matching can become bottlenecks even when the top-3 context contains the evidence.

## Failure Split

| Method | Retrieval Failures | Generation Failures | Successes |
| --- | ---: | ---: | ---: |
| BM25 | 16 | 23 | 11 |
| LoRA v3 | 14 | 22 | 14 |
| Cross-Encoder | 6 | 30 | 14 |

Cross-Encoder substantially reduces retrieval failures, but many remaining errors are generation failures: the answer evidence is present, but the generated answer does not pass exact/containment hit criteria.

## Case Studies

Case studies are saved in `outputs/downstream_rag_eval/downstream_rag_case_study.json`. They include:

- LoRA better than BM25
- Cross-Encoder better than LoRA
- answer present in context but generation still fails
- all-method failure examples

One representative case is `msmarco_test_000001`: the gold answer is `8 hours 43 minutes`. BM25 and Cross-Encoder include the positive passage in top-3 and generate answers containing the gold duration. LoRA v3 includes the positive passage at rank 2 but also ranks a conflicting hard negative first, and the generator answers `10 Hours 20 Minutes`; this is classified as a generation failure because the evidence is available but the final answer follows the wrong context.

## Limitations

- The experiment uses a 50-question subset, not full MS MARCO.
- Gold answers are restored by query-text matching against MS MARCO validation.
- Automatic answer hit is strict; valid paraphrases may be undercounted.
- Some MS MARCO negatives can contain answer-like evidence, so label-based context availability and gold-answer-in-context are both reported.
- Results should not be interpreted as full RankRAG answer generation reproduction.
- The 7B generator comparison is a downstream diagnostic extension, not a new RankRAG main-result protocol.
- The 1.5B strict-prompt ablation is also a downstream extension experiment, not a reranking main result.

## Reproduction Commands

Set the local generator path outside the repository:

```powershell
$env:QWEN_GENERATOR_MODEL_PATH = "<local Qwen2.5-1.5B-Instruct directory>"
```

Run:

```powershell
python src/audit_downstream_rag_data.py --config configs/downstream_rag_50q.yaml
python src/build_downstream_qa_subset.py --config configs/downstream_rag_50q.yaml
python scripts/run_downstream_rag_eval.py --config configs/downstream_rag_50q.yaml --methods bm25,lora_v3,cross_encoder --top-k 3
python src/aggregate_downstream_rag_results.py --config configs/downstream_rag_50q.yaml --input-dir outputs/downstream_rag_eval --top-k 3
python scripts/validate_downstream_rag_results.py --config configs/downstream_rag_50q.yaml --output-dir outputs/downstream_rag_eval --top-k 3
```
