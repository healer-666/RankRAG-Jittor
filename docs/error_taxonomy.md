# Stage G: Error Taxonomy Analysis

## 1. Purpose

This analysis explains why existing reranking and downstream answer-generation results fail on selected diagnostic cases. It does not rerun models, does not train, and does not claim a full reproduction of RankRAG; the project remains a Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.

## 2. Data and Case Selection

- Ranking sources audited: 7.
- Unified query rows: 500.
- Selected diagnostic cases: 30 unique queries.
- Seed: 42.
- Selection buckets: A BM25 lexical/semantic gap, B BM25 right but neural wrong, C LoRA/Cross-Encoder divergence, D all core methods fail, E evidence in top-3 but generation fails.
- The sample is stratified for diagnosis. It is not an unbiased estimate of error proportions over all 500 test queries.

## 3. Error Taxonomy

- `lexical_trap`: the wrong passage shares surface keywords or entities with the query but does not answer the requested relation.
- `semantic_paraphrase`: the gold passage answers through paraphrase, synonymy, or indirect wording that surface matching underweights.
- `background_only`: the wrong passage is on-topic background but lacks the specific answer.
- `multi_evidence_confusion`: several passages are partially relevant, but the top passage is less complete than the gold evidence.
- `small_model_semantic_limit`: lightweight MLP/TextCNN rankings miss a fine-grained semantic relation that stronger rerankers recover.
- `llm_overjudgment`: an LLM-style reranker overvalues plausible, fluent, or weakly related evidence.
- `candidate_or_label_issue`: the candidate pool or gold label is materially debatable.
- `ambiguous_query`: the query lacks enough constraints, allowing multiple reasonable interpretations.
- `evidence_utilization_failure`: gold evidence is in downstream top-k, but the generator ignores it, uses the wrong evidence, or fails the answer format/metric.

## 4. Annotation Procedure

Method names were first blinded as Method A-E. Round 1 assigned one primary type per case, with an optional secondary subtype for evidence-utilization cases. A seed=42 sample of 10 cases was reread for repeated annotation self-consistency. This is single-annotator self-consistency, not inter-annotator or human-agreement measurement.

## 5. Quantitative Summary

- Cases by bucket: `{'A': 6, 'B': 6, 'C': 6, 'D': 6, 'E': 6}`.
- Cases by primary type: `{'ambiguous_query': 3, 'background_only': 2, 'candidate_or_label_issue': 3, 'evidence_utilization_failure': 6, 'lexical_trap': 4, 'llm_overjudgment': 4, 'multi_evidence_confusion': 2, 'small_model_semantic_limit': 6}`.
- Confidence distribution: `{'high': 16, 'medium': 14}`.
- Reannotation sample size: 10.
- Primary label agreement: 0.700.
- Secondary subtype agreement: 1.000.

## 6. Representative Cases

The table intentionally spans the five selection buckets instead of listing the easiest examples only.

| Case | Query ID | Bucket | Query | Error type | Judgment |
| --- | --- | --- | --- | --- | --- |
| G019 | msmarco_test_000199 | A | what is gamification in marketing | lexical_trap | The query asks: what is gamification in marketing. The top-ranked passage emphasizes 'I've been trying to think of a way to explain what gamification is with... |
| G025 | msmarco_test_000001 | B | how long does it take to drive from st louis to detroit | small_model_semantic_limit | The query asks: how long does it take to drive from st louis to detroit. The top-ranked passage emphasizes 'How long is the drive from Detroit, MI to Saint L... |
| G015 | msmarco_test_000344 | C | passing of stool with gas and pain many times a day | multi_evidence_confusion | The query asks: passing of stool with gas and pain many times a day. The top-ranked passage emphasizes 'Bowel habits, including the number of times a person ... |
| G013 | msmarco_test_000050 | C | average house build cost | llm_overjudgment | The query asks: average house build cost. The top-ranked passage emphasizes 'The average cost to build a log home in the U.S. is approximately *$150.00-$200.... |
| G007 | msmarco_test_000091 | D | what is single malt whisky | candidate_or_label_issue | The query asks: what is single malt whisky. The top-ranked passage emphasizes 'Single malt Scotch is single malt whisky made in Scotland. To be a single malt... |
| G001 | msmarco_test_000002 | E | what is cetirizine hydrochloride | evidence_utilization_failure | The query asks: what is cetirizine hydrochloride. The selected top-3 context includes evidence such as 'Cetirizine hydrochloride is the generic name for a pr... |
| G020 | msmarco_test_000396 | A | what is a brachiopod | lexical_trap | The query asks: what is a brachiopod. The top-ranked passage emphasizes 'Definition of BRACHIOPOD. : any of a phylum (Brachiopoda) of marine invertebrates wi... |
| G021 | msmarco_test_000028 | A | what is codeine | background_only | The query asks: what is codeine. The top-ranked passage emphasizes 'Codeine is the second-most predominant alkaloid in opium, at up to three percent. Althoug... |

Full passages and method-specific top-1 evidence are kept in `outputs/error_taxonomy_cases.csv`.

## 7. Main Findings

- BM25 failures in the selected sample mostly appear when surface overlap points to a topically plausible but answer-incomplete passage.
- Jittor MLP/TextCNN failures are concentrated in cases where BM25 or stronger semantic rerankers can still identify the gold passage, indicating limited fine-grained semantic discrimination in the lightweight models.
- LoRA 10k-rerun improves many lexical mismatch cases over BM25, but selected divergence cases show that it can still overvalue weakly related passages.
- Cross-Encoder is often the stronger semantic reranker in the selected divergence cases, but it is not immune to multi-evidence confusion or ambiguous labels.
- When all core methods fail, the case often needs review for candidate-pool, label, or query ambiguity rather than only model capacity.
- Evidence-utilization failures show that ranking success does not guarantee answer success: the generator can use a wrong top passage, ignore gold evidence in context, or produce a metric-mismatched answer.

## 8. Limitations

- The diagnostic sample is about 30 cases and intentionally stratified.
- Counts in this document describe selected cases only, not global error rates.
- There is one annotator; self-consistency is not inter-annotator agreement.
- Some methods have richer per-query outputs than others.
- Some gold labels and short queries remain debatable.
- Downstream evidence-utilization analysis is limited to the existing Qwen2.5-1.5B strict-short-answer slice.
