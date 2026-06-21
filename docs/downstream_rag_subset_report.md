# Downstream RAG QA Subset Report

Dataset path: `data/processed/msmarco_downstream_qa_50.jsonl`
Source processed test path: `data/processed/msmarco_medium/test.jsonl`
Gold answer source: `microsoft/ms_marco:v1.1` validation, aligned by streaming order.

| Item | Value |
| --- | ---: |
| Requested questions | 50 |
| Kept questions | 50 |
| Source rows scanned | 10000 |
| Excluded: fewer_than_top_k_candidates | 1 |
| Excluded: no_usable_gold_answer | 25 |
| Excluded: gold_not_in_candidate_pool | 9 |

Question type distribution:

| Type | Count |
| --- | ---: |
| how many/much | 5 |
| other | 19 |
| what | 22 |
| where | 1 |
| who | 1 |
| yes/no | 2 |

Selection rules:
- Gold answer must be non-empty.
- Gold answer is kept short enough for automatic evaluation.
- Real-time/current-information queries are filtered by simple keyword rules.
- At least one candidate passage in the reused MS MARCO medium candidate pool must contain the gold answer string after normalization.
- The original `data/processed/msmarco_medium/test.jsonl` file is not modified.
