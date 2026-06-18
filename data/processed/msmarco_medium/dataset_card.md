# MS MARCO Medium Subset

Data source: `microsoft/ms_marco:v1.1`.
Run name: `msmarco_medium`.

This is an MS MARCO v1.1 small/medium query-passage ranking subset prepared for the lightweight RankRAG-style context ranking reproduction.
It is not the full MS MARCO dataset and is not a leaderboard submission setup.
This medium subset is larger than the initial small subset and increases candidates_per_query from 5 to 10, making Recall@5 and NDCG@5 more discriminative.

Construction:
- Keep queries with at least one positive passage and one negative passage.
- Positive passages come from `is_selected == 1` or official positive fields when available.
- Negative passages come from same-query unselected passages or official negative fields.
- Keep at most `10` candidates per query.
- Use deterministic streaming/sample order with seed-controlled candidate shuffling.

| Split | Queries | Avg candidates | Positives | Negatives |
| --- | ---: | ---: | ---: | ---: |
| train | 5000 | 8.11 | 5000 | 35567 |
| valid | 500 | 8.16 | 500 | 3580 |
| test | 500 | 8.09 | 500 | 3544 |
