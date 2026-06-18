# MS MARCO Small Subset

Data source: `microsoft/ms_marco:v1.1`.

This is a small MS MARCO query-passage ranking subset prepared for the lightweight RankRAG-style context ranking reproduction.
It is not the full MS MARCO dataset and is not a leaderboard submission setup.

Construction:
- Keep queries with at least one positive passage and one negative passage.
- Positive passages come from `is_selected == 1` or official positive fields when available.
- Negative passages come from same-query unselected passages or official negative fields.
- Keep at most `5` candidates per query.
- Use deterministic streaming/sample order with seed-controlled candidate shuffling.

| Split | Queries | Avg candidates | Positives | Negatives |
| --- | ---: | ---: | ---: | ---: |
| train | 1000 | 4.97 | 1000 | 3968 |
| valid | 200 | 5.00 | 200 | 799 |
| test | 200 | 4.99 | 200 | 798 |
