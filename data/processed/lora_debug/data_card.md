# LoRA reranker debug data

| Split | Queries | Pairs/Candidates | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| train | 67 | 200 | 67 | 134 |
| valid | 34 | 100 | 34 | 68 |
| test | 20 | 158 | 20 | 138 |

Labels are copied from the processed MS MARCO data. No synthetic relevance labels are created.

Generation arguments:

- `train_in`: `data/processed/msmarco_medium/train.jsonl`
- `valid_in`: `data/processed/msmarco_medium/valid.jsonl`
- `test_in`: `data/processed/msmarco_medium/test.jsonl`
- `out_dir`: `data/processed/lora_debug`
- `max_train_pairs`: `200`
- `max_valid_pairs`: `100`
- `max_test_queries`: `20`
- `negatives_per_query`: `2`
- `seed`: `42`
