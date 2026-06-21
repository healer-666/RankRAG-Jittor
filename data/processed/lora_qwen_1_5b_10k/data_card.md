# LoRA reranker debug data

| Split | Queries | Pairs/Candidates | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| train | 2011 | 10000 | 2011 | 7989 |
| valid | 201 | 1000 | 201 | 803 |
| test | 500 | 4044 | 500 | 3544 |

Labels are copied from the processed MS MARCO data. No synthetic relevance labels are created.

Generation arguments:

- `train_in`: `data/processed/msmarco_medium/train.jsonl`
- `valid_in`: `data/processed/msmarco_medium/valid.jsonl`
- `test_in`: `data/processed/msmarco_medium/test.jsonl`
- `out_dir`: `data/processed/lora_qwen_1_5b_10k`
- `max_train_pairs`: `10000`
- `max_valid_pairs`: `1000`
- `max_test_queries`: `500`
- `negatives_per_query`: `4`
- `seed`: `42`
