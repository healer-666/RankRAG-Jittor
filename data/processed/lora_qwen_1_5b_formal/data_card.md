# LoRA reranker debug data

| Split | Queries | Pairs/Candidates | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| train | 1007 | 5000 | 1007 | 3996 |
| valid | 100 | 500 | 100 | 400 |
| test | 500 | 4044 | 500 | 3544 |

Labels are copied from the processed MS MARCO data. No synthetic relevance labels are created.

Generation arguments:

- `train_in`: `data/processed/msmarco_medium/train.jsonl`
- `valid_in`: `data/processed/msmarco_medium/valid.jsonl`
- `test_in`: `data/processed/msmarco_medium/test.jsonl`
- `out_dir`: `data/processed/lora_qwen_1_5b_formal`
- `max_train_pairs`: `5000`
- `max_valid_pairs`: `500`
- `max_test_queries`: `500`
- `negatives_per_query`: `4`
- `seed`: `42`
