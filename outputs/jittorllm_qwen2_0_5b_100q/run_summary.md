# Qwen2.5-0.5B JittorLLM Reranking Run

- Status: ready
- Backend: qwen2_jittorllms
- Model: Qwen2.5-0.5B-Instruct
- Dataset: `data/processed/msmarco_medium/test.jsonl`
- Num queries: 100
- Num candidate pairs: 807
- Cached pairs reused: 158
- New inference calls: 649
- Scoring method: label_token_logit_margin
- Parse failures: 0 (0.00%)
- Model inference time: 26.221s
- Pairs per second: 24.7512
- Peak observed GPU memory: 1152 MiB
- Label-token logit scoring: available
- Generated-label fallback: not used

## Metrics

| Metric | Value |
| --- | ---: |
| recall@1 | 0.1400 |
| recall@3 | 0.5200 |
| recall@5 | 0.7200 |
| recall@10 | 1.0000 |
| ndcg@1 | 0.1400 |
| ndcg@3 | 0.3549 |
| ndcg@5 | 0.4375 |
| ndcg@10 | 0.5317 |
| mrr | 0.3853 |
| pairwise_accuracy | 0.5896 |

## Score Statistics

| Statistic | Value |
| --- | ---: |
| min | -2.031250 |
| max | 3.703125 |
| mean | 0.028733 |
| std | 0.750920 |
| positive mean | 0.134531 |
| negative mean | 0.013769 |
| tie groups | 155 |
| tied items | 759 |
| tied pairs | 1954 |

## Prompt

```text
Decide whether the passage directly answers the search query.
Return only one digit:
1 = relevant and directly answers the query
0 = irrelevant or does not directly answer the query

Query: {query}
Passage: {passage}
Label:

```

## Limitations

This is label-token logit-margin zero-shot scoring on a small subset. It is a Jittor LLM inference proof of concept, not a replacement for a trained cross-encoder.

## Correct Top-1 Cases

- `msmarco_test_000006` / `msmarco_test_000006_pos_1` score=0.046875: average gas in harrisonburg, va | 8 hours ago. $2.05update. There are 31 Regular gas price reports in the past 5 days in Harrisonburg, VA. The average Regular gas price in Harrisonburg, VA is $2.03, which is $0.73 lower than U.S. national average Regular gas price $2.76. Th
- `msmarco_test_000007` / `msmarco_test_000007_pos_3` score=1.703125: what is the incubation period for chlamydia | Chlamydia-Symptoms. Most women and men with chlamydia do not have symptoms. The time between exposure to chlamydia and the start of symptoms-the incubation period-may range from days to months. If symptoms appear, it is usually 1 to 3 weeks
- `msmarco_test_000009` / `msmarco_test_000009_pos_8` score=0.765625: how much do marketers earn | Salary. The Bureau of Labor Statistics reports that marketing managers in the United States earned an average annual wage of $126,190 as of May 2011. The median wage, or midway point between the lowest and highest salary, was $116,010 per y
- `msmarco_test_000012` / `msmarco_test_000012_pos_7` score=0.578125: escitalopram dosage for depression | Dosing Recommendations for Lexapro. The recommended starting dose of Lexapro for adults or adolescents (age 12 to 17 years old) with depression or adults with generalized anxiety disorder is 10 mg once a day. Your healthcare provider may ch
- `msmarco_test_000014` / `msmarco_test_000014_pos_6` score=1.437500: gastroscopy cost | In U.S. hospitals gastroscopy cost can range from $350 to $1,200, when extra pathology and surgical services are involved, whereas, abroad expenses may be just a fraction of this sum. Get exclusive information about [page_title]. Our profes

## Incorrect Top-1 Cases

- `msmarco_test_000001` / `msmarco_test_000001_neg_7` score=1.031250: how long does it take to drive from st louis to detroit | 552 mi (888km) - about 8 hours 39 mins Saint Louis, MO, USA to Detroit, MI, USA.
- `msmarco_test_000002` / `msmarco_test_000002_neg_7` score=1.953125: what is cetirizine hydrochloride | Prior to taking cetirizine, it is a good idea to review the drug's safety information. For example, side effects such as fatigue, drowsiness, and dry mouth may occur. In addition, this medicine is not right for certain people, such as those
- `msmarco_test_000003` / `msmarco_test_000003_neg_8` score=0.531250: how much does it cost to join the masters golf club | The Bridge in Bridgehampton, N.Y., charged a whopping $750,000 to join, according to a 2007 Wall Street Journal article. Another Hamptons club, Sebonack Golf Club, billed new members for $650,000 when it opened in 2006, Bloomberg reported.
- `msmarco_test_000004` / `msmarco_test_000004_neg_7` score=0.375000: which blood test is for folate | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount of folate in the blood. Folate and folic
- `msmarco_test_000005` / `msmarco_test_000005_neg_7` score=0.921875: what is the cause of HPV | HPV stands for human papillomavirus. These viruses are the direct causes of HPV and infect the skin or mucous membranes (mucous membranes are the moist, inner lining of some organs and body cavities, such as the nose, mouth, lungs, and stom
