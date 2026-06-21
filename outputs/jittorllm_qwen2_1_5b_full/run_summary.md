# Qwen2.5-0.5B JittorLLM Reranking Run

- Status: ready
- Backend: qwen2_jittorllms
- Model: Qwen2.5-1.5B-Instruct
- Dataset: `data/processed/msmarco_medium/test.jsonl`
- Num queries: 500
- Num candidate pairs: 4044
- Cached pairs reused: 0
- New inference calls: 4044
- Scoring method: label_token_logit_margin
- Parse failures: 0 (0.00%)
- Model inference time: 114.582s
- Pairs per second: 35.2936
- Peak observed GPU memory: 3608 MiB
- Label-token logit scoring: available
- Generated-label fallback: not used

## Metrics

| Metric | Value |
| --- | ---: |
| recall@1 | 0.2360 |
| recall@3 | 0.5520 |
| recall@5 | 0.8120 |
| recall@10 | 1.0000 |
| ndcg@1 | 0.2360 |
| ndcg@3 | 0.4139 |
| ndcg@5 | 0.5210 |
| ndcg@10 | 0.5832 |
| mrr | 0.4525 |
| pairwise_accuracy | 0.6342 |

## Score Statistics

| Statistic | Value |
| --- | ---: |
| min | -8.156250 |
| max | 10.062500 |
| mean | 1.410160 |
| std | 2.897395 |
| positive mean | 2.381719 |
| negative mean | 1.273089 |
| tie groups | 683 |
| tied items | 3896 |
| tied pairs | 12094 |

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

- `msmarco_test_000001` / `msmarco_test_000001_pos_8` score=9.703125: how long does it take to drive from st louis to detroit | How long is the drive from Detroit, MI to Saint Louis, MO? The total driving time is 8 hours, 43 minutes.
- `msmarco_test_000002` / `msmarco_test_000002_pos_1` score=5.125000: what is cetirizine hydrochloride | Cetirizine hydrochloride is the generic name for a prescription and over-the-counter antihistamine medication. Doctors recommend cetirizine hydrochloride for the relief of symptoms associated with allergies. Cetirizine hydrochloride is avai
- `msmarco_test_000004` / `msmarco_test_000004_pos_5` score=6.718750: which blood test is for folate | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. The procedure consists of a simple blood t
- `msmarco_test_000006` / `msmarco_test_000006_pos_1` score=1.703125: average gas in harrisonburg, va | 8 hours ago. $2.05update. There are 31 Regular gas price reports in the past 5 days in Harrisonburg, VA. The average Regular gas price in Harrisonburg, VA is $2.03, which is $0.73 lower than U.S. national average Regular gas price $2.76. Th
- `msmarco_test_000008` / `msmarco_test_000008_pos_4` score=1.156250: how tall was lebron james parents | I'm always interested in the height of really tall people's parents because I'm wondering if their height is genetic. Kobe Bryant's father's height is 6'9 so his height is obviously genetic. I heard that Lebron James' dad is 5'7 and his mom

## Incorrect Top-1 Cases

- `msmarco_test_000003` / `msmarco_test_000003_neg_7` score=4.187500: how much does it cost to join the masters golf club | In 2009, Golf World magazine published an article titled Inside Augusta National Golf Club .. A club member (speaking anonymously) told the magazine that the initiation fee is in the low five-figures.. So definitely less than $50,000, and w
- `msmarco_test_000005` / `msmarco_test_000005_neg_4` score=8.421875: what is the cause of HPV | Certain HPV types are classified as high-risk because they lead to abnormal cell changes and can cause genital cancers: cervical cancer as well as cancer of the vulva, anus, and penis. In fact, researchers say that virtually all cervical ca
- `msmarco_test_000007` / `msmarco_test_000007_neg_4` score=8.109375: what is the incubation period for chlamydia | Therefore, they do not know they have the disease. The incubation period for chlamydia is quite variable and may range from days to months after the initial exposure. The average time from exposure to the development of symptoms is usually
- `msmarco_test_000009` / `msmarco_test_000009_neg_3` score=8.390625: how much do marketers earn | Most chief marketers earn between $100,000 and $350,000 per year, according to a recent report from The CMO Council.
- `msmarco_test_000010` / `msmarco_test_000010_neg_8` score=7.921875: does using a proxy server hide from your isp? | Another user mentioned to use proxy servers to hide your tracks, but you cannot hide from your ISP as all internet traffic will go through your ISP no matter if you use a proxy server or not; to connect to a proxy server, you still use your
