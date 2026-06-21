# Qwen2.5 JittorLLM Full-run Case Study

This case study samples top-1 outcomes from the 500-query runs. Scores use `logit("1") - logit("0")`; higher is ranked earlier.

## Qwen2.5-0.5B Cases

### Top-1 Correct Cases

#### msmarco_test_000006

Query: average gas in harrisonburg, va

Top-1 passage: `msmarco_test_000006_pos_1`, score `0.046875`

Positive passage: `msmarco_test_000006_pos_1`, score `0.046875`

The top passage states the average regular gas price in Harrisonburg, VA and directly answers the query. The score margin is small, but the positive still ranks first.

#### msmarco_test_000007

Query: what is the incubation period for chlamydia

Top-1 passage: `msmarco_test_000007_pos_3`, score `1.703125`

Positive passage: `msmarco_test_000007_pos_3`, score `1.703125`

The top passage explicitly says symptoms usually appear 1 to 3 weeks after sexual contact, matching the requested incubation period.

#### msmarco_test_000009

Query: how much do marketers earn

Top-1 passage: `msmarco_test_000009_pos_8`, score `0.765625`

Positive passage: `msmarco_test_000009_pos_8`, score `0.765625`

The selected passage provides annual and median wage numbers for marketing managers, so the ranking succeeds on a direct numeric answer.

#### msmarco_test_000012

Query: escitalopram dosage for depression

Top-1 passage: `msmarco_test_000012_pos_7`, score `0.578125`

Positive passage: `msmarco_test_000012_pos_7`, score `0.578125`

The top passage gives Lexapro starting dosage and dose range for depression, which is a direct answer despite using the brand name.

#### msmarco_test_000014

Query: gastroscopy cost

Top-1 passage: `msmarco_test_000014_pos_6`, score `1.437500`

Positive passage: `msmarco_test_000014_pos_6`, score `1.437500`

The selected passage contains a clear cost range and repeats the gastroscopy-cost framing, making it an easy match for the scorer.

### Top-1 Incorrect Cases

#### msmarco_test_000001

Query: how long does it take to drive from st louis to detroit

Top-1 passage: `msmarco_test_000001_neg_7`, score `1.031250`

Positive passage: `msmarco_test_000001_pos_8`, score `1.031250`

Top-1 excerpt: 552 mi (888km) - about 8 hours 39 mins Saint Louis, MO, USA to Detroit, MI, USA.

Positive excerpt: How long is the drive from Detroit, MI to Saint Louis, MO? The total driving time is 8 hours, 43 minutes.

Analysis: the negative and positive have essentially the same entity pair and travel-time format. They also tie on score, so the stored stable ordering places the negative first.

#### msmarco_test_000002

Query: what is cetirizine hydrochloride

Top-1 passage: `msmarco_test_000002_neg_7`, score `1.953125`

Positive passage: `msmarco_test_000002_pos_1`, score `1.078125`

Top-1 excerpt: Prior to taking cetirizine, it is a good idea to review the drug's safety information. ... Cetirizine Hydrochloride.

Positive excerpt: Cetirizine hydrochloride is the generic name for a prescription and over-the-counter antihistamine medication.

Analysis: the negative contains the query phrase and useful drug context, but it focuses on safety information rather than the definition. The model over-rewards topical overlap.

#### msmarco_test_000003

Query: how much does it cost to join the masters golf club

Top-1 passage: `msmarco_test_000003_neg_8`, score `0.531250`

Positive passage: `msmarco_test_000003_pos_1`, score `-0.671875`

Top-1 excerpt: The Bridge in Bridgehampton, N.Y., charged a whopping $750,000 to join ... Augusta wasn't talking either ...

Positive excerpt: Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiation fee is between $10,000 and $30,000 ...

Analysis: the negative discusses expensive golf-club membership fees and mentions Augusta, so it looks strongly related even though the positive gives the requested estimate.

#### msmarco_test_000004

Query: which blood test is for folate

Top-1 passage: `msmarco_test_000004_neg_7`, score `0.375000`

Positive passage: `msmarco_test_000004_pos_5`, score `0.078125`

Top-1 excerpt: The folate (folic acid) test measures the amount of folate in the blood.

Positive excerpt: A folic acid test is usually ordered to check for a lack of folate or Vitamin B9.

Analysis: both passages answer the query closely. The negative is likely a dataset-hard negative rather than a clearly irrelevant passage, which exposes label noise or near-duplicate ambiguity.

#### msmarco_test_000005

Query: what is the cause of HPV

Top-1 passage: `msmarco_test_000005_neg_7`, score `0.921875`

Positive passage: `msmarco_test_000005_pos_1`, score `0.468750`

Top-1 excerpt: HPV stands for human papillomavirus. These viruses are the direct causes of HPV and infect the skin or mucous membranes.

Positive excerpt: Human papilloma virus (HPV) is the major cause of cervical cancer.

Analysis: the query asks about the cause of HPV, but the positive passage answers the cause of cervical cancer. The scorer prefers the passage that literally explains HPV itself, showing that some judged labels are not aligned with a strict reading of the query.

## Qwen2.5-1.5B Cases

### Top-1 Correct Cases

#### msmarco_test_000001

Query: how long does it take to drive from st louis to detroit

Top-1 passage: `msmarco_test_000001_pos_8`, score `9.703125`

Positive passage: `msmarco_test_000001_pos_8`, score `9.703125`

The top passage gives the exact driving time. This is also a case where 1.5B fixes a 0.5B tie/ordering failure.

#### msmarco_test_000002

Query: what is cetirizine hydrochloride

Top-1 passage: `msmarco_test_000002_pos_1`, score `5.125000`

Positive passage: `msmarco_test_000002_pos_1`, score `5.125000`

The passage directly defines cetirizine hydrochloride as an antihistamine medication and explains its use.

#### msmarco_test_000004

Query: which blood test is for folate

Top-1 passage: `msmarco_test_000004_pos_5`, score `6.718750`

Positive passage: `msmarco_test_000004_pos_5`, score `6.718750`

The selected passage names the folic acid test and describes it as a simple blood test for folate or vitamin B9.

#### msmarco_test_000006

Query: average gas in harrisonburg, va

Top-1 passage: `msmarco_test_000006_pos_1`, score `1.703125`

Positive passage: `msmarco_test_000006_pos_1`, score `1.703125`

The selected passage states the average regular gas price in Harrisonburg, VA and remains a straightforward factoid success.

#### msmarco_test_000008

Query: how tall was lebron james parents

Top-1 passage: `msmarco_test_000008_pos_4`, score `1.156250`

Positive passage: `msmarco_test_000008_pos_4`, score `1.156250`

The passage contains the parent heights mentioned in the query context, so the model ranks it first.

### Top-1 Incorrect Cases

#### msmarco_test_000003

Query: how much does it cost to join the masters golf club

Top-1 passage: `msmarco_test_000003_neg_7`, score `4.187500`

Positive passage: `msmarco_test_000003_pos_1`, score `1.718750`

Top-1 excerpt: In 2009, Golf World magazine published an article titled Inside Augusta National Golf Club ... the initiation fee is in the low five-figures.

Positive excerpt: Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiation fee is between $10,000 and $30,000.

Analysis: the top negative is very close to the query and provides an apparent direct answer. This is best treated as a potential false negative or incomplete label, not as a simple model hallucination.

#### msmarco_test_000005

Query: what is the cause of HPV

Top-1 passage: `msmarco_test_000005_neg_4`, score `8.421875`

Positive passage: `msmarco_test_000005_pos_1`, score `5.937500`

Top-1 excerpt: Certain HPV types are classified as high-risk because they lead to abnormal cell changes and can cause genital cancers.

Positive excerpt: Human papilloma virus (HPV) is the major cause of cervical cancer.

Analysis: both passages discuss HPV and cancer causality. The query wording is ambiguous, and the negative is topically strong.

#### msmarco_test_000007

Query: what is the incubation period for chlamydia

Top-1 passage: `msmarco_test_000007_neg_4`, score `8.109375`

Positive passage: `msmarco_test_000007_pos_3`, score `7.046875`

Top-1 excerpt: The incubation period for chlamydia is quite variable and may range from days to months after the initial exposure.

Positive excerpt: The time between exposure to chlamydia and the start of symptoms-the incubation period-may range from days to months.

Analysis: the top negative directly answers the query. This is another likely false-negative or duplicate-answer case.

#### msmarco_test_000009

Query: how much do marketers earn

Top-1 passage: `msmarco_test_000009_neg_3`, score `8.390625`

Positive passage: `msmarco_test_000009_pos_8`, score `5.250000`

Top-1 excerpt: Most chief marketers earn between $100,000 and $350,000 per year.

Positive excerpt: Marketing managers ... earned an average annual wage of $126,190 as of May 2011.

Analysis: the top passage answers a closely related role, chief marketers, with a salary range. The model prefers this direct salary span even though the labeled positive gives the target occupational statistic.

#### msmarco_test_000010

Query: does using a proxy server hide from your isp?

Top-1 passage: `msmarco_test_000010_neg_8`, score `7.921875`

Positive passage: `msmarco_test_000010_pos_5`, score `3.921875`

Top-1 excerpt: you cannot hide from your ISP as all internet traffic will go through your ISP no matter if you use a proxy server or not.

Positive excerpt: they can always tell that you're connecting to one of these proxying or anonymization services.

Analysis: the selected negative appears to answer the query directly. This case should be interpreted as a potential labeling incompleteness rather than a clear semantic failure.

## Summary

The successful examples are straightforward factoid matches. The failures are mostly high-overlap negatives, answer-type confusions, tied scores, or passages marked negative that can still answer the query. This supports the main report's conclusion: Qwen2.5 zero-shot scoring is useful as a Jittor LLM inference proof of concept, and 1.5B improves over 0.5B, but task-specific training remains important.
