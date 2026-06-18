# MS MARCO Case Study and Error Analysis

These examples are selected from the medium MS MARCO subset and use automatic rule-based notes. They should be read as qualitative diagnostics, not proof of generalization.

The comparison highlights how BM25, MLP Jittor, and TextCNN Jittor behave on the same candidate set. Strong keyword overlap often helps BM25, while the lightweight neural rankers can still struggle with hard negatives because they do not use pretrained semantic encoders or LLM-style reranking.

## Success Cases

### msmarco_test_000011

Query: what is the structure of vyvanse

**TF-IDF** positive rank: 6

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (is, of, the, vyvanse), which can help lexical baselines.

- rank 1 | msmarco_test_000011_neg_9 | label=0 | score=0.0766 | 3.2.1 Pharmacology. Vyvanse ™ (lisdexamfetamine dimesylate) is a prodrug of d-amphetamine. After oral administration, Vyvanse ™ is converted to d-amphetamine, the compound responsible for the drug's t
- rank 2 | msmarco_test_000011_neg_7 | label=0 | score=0.0732 | Vyvanse comes in the form of a capsule that is usually taken once a day in the morning. Advertisement. Vyvanse™ (lisdexamfetamine dimesylate) is a prescription medication that is used for the treatmen
- rank 3 | msmarco_test_000011_neg_6 | label=0 | score=0.0703 | 30mg Vyvanse capsules. Lisdexamfetamine (contracted from L-lysine-dextroamphetamine) is a central nervous system (CNS) stimulant and dextroamphetamine prodrug of the phenethylamine class and amphetami
- rank 4 | msmarco_test_000011_neg_8 | label=0 | score=0.0549 | Adderall is a brand name of amphetamine salts-based medication used for attention deficit hyperactivity disorder and narcolepsy, legal only in the United States and Canada. Vyvanse is a brand name of 
- rank 5 | msmarco_test_000011_neg_5 | label=0 | score=0.0532 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). Vyvanse capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (is, of, structure, the, vyvanse), which can help lexical baselines.

- rank 1 | msmarco_test_000011_neg_5 | label=0 | score=2.7225 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). Vyvanse capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg
- rank 2 | msmarco_test_000011_pos_1 | label=1 | score=2.5197 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). VYVANSE capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg
- rank 3 | msmarco_test_000011_neg_9 | label=0 | score=2.0876 | 3.2.1 Pharmacology. Vyvanse ™ (lisdexamfetamine dimesylate) is a prodrug of d-amphetamine. After oral administration, Vyvanse ™ is converted to d-amphetamine, the compound responsible for the drug's t
- rank 4 | msmarco_test_000011_neg_7 | label=0 | score=1.9898 | Vyvanse comes in the form of a capsule that is usually taken once a day in the morning. Advertisement. Vyvanse™ (lisdexamfetamine dimesylate) is a prescription medication that is used for the treatmen
- rank 5 | msmarco_test_000011_neg_6 | label=0 | score=1.9065 | 30mg Vyvanse capsules. Lisdexamfetamine (contracted from L-lysine-dextroamphetamine) is a central nervous system (CNS) stimulant and dextroamphetamine prodrug of the phenethylamine class and amphetami

**MLP Jittor** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (is, of, the, vyvanse), which can help lexical baselines.

- rank 1 | msmarco_test_000011_neg_8 | label=0 | score=-0.5808 | Adderall is a brand name of amphetamine salts-based medication used for attention deficit hyperactivity disorder and narcolepsy, legal only in the United States and Canada. Vyvanse is a brand name of 
- rank 2 | msmarco_test_000011_pos_1 | label=1 | score=-1.9632 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). VYVANSE capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg
- rank 3 | msmarco_test_000011_neg_7 | label=0 | score=-2.1294 | Vyvanse comes in the form of a capsule that is usually taken once a day in the morning. Advertisement. Vyvanse™ (lisdexamfetamine dimesylate) is a prescription medication that is used for the treatmen
- rank 4 | msmarco_test_000011_neg_5 | label=0 | score=-2.7892 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). Vyvanse capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg
- rank 5 | msmarco_test_000011_neg_3 | label=0 | score=-3.4146 | Lisdexamfetamine is used to treat attention deficit hyperactivity disorder (ADHD) as part of a total treatment plan, including psychological, social, and other treatments. It may help to increase the 

**TextCNN Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (is, of, structure, the, vyvanse), which can help lexical baselines.

- rank 1 | msmarco_test_000011_pos_1 | label=1 | score=6.8186 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). VYVANSE capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg
- rank 2 | msmarco_test_000011_neg_5 | label=0 | score=4.8276 | The chemical structure is: Lisdexamfetamine dimesylate is a white to off-white powder that is soluble in water (792 mg/mL). Vyvanse capsules contain 10 mg, 20 mg, 30 mg, 40 mg, 50 mg, 60 mg, and 70 mg
- rank 3 | msmarco_test_000011_neg_8 | label=0 | score=4.2584 | Adderall is a brand name of amphetamine salts-based medication used for attention deficit hyperactivity disorder and narcolepsy, legal only in the United States and Canada. Vyvanse is a brand name of 
- rank 4 | msmarco_test_000011_neg_9 | label=0 | score=3.7023 | 3.2.1 Pharmacology. Vyvanse ™ (lisdexamfetamine dimesylate) is a prodrug of d-amphetamine. After oral administration, Vyvanse ™ is converted to d-amphetamine, the compound responsible for the drug's t
- rank 5 | msmarco_test_000011_neg_3 | label=0 | score=2.9716 | Lisdexamfetamine is used to treat attention deficit hyperactivity disorder (ADHD) as part of a total treatment plan, including psychological, social, and other treatments. It may help to increase the 

### msmarco_test_000014

Query: gastroscopy cost

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is limited (cost, gastroscopy), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000014_pos_6 | label=1 | score=0.2008 | In U.S. hospitals gastroscopy cost can range from $350 to $1,200, when extra pathology and surgical services are involved, whereas, abroad expenses may be just a fraction of this sum. Get exclusive in
- rank 2 | msmarco_test_000014_neg_8 | label=0 | score=0.0937 | Cost-effectiveness analyses in dyspepsia management and the role of gastroscopy are difficult to interpret and impossible to compare due to a lack of uniformity in designing, measuring and reporting c
- rank 3 | msmarco_test_000014_neg_1 | label=0 | score=0.0481 | Cost of Upper Endoscopy. The cost of upper endoscopy (without insurance) may range from $800 to $1,800 and depends on country, hospital, procedures performed during investigation and the cost of sedat
- rank 4 | msmarco_test_000014_neg_7 | label=0 | score=0.0422 | 1 If it is necessary to take a biopsy, that would add a procedure cost and laboratory cost to the final bill. 2  This could add as much as several thousand dollars to the cost. 3  For example, at Good
- rank 5 | msmarco_test_000014_neg_9 | label=0 | score=0.0379 | 1 At Avcnh.com, the cost of an endoscopy costs $850. 2  On the other hand, a rhinoscopy will cost around $975. 3  According to one veterinarian that answered a question on Justanswer.com, the price of

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is limited (cost, gastroscopy), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000014_neg_8 | label=0 | score=2.0122 | Cost-effectiveness analyses in dyspepsia management and the role of gastroscopy are difficult to interpret and impossible to compare due to a lack of uniformity in designing, measuring and reporting c
- rank 2 | msmarco_test_000014_pos_6 | label=1 | score=1.9803 | In U.S. hospitals gastroscopy cost can range from $350 to $1,200, when extra pathology and surgical services are involved, whereas, abroad expenses may be just a fraction of this sum. Get exclusive in
- rank 3 | msmarco_test_000014_neg_1 | label=0 | score=0.5605 | Cost of Upper Endoscopy. The cost of upper endoscopy (without insurance) may range from $800 to $1,800 and depends on country, hospital, procedures performed during investigation and the cost of sedat
- rank 4 | msmarco_test_000014_neg_7 | label=0 | score=0.5238 | 1 If it is necessary to take a biopsy, that would add a procedure cost and laboratory cost to the final bill. 2  This could add as much as several thousand dollars to the cost. 3  For example, at Good
- rank 5 | msmarco_test_000014_neg_9 | label=0 | score=0.5137 | 1 At Avcnh.com, the cost of an endoscopy costs $850. 2  On the other hand, a rhinoscopy will cost around $975. 3  According to one veterinarian that answered a question on Justanswer.com, the price of

**MLP Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is limited (cost, gastroscopy), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000014_pos_6 | label=1 | score=0.1760 | In U.S. hospitals gastroscopy cost can range from $350 to $1,200, when extra pathology and surgical services are involved, whereas, abroad expenses may be just a fraction of this sum. Get exclusive in
- rank 2 | msmarco_test_000014_neg_9 | label=0 | score=-2.4916 | 1 At Avcnh.com, the cost of an endoscopy costs $850. 2  On the other hand, a rhinoscopy will cost around $975. 3  According to one veterinarian that answered a question on Justanswer.com, the price of
- rank 3 | msmarco_test_000014_neg_7 | label=0 | score=-2.8625 | 1 If it is necessary to take a biopsy, that would add a procedure cost and laboratory cost to the final bill. 2  This could add as much as several thousand dollars to the cost. 3  For example, at Good
- rank 4 | msmarco_test_000014_neg_8 | label=0 | score=-3.1774 | Cost-effectiveness analyses in dyspepsia management and the role of gastroscopy are difficult to interpret and impossible to compare due to a lack of uniformity in designing, measuring and reporting c
- rank 5 | msmarco_test_000014_neg_5 | label=0 | score=-3.4286 | 1 On average, a dog endoscopy is going to cost anywhere from $800 to as much as $2,000. 2  At Avcnh.com, the cost of an endoscopy costs $850. 1 An initial cost of $70 to $100 for the vet examination f

**TextCNN Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is limited (cost, gastroscopy), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000014_pos_6 | label=1 | score=7.4583 | In U.S. hospitals gastroscopy cost can range from $350 to $1,200, when extra pathology and surgical services are involved, whereas, abroad expenses may be just a fraction of this sum. Get exclusive in
- rank 2 | msmarco_test_000014_neg_2 | label=0 | score=5.6016 | Upper GI Endoscopy = Esophago-Gastro-Duodenoscopy (EGD). Upper gastrointestinal (GI) endoscopy is a diagnostic procedure enabling your doctor to see inside your esophagus, stomach and duodenum (the fi
- rank 3 | msmarco_test_000014_neg_3 | label=0 | score=5.1187 | 1 For patients not covered by health insurance, upper gastrointestinal endoscopy typically costs between about $1,500-$10,000 or more, depending on the provider, geographic location, whether sedation 
- rank 4 | msmarco_test_000014_neg_7 | label=0 | score=5.0116 | 1 If it is necessary to take a biopsy, that would add a procedure cost and laboratory cost to the final bill. 2  This could add as much as several thousand dollars to the cost. 3  For example, at Good
- rank 5 | msmarco_test_000014_neg_4 | label=0 | score=5.0040 | 1 The national average cost is $2,700 -- but prices can range from $1,350 to $10,400, according to NewChoiceHealth.com. 2  At the Dartmouth-Hitchcock Medical Center in New Hampshire, a patient without

### msmarco_test_000021

Query: what are the only multicellular protists

**TF-IDF** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (are, multicellular, only, protists, the), which can help lexical baselines.

- rank 1 | msmarco_test_000021_neg_5 | label=0 | score=0.1280 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P
- rank 2 | msmarco_test_000021_pos_6 | label=1 | score=0.1268 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P
- rank 3 | msmarco_test_000021_neg_4 | label=0 | score=0.0847 | Kingdom Protista is a diverse group of eukaryotic organisms. Protists are unicellular, some are colonial or multicellular, they do not have specialized tissue organization. The simple cellular organiz
- rank 4 | msmarco_test_000021_neg_2 | label=0 | score=0.0736 | 1 Most of the organisms are unicellular, some are colonial and some are multicellular like algae. 2  Most of the protists live in water, some in moist soil or even the body of human and plants. 3  The
- rank 5 | msmarco_test_000021_neg_3 | label=0 | score=0.0703 | Volvox is one of several multicellular protists. This is kelp, a multicellular algae: Mycoprotists, or fungus-like protists, have characteristics of fungus cells. Examples: Water molds, which are unic

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (are, multicellular, only, protists, the), which can help lexical baselines.

- rank 1 | msmarco_test_000021_neg_5 | label=0 | score=1.5823 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P
- rank 2 | msmarco_test_000021_pos_6 | label=1 | score=1.5707 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P
- rank 3 | msmarco_test_000021_neg_4 | label=0 | score=1.0547 | Kingdom Protista is a diverse group of eukaryotic organisms. Protists are unicellular, some are colonial or multicellular, they do not have specialized tissue organization. The simple cellular organiz
- rank 4 | msmarco_test_000021_neg_2 | label=0 | score=0.9830 | 1 Most of the organisms are unicellular, some are colonial and some are multicellular like algae. 2  Most of the protists live in water, some in moist soil or even the body of human and plants. 3  The
- rank 5 | msmarco_test_000021_neg_1 | label=0 | score=0.9799 | The majority of organisms classified as protists are unicellular though there are a few multicellular organisms. For example, kelp (“seaweed”) is technically a protist even though it is multicellular.

**MLP Jittor** positive rank: 6

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (are, multicellular, protists), which can help lexical baselines.

- rank 1 | msmarco_test_000021_neg_3 | label=0 | score=-2.6135 | Volvox is one of several multicellular protists. This is kelp, a multicellular algae: Mycoprotists, or fungus-like protists, have characteristics of fungus cells. Examples: Water molds, which are unic
- rank 2 | msmarco_test_000021_neg_4 | label=0 | score=-3.2688 | Kingdom Protista is a diverse group of eukaryotic organisms. Protists are unicellular, some are colonial or multicellular, they do not have specialized tissue organization. The simple cellular organiz
- rank 3 | msmarco_test_000021_neg_1 | label=0 | score=-3.6856 | The majority of organisms classified as protists are unicellular though there are a few multicellular organisms. For example, kelp (“seaweed”) is technically a protist even though it is multicellular.
- rank 4 | msmarco_test_000021_neg_2 | label=0 | score=-5.2419 | 1 Most of the organisms are unicellular, some are colonial and some are multicellular like algae. 2  Most of the protists live in water, some in moist soil or even the body of human and plants. 3  The
- rank 5 | msmarco_test_000021_neg_5 | label=0 | score=-6.1816 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P

**TextCNN Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (are, multicellular, only, protists, the), which can help lexical baselines.

- rank 1 | msmarco_test_000021_pos_6 | label=1 | score=7.7604 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P
- rank 2 | msmarco_test_000021_neg_5 | label=0 | score=7.6197 | In Biology. The only type of multicellular protists are plant-like seaweeds  known as algae. There are three different types of algae that are  differentiated by color. The brown algae … is known as P
- rank 3 | msmarco_test_000021_neg_1 | label=0 | score=5.1848 | The majority of organisms classified as protists are unicellular though there are a few multicellular organisms. For example, kelp (“seaweed”) is technically a protist even though it is multicellular.
- rank 4 | msmarco_test_000021_neg_4 | label=0 | score=4.4539 | Kingdom Protista is a diverse group of eukaryotic organisms. Protists are unicellular, some are colonial or multicellular, they do not have specialized tissue organization. The simple cellular organiz
- rank 5 | msmarco_test_000021_neg_2 | label=0 | score=3.5769 | 1 Most of the organisms are unicellular, some are colonial and some are multicellular like algae. 2  Most of the protists live in water, some in moist soil or even the body of human and plants. 3  The

## Middle Cases

### msmarco_test_000004

Query: which blood test is for folate

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (blood, folate, for, is, test, which), which can help lexical baselines.

- rank 1 | msmarco_test_000004_pos_5 | label=1 | score=0.1707 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. Th
- rank 2 | msmarco_test_000004_neg_3 | label=0 | score=0.1464 | This test measures the levels of vitamin B12 and folate in your blood. Your body needs vitamin B12, also called cobalamin, and folate, also called folic acid, to function normally. Both nutrients play
- rank 3 | msmarco_test_000004_neg_7 | label=0 | score=0.0961 | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount 
- rank 4 | msmarco_test_000004_neg_8 | label=0 | score=0.0847 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 
- rank 5 | msmarco_test_000004_neg_9 | label=0 | score=0.0637 | A folic acid test measures the amount of folic acid in the blood. Folic acid is one of many B vitamins. The body needs folic acid to make red blood cells (RBC) , white blood cells (WBC) , and platelet

**BM25** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (blood, folate, for, is, test, which), which can help lexical baselines.

- rank 1 | msmarco_test_000004_pos_5 | label=1 | score=3.6124 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. Th
- rank 2 | msmarco_test_000004_neg_8 | label=0 | score=2.2926 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 
- rank 3 | msmarco_test_000004_neg_3 | label=0 | score=2.1767 | This test measures the levels of vitamin B12 and folate in your blood. Your body needs vitamin B12, also called cobalamin, and folate, also called folic acid, to function normally. Both nutrients play
- rank 4 | msmarco_test_000004_neg_7 | label=0 | score=1.8563 | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount 
- rank 5 | msmarco_test_000004_neg_4 | label=0 | score=1.5951 | Test Overview. A folic acid test measures the amount of folic acid in the blood. Folic acid is one of many B vitamins. The body needs folic acid to make red blood cells (RBC) , white blood cells (WBC)

**MLP Jittor** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (blood, folate, for, is, test), which can help lexical baselines.

- rank 1 | msmarco_test_000004_neg_3 | label=0 | score=1.9444 | This test measures the levels of vitamin B12 and folate in your blood. Your body needs vitamin B12, also called cobalamin, and folate, also called folic acid, to function normally. Both nutrients play
- rank 2 | msmarco_test_000004_neg_2 | label=0 | score=1.6911 | B12 and folate levels may be ordered when a complete blood count (CBC) and/or blood smear, done as part of a health checkup or an evaluation for anemia, indicates a low red blood cell (RBC) count with
- rank 3 | msmarco_test_000004_neg_6 | label=0 | score=1.2911 | Vitamin B12 and folate are separate tests often used in conjunction to detect deficiencies and to help diagnose the cause of certain anemias, such as pernicious anemia, an autoimmune disease that affe
- rank 4 | msmarco_test_000004_pos_5 | label=1 | score=0.2112 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. Th
- rank 5 | msmarco_test_000004_neg_8 | label=0 | score=0.0951 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 

**TextCNN Jittor** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (blood, folate, is, test), which can help lexical baselines.

- rank 1 | msmarco_test_000004_neg_7 | label=0 | score=7.6205 | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount 
- rank 2 | msmarco_test_000004_pos_5 | label=1 | score=6.7732 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. Th
- rank 3 | msmarco_test_000004_neg_6 | label=0 | score=6.4372 | Vitamin B12 and folate are separate tests often used in conjunction to detect deficiencies and to help diagnose the cause of certain anemias, such as pernicious anemia, an autoimmune disease that affe
- rank 4 | msmarco_test_000004_neg_2 | label=0 | score=6.4230 | B12 and folate levels may be ordered when a complete blood count (CBC) and/or blood smear, done as part of a health checkup or an evaluation for anemia, indicates a low red blood cell (RBC) count with
- rank 5 | msmarco_test_000004_neg_8 | label=0 | score=5.2367 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 

### msmarco_test_000006

Query: average gas in harrisonburg, va

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (average, gas, harrisonburg, in, va), which can help lexical baselines.

- rank 1 | msmarco_test_000006_pos_1 | label=1 | score=0.2842 | 8 hours ago. $2.05update. There are 31 Regular gas price reports in the past 5 days in Harrisonburg, VA. The average Regular gas price in Harrisonburg, VA is $2.03, which is $0.73 lower than U.S. nati
- rank 2 | msmarco_test_000006_neg_2 | label=0 | score=0.2821 | 23 hours ago. $2.40update. There are 29 Diesel gas price reports in the past 5 days in Harrisonburg, VA. The average Diesel gas price in Harrisonburg, VA is $3.17, which is $0.37 higher than U.S. nati

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (average, gas, harrisonburg, in, va), which can help lexical baselines.

- rank 1 | msmarco_test_000006_neg_2 | label=0 | score=-1.7674 | 23 hours ago. $2.40update. There are 29 Diesel gas price reports in the past 5 days in Harrisonburg, VA. The average Diesel gas price in Harrisonburg, VA is $3.17, which is $0.37 higher than U.S. nati
- rank 2 | msmarco_test_000006_pos_1 | label=1 | score=-1.7879 | 8 hours ago. $2.05update. There are 31 Regular gas price reports in the past 5 days in Harrisonburg, VA. The average Regular gas price in Harrisonburg, VA is $2.03, which is $0.73 lower than U.S. nati

**MLP Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (average, gas, harrisonburg, in, va), which can help lexical baselines.

- rank 1 | msmarco_test_000006_pos_1 | label=1 | score=-1.4150 | 8 hours ago. $2.05update. There are 31 Regular gas price reports in the past 5 days in Harrisonburg, VA. The average Regular gas price in Harrisonburg, VA is $2.03, which is $0.73 lower than U.S. nati
- rank 2 | msmarco_test_000006_neg_2 | label=0 | score=-2.9657 | 23 hours ago. $2.40update. There are 29 Diesel gas price reports in the past 5 days in Harrisonburg, VA. The average Diesel gas price in Harrisonburg, VA is $3.17, which is $0.37 higher than U.S. nati

**TextCNN Jittor** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (average, gas, harrisonburg, in, va), which can help lexical baselines.

- rank 1 | msmarco_test_000006_neg_2 | label=0 | score=5.1948 | 23 hours ago. $2.40update. There are 29 Diesel gas price reports in the past 5 days in Harrisonburg, VA. The average Diesel gas price in Harrisonburg, VA is $3.17, which is $0.37 higher than U.S. nati
- rank 2 | msmarco_test_000006_pos_1 | label=1 | score=4.5264 | 8 hours ago. $2.05update. There are 31 Regular gas price reports in the past 5 days in Harrisonburg, VA. The average Regular gas price in Harrisonburg, VA is $2.03, which is $0.73 lower than U.S. nati

### msmarco_test_000009

Query: how much do marketers earn

**TF-IDF** positive rank: 8

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is limited (earn, marketers), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000009_neg_3 | label=0 | score=0.1126 | Most chief marketers earn between $100,000 and $350,000 per year, according to a recent report from The CMO Council.
- rank 2 | msmarco_test_000009_neg_7 | label=0 | score=0.0558 | It depends on how good your niche and site is. It all varies some sites can earn nothing and others can earn hundreds a month. ----------------- Internet Marketing and making money online is not a sci
- rank 3 | msmarco_test_000009_neg_5 | label=0 | score=0.0154 | If this is your first day of affiliate marketing and you earn $1000, don’t assume that you’ll bank $365,000 in the next year. Your current earnings, in relation to a salary, are $2.73/day.
- rank 4 | msmarco_test_000009_neg_1 | label=0 | score=0.0146 | E-commerce marketing managers who focus on online purchase behaviors earn an average of $76,500 to $103,250 per year. Managers in interactive marketing, where sellers try to anticipate and address the
- rank 5 | msmarco_test_000009_neg_2 | label=0 | score=0.0000 | Marketing managers earned a median annual wage of $123,220 in 2013, according to the BLS. The best-paid 10 percent in the field made more than $187,199 per year, while the bottom 10 percent made less 

**BM25** positive rank: 8

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is limited (earn, marketers), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000009_neg_3 | label=0 | score=2.0749 | Most chief marketers earn between $100,000 and $350,000 per year, according to a recent report from The CMO Council.
- rank 2 | msmarco_test_000009_neg_7 | label=0 | score=1.6073 | It depends on how good your niche and site is. It all varies some sites can earn nothing and others can earn hundreds a month. ----------------- Internet Marketing and making money online is not a sci
- rank 3 | msmarco_test_000009_neg_1 | label=0 | score=0.0000 | E-commerce marketing managers who focus on online purchase behaviors earn an average of $76,500 to $103,250 per year. Managers in interactive marketing, where sellers try to anticipate and address the
- rank 4 | msmarco_test_000009_neg_2 | label=0 | score=0.0000 | Marketing managers earned a median annual wage of $123,220 in 2013, according to the BLS. The best-paid 10 percent in the field made more than $187,199 per year, while the bottom 10 percent made less 
- rank 5 | msmarco_test_000009_neg_4 | label=0 | score=0.0000 | The BLS reports that the upper 10 percent of marketing managers earned more than $187,200, as did managers in advertising and promotions. Market research analysts in the top ten percent earned more th

**MLP Jittor** positive rank: 7

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is limited (earn, marketers), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000009_neg_3 | label=0 | score=-1.5333 | Most chief marketers earn between $100,000 and $350,000 per year, according to a recent report from The CMO Council.
- rank 2 | msmarco_test_000009_neg_2 | label=0 | score=-1.8626 | Marketing managers earned a median annual wage of $123,220 in 2013, according to the BLS. The best-paid 10 percent in the field made more than $187,199 per year, while the bottom 10 percent made less 
- rank 3 | msmarco_test_000009_neg_5 | label=0 | score=-4.7415 | If this is your first day of affiliate marketing and you earn $1000, don’t assume that you’ll bank $365,000 in the next year. Your current earnings, in relation to a salary, are $2.73/day.
- rank 4 | msmarco_test_000009_neg_6 | label=0 | score=-5.2301 | Marketing managers in the New York metropolitan area, for example, averaged $167,590 per year in May 2011, the BLS reports. Those working in the San Francisco metro area averaged $169,520 per year. Th
- rank 5 | msmarco_test_000009_neg_4 | label=0 | score=-5.7785 | The BLS reports that the upper 10 percent of marketing managers earned more than $187,200, as did managers in advertising and promotions. Market research analysts in the top ten percent earned more th

**TextCNN Jittor** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is very low, which makes the case difficult for lightweight lexical and neural rankers.

- rank 1 | msmarco_test_000009_neg_6 | label=0 | score=6.5857 | Marketing managers in the New York metropolitan area, for example, averaged $167,590 per year in May 2011, the BLS reports. Those working in the San Francisco metro area averaged $169,520 per year. Th
- rank 2 | msmarco_test_000009_pos_8 | label=1 | score=4.5716 | Salary. The Bureau of Labor Statistics reports that marketing managers in the United States earned an average annual wage of $126,190 as of May 2011. The median wage, or midway point between the lowes
- rank 3 | msmarco_test_000009_neg_3 | label=0 | score=4.4393 | Most chief marketers earn between $100,000 and $350,000 per year, according to a recent report from The CMO Council.
- rank 4 | msmarco_test_000009_neg_1 | label=0 | score=0.8141 | E-commerce marketing managers who focus on online purchase behaviors earn an average of $76,500 to $103,250 per year. Managers in interactive marketing, where sellers try to anticipate and address the
- rank 5 | msmarco_test_000009_neg_7 | label=0 | score=0.7743 | It depends on how good your niche and site is. It all varies some sites can earn nothing and others can earn hundreds a month. ----------------- Internet Marketing and making money online is not a sci

## Failure Cases

### msmarco_test_000001

Query: how long does it take to drive from st louis to detroit

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (detroit, drive, from, how, long, louis), which can help lexical baselines.

- rank 1 | msmarco_test_000001_pos_8 | label=1 | score=0.2088 | How long is the drive from Detroit, MI to Saint Louis, MO? The total driving time is 8 hours, 43 minutes.
- rank 2 | msmarco_test_000001_neg_2 | label=0 | score=0.1332 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa
- rank 3 | msmarco_test_000001_neg_6 | label=0 | score=0.1319 | You can also calculate the cost to drive from Saint Louis, MO to Detroit, MI based on current local gas prices and an estimate of your car's best gas mileage.
- rank 4 | msmarco_test_000001_neg_3 | label=0 | score=0.0902 | Your trip begins in Saint Louis, Missouri. It ends in Detroit, Michigan. If you're planning a road trip, you might be interested in seeing the total driving distance from Saint Louis, MO to Detroit, M
- rank 5 | msmarco_test_000001_neg_7 | label=0 | score=0.0684 | 552 mi (888km) - about 8 hours 39 mins Saint Louis, MO, USA to Detroit, MI, USA.

**BM25** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (detroit, drive, from, how, long, louis), which can help lexical baselines.

- rank 1 | msmarco_test_000001_pos_8 | label=1 | score=6.2511 | How long is the drive from Detroit, MI to Saint Louis, MO? The total driving time is 8 hours, 43 minutes.
- rank 2 | msmarco_test_000001_neg_2 | label=0 | score=3.8144 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa
- rank 3 | msmarco_test_000001_neg_3 | label=0 | score=3.1887 | Your trip begins in Saint Louis, Missouri. It ends in Detroit, Michigan. If you're planning a road trip, you might be interested in seeing the total driving distance from Saint Louis, MO to Detroit, M
- rank 4 | msmarco_test_000001_neg_1 | label=0 | score=2.9807 | Since this is a long drive, you might want to stop halfway and stay overnight in a hotel. You can find the city that is halfway between Saint Louis, MO and Detroit, MI.
- rank 5 | msmarco_test_000001_neg_4 | label=0 | score=2.5266 | Merge onto I-71 south and stay on it for approximately 1 hour and 10 minutes, then take Exit 1 to I-64 west. Continue on I-64 west for approximately 1 hour then take Exit 63. Turn left (south) onto Hi

**MLP Jittor** positive rank: 9

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (drive, from, louis, st, take, to), which can help lexical baselines.

- rank 1 | msmarco_test_000001_neg_2 | label=0 | score=-3.5292 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa
- rank 2 | msmarco_test_000001_neg_9 | label=0 | score=-5.6620 | The distance between Missouri, USA, and Detroit, MI, is 660 miles and will take about 10 Hours 20 Minutes of driving time.
- rank 3 | msmarco_test_000001_neg_6 | label=0 | score=-6.4313 | You can also calculate the cost to drive from Saint Louis, MO to Detroit, MI based on current local gas prices and an estimate of your car's best gas mileage.
- rank 4 | msmarco_test_000001_neg_4 | label=0 | score=-6.6795 | Merge onto I-71 south and stay on it for approximately 1 hour and 10 minutes, then take Exit 1 to I-64 west. Continue on I-64 west for approximately 1 hour then take Exit 63. Turn left (south) onto Hi
- rank 5 | msmarco_test_000001_neg_5 | label=0 | score=-6.6961 | – Take I-75 south to southwestern Ohio, then I-71 south into Louisville, KY. Take I-64 west into Indiana. Stay on I-64 west for nearly an hour, then take Exit 63. Turn left (south) onto Highway 162 an

**TextCNN Jittor** positive rank: 7

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (drive, from, louis, st, take, to), which can help lexical baselines.

- rank 1 | msmarco_test_000001_neg_2 | label=0 | score=7.2690 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa
- rank 2 | msmarco_test_000001_neg_3 | label=0 | score=6.6562 | Your trip begins in Saint Louis, Missouri. It ends in Detroit, Michigan. If you're planning a road trip, you might be interested in seeing the total driving distance from Saint Louis, MO to Detroit, M
- rank 3 | msmarco_test_000001_neg_9 | label=0 | score=5.0788 | The distance between Missouri, USA, and Detroit, MI, is 660 miles and will take about 10 Hours 20 Minutes of driving time.
- rank 4 | msmarco_test_000001_neg_5 | label=0 | score=4.7150 | – Take I-75 south to southwestern Ohio, then I-71 south into Louisville, KY. Take I-64 west into Indiana. Stay on I-64 west for nearly an hour, then take Exit 63. Turn left (south) onto Highway 162 an
- rank 5 | msmarco_test_000001_neg_10 | label=0 | score=3.7280 | The driving distance between Detroit, MI and Nashville, TN is approximately 535 miles. The driving time would be approximately 8 hours 30 minutes if you were to travel non-sto … p in good driving cond

### msmarco_test_000002

Query: what is cetirizine hydrochloride

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (cetirizine, hydrochloride, is), which can help lexical baselines.

- rank 1 | msmarco_test_000002_pos_1 | label=1 | score=0.1529 | Cetirizine hydrochloride is the generic name for a prescription and over-the-counter antihistamine medication. Doctors recommend cetirizine hydrochloride for the relief of symptoms associated with all
- rank 2 | msmarco_test_000002_neg_7 | label=0 | score=0.0910 | Prior to taking cetirizine, it is a good idea to review the drug's safety information. For example, side effects such as fatigue, drowsiness, and dry mouth may occur. In addition, this medicine is not
- rank 3 | msmarco_test_000002_neg_2 | label=0 | score=0.0905 | Function. Cetirizine hydrochloride works by blocking receptors in your body that are sensitive to histamine, a chemical released by your immune system in the presence of something that it mistakenly p
- rank 4 | msmarco_test_000002_neg_6 | label=0 | score=0.0587 | Cetirizine /sɛˈtɪrɨziːn/ (trade names Zirtec, Zyrtec, Reactine) is a second-generation antihistamine used in the treatment of hay fever, allergies, angioedema, and urticaria. It is a major metabolite 
- rank 5 | msmarco_test_000002_neg_5 | label=0 | score=0.0482 | Allergies [edit]. Cetirizine's primary indication is for hay fever and other allergies. Because the symptoms of itching and redness in these conditions are caused by histamine acting on the H1 recepto

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (cetirizine, hydrochloride, is, what), which can help lexical baselines.

- rank 1 | msmarco_test_000002_neg_7 | label=0 | score=2.5947 | Prior to taking cetirizine, it is a good idea to review the drug's safety information. For example, side effects such as fatigue, drowsiness, and dry mouth may occur. In addition, this medicine is not
- rank 2 | msmarco_test_000002_pos_1 | label=1 | score=1.4117 | Cetirizine hydrochloride is the generic name for a prescription and over-the-counter antihistamine medication. Doctors recommend cetirizine hydrochloride for the relief of symptoms associated with all
- rank 3 | msmarco_test_000002_neg_2 | label=0 | score=1.1662 | Function. Cetirizine hydrochloride works by blocking receptors in your body that are sensitive to histamine, a chemical released by your immune system in the presence of something that it mistakenly p
- rank 4 | msmarco_test_000002_neg_6 | label=0 | score=1.1250 | Cetirizine /sɛˈtɪrɨziːn/ (trade names Zirtec, Zyrtec, Reactine) is a second-generation antihistamine used in the treatment of hay fever, allergies, angioedema, and urticaria. It is a major metabolite 
- rank 5 | msmarco_test_000002_neg_5 | label=0 | score=1.0696 | Allergies [edit]. Cetirizine's primary indication is for hay fever and other allergies. Because the symptoms of itching and redness in these conditions are caused by histamine acting on the H1 recepto

**MLP Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (cetirizine, hydrochloride, is), which can help lexical baselines.

- rank 1 | msmarco_test_000002_pos_1 | label=1 | score=2.3521 | Cetirizine hydrochloride is the generic name for a prescription and over-the-counter antihistamine medication. Doctors recommend cetirizine hydrochloride for the relief of symptoms associated with all
- rank 2 | msmarco_test_000002_neg_2 | label=0 | score=-0.7210 | Function. Cetirizine hydrochloride works by blocking receptors in your body that are sensitive to histamine, a chemical released by your immune system in the presence of something that it mistakenly p
- rank 3 | msmarco_test_000002_neg_8 | label=0 | score=-1.2463 | Zyrtec-D (cetirizine and pseudoephedrine) is a combination of an antihistamine and a decongestant used to treat cold or allergy symptoms such as nasal and sinus congestion, sneezing, itching, watery e
- rank 4 | msmarco_test_000002_neg_6 | label=0 | score=-1.6227 | Cetirizine /sɛˈtɪrɨziːn/ (trade names Zirtec, Zyrtec, Reactine) is a second-generation antihistamine used in the treatment of hay fever, allergies, angioedema, and urticaria. It is a major metabolite 
- rank 5 | msmarco_test_000002_neg_5 | label=0 | score=-2.6034 | Allergies [edit]. Cetirizine's primary indication is for hay fever and other allergies. Because the symptoms of itching and redness in these conditions are caused by histamine acting on the H1 recepto

**TextCNN Jittor** positive rank: 8

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (cetirizine, hydrochloride, is), which can help lexical baselines.

- rank 1 | msmarco_test_000002_neg_2 | label=0 | score=10.8324 | Function. Cetirizine hydrochloride works by blocking receptors in your body that are sensitive to histamine, a chemical released by your immune system in the presence of something that it mistakenly p
- rank 2 | msmarco_test_000002_neg_5 | label=0 | score=10.4196 | Allergies [edit]. Cetirizine's primary indication is for hay fever and other allergies. Because the symptoms of itching and redness in these conditions are caused by histamine acting on the H1 recepto
- rank 3 | msmarco_test_000002_neg_8 | label=0 | score=7.5559 | Zyrtec-D (cetirizine and pseudoephedrine) is a combination of an antihistamine and a decongestant used to treat cold or allergy symptoms such as nasal and sinus congestion, sneezing, itching, watery e
- rank 4 | msmarco_test_000002_neg_3 | label=0 | score=5.9701 | Last reviewed on RxList 4/2/2015. Zyrtec (cetirizine hydrochloride) is an antihistamine that treats symptoms, such as itching, runny nose, watery eyes, and sneezing from hay fever (allergic rhinitis) 
- rank 5 | msmarco_test_000002_neg_4 | label=0 | score=5.5558 | Uses. Cetirizine is an antihistamine used to relieve allergy symptoms such as watery eyes, runny nose, itching eyes /nose, sneezing, hives, and itching. It works by blocking a certain natural substanc

### msmarco_test_000003

Query: how much does it cost to join the masters golf club

**TF-IDF** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (club, golf, how, it, join, much), which can help lexical baselines.

- rank 1 | msmarco_test_000003_neg_8 | label=0 | score=0.1541 | The Bridge in Bridgehampton, N.Y., charged a whopping $750,000 to join, according to a 2007 Wall Street Journal article. Another Hamptons club, Sebonack Golf Club, billed new members for $650,000 when
- rank 2 | msmarco_test_000003_pos_1 | label=1 | score=0.1187 | Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiation fee is between $10,000 and $30,000 with yearly dues costing between $3,000 and $10,000. Dining and
- rank 3 | msmarco_test_000003_neg_2 | label=0 | score=0.1165 | You cannot play Augusta as a walk-on; you mu … st be invited as a guest of one of the club's members. Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiat
- rank 4 | msmarco_test_000003_neg_3 | label=0 | score=0.1062 | 1 According to an article in BusinessInsider, membership fees to the Augusta National Golf Club are about $10,000 a year. 2  A Huffington Post article states that the Augusta National Golf Club initia
- rank 5 | msmarco_test_000003_neg_4 | label=0 | score=0.0989 | 1 The cost to use a room at the Augusta National club is $100 a night. 2  This could be used for weddings, conferences, or banquets of any kind. 3  A bottle of wine at the club can run $1,000 each. 4 

**BM25** positive rank: 3

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (club, golf, how, it, join, much), which can help lexical baselines.

- rank 1 | msmarco_test_000003_neg_8 | label=0 | score=5.7508 | The Bridge in Bridgehampton, N.Y., charged a whopping $750,000 to join, according to a 2007 Wall Street Journal article. Another Hamptons club, Sebonack Golf Club, billed new members for $650,000 when
- rank 2 | msmarco_test_000003_neg_2 | label=0 | score=4.2830 | You cannot play Augusta as a walk-on; you mu … st be invited as a guest of one of the club's members. Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiat
- rank 3 | msmarco_test_000003_pos_1 | label=1 | score=3.4220 | Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiation fee is between $10,000 and $30,000 with yearly dues costing between $3,000 and $10,000. Dining and
- rank 4 | msmarco_test_000003_neg_3 | label=0 | score=2.4876 | 1 According to an article in BusinessInsider, membership fees to the Augusta National Golf Club are about $10,000 a year. 2  A Huffington Post article states that the Augusta National Golf Club initia
- rank 5 | msmarco_test_000003_neg_4 | label=0 | score=2.3950 | 1 The cost to use a room at the Augusta National club is $100 a night. 2  This could be used for weddings, conferences, or banquets of any kind. 3  A bottle of wine at the club can run $1,000 each. 4 

**MLP Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (club, cost, does, golf, it, masters), which can help lexical baselines.

- rank 1 | msmarco_test_000003_pos_1 | label=1 | score=-0.5556 | Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiation fee is between $10,000 and $30,000 with yearly dues costing between $3,000 and $10,000. Dining and
- rank 2 | msmarco_test_000003_neg_2 | label=0 | score=-1.4455 | You cannot play Augusta as a walk-on; you mu … st be invited as a guest of one of the club's members. Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiat
- rank 3 | msmarco_test_000003_neg_7 | label=0 | score=-2.2285 | In 2009, Golf World magazine published an article titled  Inside Augusta National Golf Club .. A club member (speaking anonymously) told the magazine that the initiation fee is in the low five-figures
- rank 4 | msmarco_test_000003_neg_3 | label=0 | score=-2.8651 | 1 According to an article in BusinessInsider, membership fees to the Augusta National Golf Club are about $10,000 a year. 2  A Huffington Post article states that the Augusta National Golf Club initia
- rank 5 | msmarco_test_000003_neg_4 | label=0 | score=-3.8412 | 1 The cost to use a room at the Augusta National club is $100 a night. 2  This could be used for weddings, conferences, or banquets of any kind. 3  A bottle of wine at the club can run $1,000 each. 4 

**TextCNN Jittor** positive rank: 5

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (club, golf, the, to), which can help lexical baselines.

- rank 1 | msmarco_test_000003_neg_5 | label=0 | score=11.9461 | Augusta National Golf Club, located in Augusta, Georgia, is one of the most famous golf clubs in the world. Founded by Bobby Jones and Clifford Roberts on the site of the former Fruitland (later Fruit
- rank 2 | msmarco_test_000003_neg_3 | label=0 | score=11.8187 | 1 According to an article in BusinessInsider, membership fees to the Augusta National Golf Club are about $10,000 a year. 2  A Huffington Post article states that the Augusta National Golf Club initia
- rank 3 | msmarco_test_000003_neg_6 | label=0 | score=10.3245 | Augusta National Golf Club has about 300 members at any given time. Membership is strictly by invitation; there is no application process. In 2004, USA Today published a list of all the current member
- rank 4 | msmarco_test_000003_neg_7 | label=0 | score=9.0193 | In 2009, Golf World magazine published an article titled  Inside Augusta National Golf Club .. A club member (speaking anonymously) told the magazine that the initiation fee is in the low five-figures
- rank 5 | msmarco_test_000003_pos_1 | label=1 | score=8.4989 | Augusta National Golf Club has never disclosed the membership cost, but it is believed the initiation fee is between $10,000 and $30,000 with yearly dues costing between $3,000 and $10,000. Dining and
