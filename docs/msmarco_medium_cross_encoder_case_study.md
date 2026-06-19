# MS MARCO Medium Cross-Encoder Case Study

This qualitative analysis compares BM25, lightweight Jittor rerankers, and an external pretrained Cross-Encoder. The Cross-Encoder is not the Jittor reproduction body.

The pretrained cross-encoder may better capture semantic relevance, while BM25 is strong when lexical overlap is reliable. Lightweight Jittor models still lack pretrained semantic knowledge.

## Cross-Encoder succeeds while BM25 fails

### msmarco_test_000015

Query: how to get rid of serve burn acne

**BM25** positive rank: 6
- rank 1 | label=0 | score=2.4085 | msmarco_test_000015_neg_3 | Scars are the unwanted marks left on the skin after a burn, acne, or surgery. They are a mark of the natural healing of any wound. No one likes distracting spots or signs on the skin. Effective skin treatment including n
- rank 2 | label=0 | score=2.1041 | msmarco_test_000015_neg_1 | Drink 3-4 liters of water daily to get rid of acne scars naturally fast. Water helps in ex-foliating the skin and removes all the skin impurities. Additional Tips to get rid of acne scars: 1.) Use single treatment at a t
- rank 3 | label=0 | score=1.8961 | msmarco_test_000015_neg_6 | 1.) Get rid of acne scars overnight naturally fast using Baking Soda: Baking soda can be used to get rid of acne scars naturally fast. Mix one teaspoon of baking soda with 3 teaspoons of water to make a paste. Apply this
- rank 4 | label=0 | score=1.5634 | msmarco_test_000015_neg_5 | A lot of the time, people are so desperate to get rid of acne scars and skin discolorations that they will use all manner of abrasive products and methods which can irritate the skin and make the situation worse.
- rank 5 | label=0 | score=1.4675 | msmarco_test_000015_neg_7 | Apply the mixture on the face. Wash off with water after 15 minutes. Lemon juice acts as a natural skin bleach. Lemon juice helps in fading of scars, which acts as an effective way to get rid of acne scars naturally fast

**MLP Jittor** positive rank: 2
- rank 1 | label=0 | score=-4.6463 | msmarco_test_000015_neg_1 | Drink 3-4 liters of water daily to get rid of acne scars naturally fast. Water helps in ex-foliating the skin and removes all the skin impurities. Additional Tips to get rid of acne scars: 1.) Use single treatment at a t
- rank 2 | label=1 | score=-4.6509 | msmarco_test_000015_pos_2 | 1 Rose water and sandalwood: Make a paste of rose water and sandalwood and gently apply it on your acne scars. 2  Leave the paste on your skin overnight then wash it with cold water the next morning. 3  Do this regularly
- rank 3 | label=0 | score=-5.2441 | msmarco_test_000015_neg_3 | Scars are the unwanted marks left on the skin after a burn, acne, or surgery. They are a mark of the natural healing of any wound. No one likes distracting spots or signs on the skin. Effective skin treatment including n
- rank 4 | label=0 | score=-6.1824 | msmarco_test_000015_neg_5 | A lot of the time, people are so desperate to get rid of acne scars and skin discolorations that they will use all manner of abrasive products and methods which can irritate the skin and make the situation worse.
- rank 5 | label=0 | score=-6.3930 | msmarco_test_000015_neg_7 | Apply the mixture on the face. Wash off with water after 15 minutes. Lemon juice acts as a natural skin bleach. Lemon juice helps in fading of scars, which acts as an effective way to get rid of acne scars naturally fast

**TextCNN Jittor** positive rank: 4
- rank 1 | label=0 | score=4.8540 | msmarco_test_000015_neg_6 | 1.) Get rid of acne scars overnight naturally fast using Baking Soda: Baking soda can be used to get rid of acne scars naturally fast. Mix one teaspoon of baking soda with 3 teaspoons of water to make a paste. Apply this
- rank 2 | label=0 | score=4.2430 | msmarco_test_000015_neg_7 | Apply the mixture on the face. Wash off with water after 15 minutes. Lemon juice acts as a natural skin bleach. Lemon juice helps in fading of scars, which acts as an effective way to get rid of acne scars naturally fast
- rank 3 | label=0 | score=3.8130 | msmarco_test_000015_neg_5 | A lot of the time, people are so desperate to get rid of acne scars and skin discolorations that they will use all manner of abrasive products and methods which can irritate the skin and make the situation worse.
- rank 4 | label=1 | score=2.2782 | msmarco_test_000015_pos_2 | 1 Rose water and sandalwood: Make a paste of rose water and sandalwood and gently apply it on your acne scars. 2  Leave the paste on your skin overnight then wash it with cold water the next morning. 3  Do this regularly
- rank 5 | label=0 | score=1.5243 | msmarco_test_000015_neg_3 | Scars are the unwanted marks left on the skin after a burn, acne, or surgery. They are a mark of the natural healing of any wound. No one likes distracting spots or signs on the skin. Effective skin treatment including n

**Cross-Encoder** positive rank: 1
- rank 1 | label=1 | score=5.3355 | msmarco_test_000015_pos_2 | 1 Rose water and sandalwood: Make a paste of rose water and sandalwood and gently apply it on your acne scars. 2  Leave the paste on your skin overnight then wash it with cold water the next morning. 3  Do this regularly
- rank 2 | label=0 | score=5.1569 | msmarco_test_000015_neg_6 | 1.) Get rid of acne scars overnight naturally fast using Baking Soda: Baking soda can be used to get rid of acne scars naturally fast. Mix one teaspoon of baking soda with 3 teaspoons of water to make a paste. Apply this
- rank 3 | label=0 | score=4.2634 | msmarco_test_000015_neg_1 | Drink 3-4 liters of water daily to get rid of acne scars naturally fast. Water helps in ex-foliating the skin and removes all the skin impurities. Additional Tips to get rid of acne scars: 1.) Use single treatment at a t
- rank 4 | label=0 | score=3.9436 | msmarco_test_000015_neg_7 | Apply the mixture on the face. Wash off with water after 15 minutes. Lemon juice acts as a natural skin bleach. Lemon juice helps in fading of scars, which acts as an effective way to get rid of acne scars naturally fast
- rank 5 | label=0 | score=3.3962 | msmarco_test_000015_neg_4 | 1 Apply Aloe vera gel or juice: Aloe vera gel or juice helps get rid of acne scars and makes your facial skin healthier. 2  There are many brands of aloe vera gel or juice available on the market today.

### msmarco_test_000031

Query: what are the photosynthetic protists

**BM25** positive rank: 5
- rank 1 | label=0 | score=2.2784 | msmarco_test_000031_neg_6 | Almost all algae live in water. There are several kinds of chlorophyll. The most important chlorophyll is a. This certain type of chlorophyll is what makes photosynthesis possible, all the plants and algae that use photo
- rank 2 | label=0 | score=1.3430 | msmarco_test_000031_neg_2 | Algae: Protists with Chloroplasts. The algae are a polyphyletic and paraphyletic group of organisms. They are defined in differing ways, but are usually considered to be the photosynthetic organisms excepting plants.
- rank 3 | label=0 | score=1.3344 | msmarco_test_000031_neg_1 | They are also known as protistan algae or plantlike protists. They are represented by mainly the unicellular algae. These organisms are mostly planktonic and represent the phytoplanktons which account for nearly 80% of t
- rank 4 | label=0 | score=1.2840 | msmarco_test_000031_neg_4 | Autotrophic Protists: Algae The term algae embraces all photosynthetic protists. It refers to an aquatic, photosynthetic way of life, not an evolutionary kinship. Most algae live in water, but some are terrestrial. Most 
- rank 5 | label=1 | score=1.2833 | msmarco_test_000031_pos_7 | Photosynthetic Protists. These protists belong to a division called Pyrrophyta of algae. They are a well defined group of unicellular, photosynthetic forms. Most of them are flagellated and motile but some forms are non-

**MLP Jittor** positive rank: 1
- rank 1 | label=1 | score=-2.8977 | msmarco_test_000031_pos_7 | Photosynthetic Protists. These protists belong to a division called Pyrrophyta of algae. They are a well defined group of unicellular, photosynthetic forms. Most of them are flagellated and motile but some forms are non-
- rank 2 | label=0 | score=-4.1223 | msmarco_test_000031_neg_3 | These protists belong to a division called Pyrrophyta of algae. They are a well defined group of unicellular, photosynthetic forms. Most of them are flagellated and motile but some forms are non-flagellated. Some dinofla
- rank 3 | label=0 | score=-4.7168 | msmarco_test_000031_neg_1 | They are also known as protistan algae or plantlike protists. They are represented by mainly the unicellular algae. These organisms are mostly planktonic and represent the phytoplanktons which account for nearly 80% of t
- rank 4 | label=0 | score=-5.9759 | msmarco_test_000031_neg_2 | Algae: Protists with Chloroplasts. The algae are a polyphyletic and paraphyletic group of organisms. They are defined in differing ways, but are usually considered to be the photosynthetic organisms excepting plants.
- rank 5 | label=0 | score=-6.4880 | msmarco_test_000031_neg_5 | Photosynthetic Protists Pyrrhophyta: TheDinoflagellates The dinoflagellates consist of about2100 known species of primarily uni-cellular, photosynthetic organisms,most of which have two flagella. Amajority of the dinofla

**TextCNN Jittor** positive rank: 6
- rank 1 | label=0 | score=5.8616 | msmarco_test_000031_neg_4 | Autotrophic Protists: Algae The term algae embraces all photosynthetic protists. It refers to an aquatic, photosynthetic way of life, not an evolutionary kinship. Most algae live in water, but some are terrestrial. Most 
- rank 2 | label=0 | score=5.8354 | msmarco_test_000031_neg_2 | Algae: Protists with Chloroplasts. The algae are a polyphyletic and paraphyletic group of organisms. They are defined in differing ways, but are usually considered to be the photosynthetic organisms excepting plants.
- rank 3 | label=0 | score=4.6240 | msmarco_test_000031_neg_1 | They are also known as protistan algae or plantlike protists. They are represented by mainly the unicellular algae. These organisms are mostly planktonic and represent the phytoplanktons which account for nearly 80% of t
- rank 4 | label=0 | score=1.5941 | msmarco_test_000031_neg_6 | Almost all algae live in water. There are several kinds of chlorophyll. The most important chlorophyll is a. This certain type of chlorophyll is what makes photosynthesis possible, all the plants and algae that use photo
- rank 5 | label=0 | score=1.1965 | msmarco_test_000031_neg_5 | Photosynthetic Protists Pyrrhophyta: TheDinoflagellates The dinoflagellates consist of about2100 known species of primarily uni-cellular, photosynthetic organisms,most of which have two flagella. Amajority of the dinofla

**Cross-Encoder** positive rank: 1
- rank 1 | label=1 | score=9.0522 | msmarco_test_000031_pos_7 | Photosynthetic Protists. These protists belong to a division called Pyrrophyta of algae. They are a well defined group of unicellular, photosynthetic forms. Most of them are flagellated and motile but some forms are non-
- rank 2 | label=0 | score=6.7208 | msmarco_test_000031_neg_5 | Photosynthetic Protists Pyrrhophyta: TheDinoflagellates The dinoflagellates consist of about2100 known species of primarily uni-cellular, photosynthetic organisms,most of which have two flagella. Amajority of the dinofla
- rank 3 | label=0 | score=5.3950 | msmarco_test_000031_neg_3 | These protists belong to a division called Pyrrophyta of algae. They are a well defined group of unicellular, photosynthetic forms. Most of them are flagellated and motile but some forms are non-flagellated. Some dinofla
- rank 4 | label=0 | score=4.7882 | msmarco_test_000031_neg_4 | Autotrophic Protists: Algae The term algae embraces all photosynthetic protists. It refers to an aquatic, photosynthetic way of life, not an evolutionary kinship. Most algae live in water, but some are terrestrial. Most 
- rank 5 | label=0 | score=4.0140 | msmarco_test_000031_neg_1 | They are also known as protistan algae or plantlike protists. They are represented by mainly the unicellular algae. These organisms are mostly planktonic and represent the phytoplanktons which account for nearly 80% of t

### msmarco_test_000038

Query: difference in salary rn and bsn

**BM25** positive rank: 7
- rank 1 | label=0 | score=2.4307 | msmarco_test_000038_neg_4 | For example, a hemodialysis nurse with a BSN degree earns $6,000 more than an RN with a ASN degree who works in the same area. RN and BSN Demands and Salary. With all else equal, RNs with BSN degrees receive higher salar
- rank 2 | label=0 | score=2.4144 | msmarco_test_000038_neg_5 | The higher level positions have a higher average salary. For that reason, while there is no significant difference in the salary of an ADN RN and a BSN RN working as a staff nurse on a hospital unit, a nurse with a BSN w
- rank 3 | label=0 | score=2.4136 | msmarco_test_000038_neg_1 | After promotions, experienced nurses may earn as much as $67,449 on average yearly. RN versus BSN Salary Based on Degree. Typically, BSN graduates who pass the NCLEX-RN earn more than RNs who obtain the ADN or ASN. Altho
- rank 4 | label=0 | score=1.9381 | msmarco_test_000038_neg_3 | In terms of salary for RN positions, BSN graduates in the same position tend to be paid more. A 2013 article by Rasmussen College reported that, in a review of 187,423 nursing job postings made over three months, positio
- rank 5 | label=0 | score=1.7827 | msmarco_test_000038_neg_6 | Further analysis of those job postings can show the mean real-time salary for the positions posted. For the positions requiring a post-secondary or associate degree the mean salary is $66,620 and for a BSN degree it’s $7

**MLP Jittor** positive rank: 4
- rank 1 | label=0 | score=-2.5497 | msmarco_test_000038_neg_8 | High school – 6%. These stats show that an RN is eligible for 51% of these jobs, while a BSN holder is going to be eligible for 88% of the jobs. Also, the analysis in this study of salary showed that the mean salary for 
- rank 2 | label=0 | score=-3.5632 | msmarco_test_000038_neg_1 | After promotions, experienced nurses may earn as much as $67,449 on average yearly. RN versus BSN Salary Based on Degree. Typically, BSN graduates who pass the NCLEX-RN earn more than RNs who obtain the ADN or ASN. Altho
- rank 3 | label=0 | score=-4.7503 | msmarco_test_000038_neg_5 | The higher level positions have a higher average salary. For that reason, while there is no significant difference in the salary of an ADN RN and a BSN RN working as a staff nurse on a hospital unit, a nurse with a BSN w
- rank 4 | label=1 | score=-4.9873 | msmarco_test_000038_pos_7 | Payscale.com finds that there are large differences in salary for people with only an RN, compared to a BSN. 2014 data shows that an RN earns a median of $39,100, while a BSN holder earns $69,000: Education Requirements.
- rank 5 | label=0 | score=-5.5439 | msmarco_test_000038_neg_4 | For example, a hemodialysis nurse with a BSN degree earns $6,000 more than an RN with a ASN degree who works in the same area. RN and BSN Demands and Salary. With all else equal, RNs with BSN degrees receive higher salar

**TextCNN Jittor** positive rank: 6
- rank 1 | label=0 | score=9.7055 | msmarco_test_000038_neg_6 | Further analysis of those job postings can show the mean real-time salary for the positions posted. For the positions requiring a post-secondary or associate degree the mean salary is $66,620 and for a BSN degree it’s $7
- rank 2 | label=0 | score=9.3193 | msmarco_test_000038_neg_4 | For example, a hemodialysis nurse with a BSN degree earns $6,000 more than an RN with a ASN degree who works in the same area. RN and BSN Demands and Salary. With all else equal, RNs with BSN degrees receive higher salar
- rank 3 | label=0 | score=8.7753 | msmarco_test_000038_neg_1 | After promotions, experienced nurses may earn as much as $67,449 on average yearly. RN versus BSN Salary Based on Degree. Typically, BSN graduates who pass the NCLEX-RN earn more than RNs who obtain the ADN or ASN. Altho
- rank 4 | label=0 | score=8.0333 | msmarco_test_000038_neg_2 | A BSN degree augments a nurse’s wages faster than an ADN education would as experience builds up. The 2010 U.S. Department of Health and Human Services report on the registered nurse population documents that BSN nurses 
- rank 5 | label=0 | score=7.9986 | msmarco_test_000038_neg_8 | High school – 6%. These stats show that an RN is eligible for 51% of these jobs, while a BSN holder is going to be eligible for 88% of the jobs. Also, the analysis in this study of salary showed that the mean salary for 

**Cross-Encoder** positive rank: 1
- rank 1 | label=1 | score=9.1491 | msmarco_test_000038_pos_7 | Payscale.com finds that there are large differences in salary for people with only an RN, compared to a BSN. 2014 data shows that an RN earns a median of $39,100, while a BSN holder earns $69,000: Education Requirements.
- rank 2 | label=0 | score=8.9650 | msmarco_test_000038_neg_1 | After promotions, experienced nurses may earn as much as $67,449 on average yearly. RN versus BSN Salary Based on Degree. Typically, BSN graduates who pass the NCLEX-RN earn more than RNs who obtain the ADN or ASN. Altho
- rank 3 | label=0 | score=8.7134 | msmarco_test_000038_neg_5 | The higher level positions have a higher average salary. For that reason, while there is no significant difference in the salary of an ADN RN and a BSN RN working as a staff nurse on a hospital unit, a nurse with a BSN w
- rank 4 | label=0 | score=8.6440 | msmarco_test_000038_neg_4 | For example, a hemodialysis nurse with a BSN degree earns $6,000 more than an RN with a ASN degree who works in the same area. RN and BSN Demands and Salary. With all else equal, RNs with BSN degrees receive higher salar
- rank 5 | label=0 | score=8.0417 | msmarco_test_000038_neg_3 | In terms of salary for RN positions, BSN graduates in the same position tend to be paid more. A 2013 article by Rasmussen College reported that, in a review of 187,423 nursing job postings made over three months, positio

## BM25 succeeds while Cross-Encoder fails

### msmarco_test_000004

Query: which blood test is for folate

**BM25** positive rank: 1
- rank 1 | label=1 | score=3.6124 | msmarco_test_000004_pos_5 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. The procedure consists
- rank 2 | label=0 | score=2.2926 | msmarco_test_000004_neg_8 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 that are active and 
- rank 3 | label=0 | score=2.1767 | msmarco_test_000004_neg_3 | This test measures the levels of vitamin B12 and folate in your blood. Your body needs vitamin B12, also called cobalamin, and folate, also called folic acid, to function normally. Both nutrients play important roles in 
- rank 4 | label=0 | score=1.8563 | msmarco_test_000004_neg_7 | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount of folate in the blo
- rank 5 | label=0 | score=1.5951 | msmarco_test_000004_neg_4 | Test Overview. A folic acid test measures the amount of folic acid in the blood. Folic acid is one of many B vitamins. The body needs folic acid to make red blood cells (RBC) , white blood cells (WBC) , and platelets, an

**MLP Jittor** positive rank: 4
- rank 1 | label=0 | score=1.9444 | msmarco_test_000004_neg_3 | This test measures the levels of vitamin B12 and folate in your blood. Your body needs vitamin B12, also called cobalamin, and folate, also called folic acid, to function normally. Both nutrients play important roles in 
- rank 2 | label=0 | score=1.6911 | msmarco_test_000004_neg_2 | B12 and folate levels may be ordered when a complete blood count (CBC) and/or blood smear, done as part of a health checkup or an evaluation for anemia, indicates a low red blood cell (RBC) count with the presence of lar
- rank 3 | label=0 | score=1.2911 | msmarco_test_000004_neg_6 | Vitamin B12 and folate are separate tests often used in conjunction to detect deficiencies and to help diagnose the cause of certain anemias, such as pernicious anemia, an autoimmune disease that affects the absorption o
- rank 4 | label=1 | score=0.2112 | msmarco_test_000004_pos_5 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. The procedure consists
- rank 5 | label=0 | score=0.0951 | msmarco_test_000004_neg_8 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 that are active and 

**TextCNN Jittor** positive rank: 2
- rank 1 | label=0 | score=7.6205 | msmarco_test_000004_neg_7 | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount of folate in the blo
- rank 2 | label=1 | score=6.7732 | msmarco_test_000004_pos_5 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. The procedure consists
- rank 3 | label=0 | score=6.4372 | msmarco_test_000004_neg_6 | Vitamin B12 and folate are separate tests often used in conjunction to detect deficiencies and to help diagnose the cause of certain anemias, such as pernicious anemia, an autoimmune disease that affects the absorption o
- rank 4 | label=0 | score=6.4230 | msmarco_test_000004_neg_2 | B12 and folate levels may be ordered when a complete blood count (CBC) and/or blood smear, done as part of a health checkup or an evaluation for anemia, indicates a low red blood cell (RBC) count with the presence of lar
- rank 5 | label=0 | score=5.2367 | msmarco_test_000004_neg_8 | A particular drawback of testing vitamin B12 levels is that the current widely-used blood test only measures the total amount of vitamin B12 in your blood. This means it measures forms of vitamin B12 that are active and 

**Cross-Encoder** positive rank: 4
- rank 1 | label=0 | score=7.2148 | msmarco_test_000004_neg_7 | The folate (folic acid) test measures the amount of folate in the blood. Folate and folic acid are forms of B9 vitamin. Quantity: * Whole number only. The folate (folic acid) test measures the amount of folate in the blo
- rank 2 | label=0 | score=6.9503 | msmarco_test_000004_neg_2 | B12 and folate levels may be ordered when a complete blood count (CBC) and/or blood smear, done as part of a health checkup or an evaluation for anemia, indicates a low red blood cell (RBC) count with the presence of lar
- rank 3 | label=0 | score=6.3519 | msmarco_test_000004_neg_3 | This test measures the levels of vitamin B12 and folate in your blood. Your body needs vitamin B12, also called cobalamin, and folate, also called folic acid, to function normally. Both nutrients play important roles in 
- rank 4 | label=1 | score=5.4698 | msmarco_test_000004_pos_5 | Overview. Folic acid is the synthetic form of folate or vitamin B9, which is essential for healthy red blood cells. A folic acid test is usually ordered to check for a lack of folate or Vitamin B9. The procedure consists
- rank 5 | label=0 | score=4.9565 | msmarco_test_000004_neg_6 | Vitamin B12 and folate are separate tests often used in conjunction to detect deficiencies and to help diagnose the cause of certain anemias, such as pernicious anemia, an autoimmune disease that affects the absorption o

### msmarco_test_000022

Query: Sedimentary rocks that are formed partially by animals and plants are called what?

**BM25** positive rank: 1
- rank 1 | label=1 | score=4.7764 | msmarco_test_000022_pos_9 | Sedimentation is the collective name for processes that cause mineral and/or organic particles (detritus) to settle and accumulate or minerals to precipitate from a solution. Particles that form a sedimentary rock by acc
- rank 2 | label=0 | score=4.4971 | msmarco_test_000022_neg_4 | Sedimentary Rocks. Sedimentary rocks are formed from broken down bits of other rocks or even from the remains of plants or animals. The little pieces collect in low-lying areas by lakes, oceans, and deserts. They are the
- rank 3 | label=0 | score=4.2096 | msmarco_test_000022_neg_6 | When rocks are weathered and eroded (worn down), they break up into smaller pieces called sediment. Over time, the sediment may be moved by wind and rain to rivers, lakes, and oceans. Other sediments form from the shells
- rank 4 | label=0 | score=3.3562 | msmarco_test_000022_neg_1 | Fossil of shell. Fossils are very common in sedimentary rocks, such as limestone, but rare in metamorphic rocks, such as marble. Fossils are never found in igneous rocks, such as granite. Sediment layers settle over the 
- rank 5 | label=0 | score=3.3008 | msmarco_test_000022_neg_2 | The type of rock that forms where the remains of plants and animals are deposited in thick layers are called organic sedimentary rock. 10 people found this useful. Edit. Share to: Answered.

**MLP Jittor** positive rank: 2
- rank 1 | label=0 | score=0.2320 | msmarco_test_000022_neg_7 | Sedimentary rocks are formed in three ways from these different sized sediments. A sedimentary rock is a layered rock that is formed from the compaction, cementation, and the recrystallization of sediments. Compaction is
- rank 2 | label=1 | score=0.0572 | msmarco_test_000022_pos_9 | Sedimentation is the collective name for processes that cause mineral and/or organic particles (detritus) to settle and accumulate or minerals to precipitate from a solution. Particles that form a sedimentary rock by acc
- rank 3 | label=0 | score=-0.4413 | msmarco_test_000022_neg_6 | When rocks are weathered and eroded (worn down), they break up into smaller pieces called sediment. Over time, the sediment may be moved by wind and rain to rivers, lakes, and oceans. Other sediments form from the shells
- rank 4 | label=0 | score=-1.2685 | msmarco_test_000022_neg_4 | Sedimentary Rocks. Sedimentary rocks are formed from broken down bits of other rocks or even from the remains of plants or animals. The little pieces collect in low-lying areas by lakes, oceans, and deserts. They are the
- rank 5 | label=0 | score=-1.5106 | msmarco_test_000022_neg_5 | Other sediments form from the shells of animals and from plants. Eventually, these sediments get buried and are squashed together to form sedimentary rocks. Sedimentary rocks make up 80–90 per cent of the rocks on the Ea

**TextCNN Jittor** positive rank: 9
- rank 1 | label=0 | score=5.4584 | msmarco_test_000022_neg_3 | SEDIMENTARY ROCKS. Sedimentary rocks are made of particles of sediments such as sand and clay, or the skeletons and shells of sea creatures. When layers of loose sediment are buried and pressed down under more layers, th
- rank 2 | label=0 | score=5.1538 | msmarco_test_000022_neg_6 | When rocks are weathered and eroded (worn down), they break up into smaller pieces called sediment. Over time, the sediment may be moved by wind and rain to rivers, lakes, and oceans. Other sediments form from the shells
- rank 3 | label=0 | score=5.0809 | msmarco_test_000022_neg_1 | Fossil of shell. Fossils are very common in sedimentary rocks, such as limestone, but rare in metamorphic rocks, such as marble. Fossils are never found in igneous rocks, such as granite. Sediment layers settle over the 
- rank 4 | label=0 | score=4.8023 | msmarco_test_000022_neg_5 | Other sediments form from the shells of animals and from plants. Eventually, these sediments get buried and are squashed together to form sedimentary rocks. Sedimentary rocks make up 80–90 per cent of the rocks on the Ea
- rank 5 | label=0 | score=4.4081 | msmarco_test_000022_neg_8 | Sedimentary rocks. Sedimentary rocks are formed from pre-existing rocks or pieces of once-living organisms. They form from deposits that accumulate on the Earth's surface. Sedimentary rocks often have distinctive layerin

**Cross-Encoder** positive rank: 9
- rank 1 | label=0 | score=6.0807 | msmarco_test_000022_neg_2 | The type of rock that forms where the remains of plants and animals are deposited in thick layers are called organic sedimentary rock. 10 people found this useful. Edit. Share to: Answered.
- rank 2 | label=0 | score=5.4082 | msmarco_test_000022_neg_4 | Sedimentary Rocks. Sedimentary rocks are formed from broken down bits of other rocks or even from the remains of plants or animals. The little pieces collect in low-lying areas by lakes, oceans, and deserts. They are the
- rank 3 | label=0 | score=3.9902 | msmarco_test_000022_neg_6 | When rocks are weathered and eroded (worn down), they break up into smaller pieces called sediment. Over time, the sediment may be moved by wind and rain to rivers, lakes, and oceans. Other sediments form from the shells
- rank 4 | label=0 | score=3.0591 | msmarco_test_000022_neg_5 | Other sediments form from the shells of animals and from plants. Eventually, these sediments get buried and are squashed together to form sedimentary rocks. Sedimentary rocks make up 80–90 per cent of the rocks on the Ea
- rank 5 | label=0 | score=2.5118 | msmarco_test_000022_neg_1 | Fossil of shell. Fossils are very common in sedimentary rocks, such as limestone, but rare in metamorphic rocks, such as marble. Fossils are never found in igneous rocks, such as granite. Sediment layers settle over the 

### msmarco_test_000146

Query: disadvantages of north during civil war

**BM25** positive rank: 1
- rank 1 | label=1 | score=2.7601 | msmarco_test_000146_pos_2 | The Civil War began in the year 1861 and ended four years later. The end result was the Union becoming victorious in 1865. There are many advantages and disadvantages that both sides faced during the war, which ultimatel
- rank 2 | label=0 | score=2.6435 | msmarco_test_000146_neg_6 | The South claimed just 9 million people — including 3.5 million slaves — in 11 confederate states. Despite the North's greater population, however, the South had an army almost equal in size during the first year of the 
- rank 3 | label=0 | score=2.0469 | msmarco_test_000146_neg_7 | At the outbreak of the American Civil War, both the North and South believed the conflict would be over quickly. But advantages for both the Confederacy and the Union meant a prolonged war between the states. In this les
- rank 4 | label=0 | score=1.4775 | msmarco_test_000146_neg_3 | Even though they have many advantages in the Civil War, the Confederates also have some disadvantages. First, instead of using a machines to do make their weapons, railroad tracks and other supplies, they used old fashio
- rank 5 | label=0 | score=0.6516 | msmarco_test_000146_neg_5 | Confederate Advantages. The Confederates had many advantages in the Civil War. First, they had an advantage by fighting a defensive war. It was the Union that started it in the first place because they were calling each 

**MLP Jittor** positive rank: 1
- rank 1 | label=1 | score=-2.7002 | msmarco_test_000146_pos_2 | The Civil War began in the year 1861 and ended four years later. The end result was the Union becoming victorious in 1865. There are many advantages and disadvantages that both sides faced during the war, which ultimatel
- rank 2 | label=0 | score=-3.3991 | msmarco_test_000146_neg_3 | Even though they have many advantages in the Civil War, the Confederates also have some disadvantages. First, instead of using a machines to do make their weapons, railroad tracks and other supplies, they used old fashio
- rank 3 | label=0 | score=-5.2770 | msmarco_test_000146_neg_7 | At the outbreak of the American Civil War, both the North and South believed the conflict would be over quickly. But advantages for both the Confederacy and the Union meant a prolonged war between the states. In this les
- rank 4 | label=0 | score=-8.2140 | msmarco_test_000146_neg_6 | The South claimed just 9 million people — including 3.5 million slaves — in 11 confederate states. Despite the North's greater population, however, the South had an army almost equal in size during the first year of the 
- rank 5 | label=0 | score=-8.4361 | msmarco_test_000146_neg_5 | Confederate Advantages. The Confederates had many advantages in the Civil War. First, they had an advantage by fighting a defensive war. It was the Union that started it in the first place because they were calling each 

**TextCNN Jittor** positive rank: 7
- rank 1 | label=0 | score=7.7069 | msmarco_test_000146_neg_6 | The South claimed just 9 million people — including 3.5 million slaves — in 11 confederate states. Despite the North's greater population, however, the South had an army almost equal in size during the first year of the 
- rank 2 | label=0 | score=7.6446 | msmarco_test_000146_neg_7 | At the outbreak of the American Civil War, both the North and South believed the conflict would be over quickly. But advantages for both the Confederacy and the Union meant a prolonged war between the states. In this les
- rank 3 | label=0 | score=7.6380 | msmarco_test_000146_neg_1 | Union Advantages. The Union had many advantages in the Civil War. First they had a bigger population which was 22 million people over 9 million people. Since their population was bigger, they had more workers which was 1
- rank 4 | label=0 | score=6.2779 | msmarco_test_000146_neg_4 | 1 Although the Confederacy had good leaders too, the Union had more. 2  The Union states had leaders like Ulysses S. Grant, William Sherman, Philip Sheridan, and George Thomas, among others. 3  The biggest leader though 
- rank 5 | label=0 | score=5.0792 | msmarco_test_000146_neg_5 | Confederate Advantages. The Confederates had many advantages in the Civil War. First, they had an advantage by fighting a defensive war. It was the Union that started it in the first place because they were calling each 

**Cross-Encoder** positive rank: 4
- rank 1 | label=0 | score=3.7338 | msmarco_test_000146_neg_3 | Even though they have many advantages in the Civil War, the Confederates also have some disadvantages. First, instead of using a machines to do make their weapons, railroad tracks and other supplies, they used old fashio
- rank 2 | label=0 | score=3.3703 | msmarco_test_000146_neg_7 | At the outbreak of the American Civil War, both the North and South believed the conflict would be over quickly. But advantages for both the Confederacy and the Union meant a prolonged war between the states. In this les
- rank 3 | label=0 | score=2.3879 | msmarco_test_000146_neg_5 | Confederate Advantages. The Confederates had many advantages in the Civil War. First, they had an advantage by fighting a defensive war. It was the Union that started it in the first place because they were calling each 
- rank 4 | label=1 | score=1.8628 | msmarco_test_000146_pos_2 | The Civil War began in the year 1861 and ended four years later. The end result was the Union becoming victorious in 1865. There are many advantages and disadvantages that both sides faced during the war, which ultimatel
- rank 5 | label=0 | score=1.2682 | msmarco_test_000146_neg_6 | The South claimed just 9 million people — including 3.5 million slaves — in 11 confederate states. Despite the North's greater population, however, the South had an army almost equal in size during the first year of the 

## All methods fail

### msmarco_test_000012

Query: escitalopram dosage for depression

**BM25** positive rank: 7
- rank 1 | label=0 | score=1.4480 | msmarco_test_000012_neg_6 | Escitalopram Dosage Recommendations. The recommended starting dosage of escitalopram for adults or adolescents (age 12 to 17 years old) with depression or adults with generalized anxiety disorder is escitalopram 10 mg on
- rank 2 | label=0 | score=1.0692 | msmarco_test_000012_neg_3 | Follow the directions on your prescription label carefully. Your doctor will determine the best dose for you. The recommended dosage of escitalopram is 10-20 mg a day. Exceptions include the elderly and patients with hep
- rank 3 | label=0 | score=0.9072 | msmarco_test_000012_neg_2 | In general, people will take escitalopram for several months or longer. Individual dosages of escitalopram can range from 10 mg to 20 mg once a day. For elderly people or people with liver problems, the maximum dose of e
- rank 4 | label=0 | score=0.8821 | msmarco_test_000012_neg_1 | For people who are taking Lexapro for the treatment of depression or generalized anxiety disorder, dosage recommendations typically range from 10 mg to 20 mg once a day. Individual doses of Lexapro can range from 10 mg t
- rank 5 | label=0 | score=0.6744 | msmarco_test_000012_neg_5 | The medication is taken once a day, most commonly in the morning, but some prefer to take it in the evening or at noontime. For all patients, 10 mg/day is the recommended starting dose of LEXAPRO. 10 mg/day is also the m

**MLP Jittor** positive rank: 6
- rank 1 | label=0 | score=-1.1875 | msmarco_test_000012_neg_6 | Escitalopram Dosage Recommendations. The recommended starting dosage of escitalopram for adults or adolescents (age 12 to 17 years old) with depression or adults with generalized anxiety disorder is escitalopram 10 mg on
- rank 2 | label=0 | score=-1.3944 | msmarco_test_000012_neg_3 | Follow the directions on your prescription label carefully. Your doctor will determine the best dose for you. The recommended dosage of escitalopram is 10-20 mg a day. Exceptions include the elderly and patients with hep
- rank 3 | label=0 | score=-2.2987 | msmarco_test_000012_neg_2 | In general, people will take escitalopram for several months or longer. Individual dosages of escitalopram can range from 10 mg to 20 mg once a day. For elderly people or people with liver problems, the maximum dose of e
- rank 4 | label=0 | score=-2.7443 | msmarco_test_000012_neg_5 | The medication is taken once a day, most commonly in the morning, but some prefer to take it in the evening or at noontime. For all patients, 10 mg/day is the recommended starting dose of LEXAPRO. 10 mg/day is also the m
- rank 5 | label=0 | score=-3.1344 | msmarco_test_000012_neg_4 | The typical starting dose for escitalopram in adolescents or adults is 10 mg taken once daily. It may be taken in the morning or evening, with or without food. In patients who do not respond to the initial regimen, the d

**TextCNN Jittor** positive rank: 6
- rank 1 | label=0 | score=15.7294 | msmarco_test_000012_neg_2 | In general, people will take escitalopram for several months or longer. Individual dosages of escitalopram can range from 10 mg to 20 mg once a day. For elderly people or people with liver problems, the maximum dose of e
- rank 2 | label=0 | score=9.2521 | msmarco_test_000012_neg_6 | Escitalopram Dosage Recommendations. The recommended starting dosage of escitalopram for adults or adolescents (age 12 to 17 years old) with depression or adults with generalized anxiety disorder is escitalopram 10 mg on
- rank 3 | label=0 | score=9.2280 | msmarco_test_000012_neg_4 | The typical starting dose for escitalopram in adolescents or adults is 10 mg taken once daily. It may be taken in the morning or evening, with or without food. In patients who do not respond to the initial regimen, the d
- rank 4 | label=0 | score=9.0043 | msmarco_test_000012_neg_9 | The recommended dose of Lexapro is 10 mg once daily. A fixed-dose trial of Lexapro demonstrated the effectiveness of both 10 mg and 20 mg of Lexapro, but failed to demonstrate a greater benefit of 20 mg over 10 mg [see C
- rank 5 | label=0 | score=6.5917 | msmarco_test_000012_neg_8 | Adolescents. The recommended dose of Lexapro is 10 mg once daily. A flexible-dose trial of Lexapro (10 to 20 mg/day) demonstrated the effectiveness of Lexapro [see Clinical Studies ]. If the dose is increased to 20 mg, t

**Cross-Encoder** positive rank: 7
- rank 1 | label=0 | score=9.9438 | msmarco_test_000012_neg_6 | Escitalopram Dosage Recommendations. The recommended starting dosage of escitalopram for adults or adolescents (age 12 to 17 years old) with depression or adults with generalized anxiety disorder is escitalopram 10 mg on
- rank 2 | label=0 | score=8.6804 | msmarco_test_000012_neg_3 | Follow the directions on your prescription label carefully. Your doctor will determine the best dose for you. The recommended dosage of escitalopram is 10-20 mg a day. Exceptions include the elderly and patients with hep
- rank 3 | label=0 | score=6.5217 | msmarco_test_000012_neg_4 | The typical starting dose for escitalopram in adolescents or adults is 10 mg taken once daily. It may be taken in the morning or evening, with or without food. In patients who do not respond to the initial regimen, the d
- rank 4 | label=0 | score=6.0177 | msmarco_test_000012_neg_2 | In general, people will take escitalopram for several months or longer. Individual dosages of escitalopram can range from 10 mg to 20 mg once a day. For elderly people or people with liver problems, the maximum dose of e
- rank 5 | label=0 | score=2.4629 | msmarco_test_000012_neg_5 | The medication is taken once a day, most commonly in the morning, but some prefer to take it in the evening or at noontime. For all patients, 10 mg/day is the recommended starting dose of LEXAPRO. 10 mg/day is also the m

### msmarco_test_000412

Query: UTEP tuition and fees

**BM25** positive rank: 7
- rank 1 | label=0 | score=1.7405 | msmarco_test_000412_neg_2 | Course fees are authorized by the UTEP administration and The University of Texas System Board of Regents. Policies governing payment or refund of tuition, fees and other charges are approved by the Board of Regents of T
- rank 2 | label=0 | score=1.7141 | msmarco_test_000412_neg_5 | Course fees are authorized by the UTEP administration and The University of Texas System Board of Regents. Policies governing payment or refund of tuition, fees and other charges are approved by the Board of Regents of T
- rank 3 | label=0 | score=1.6985 | msmarco_test_000412_neg_4 | Pursuant to state law, The University of Texas System Board of Regents (the Board) is authorized to set tuition. Tuition and fees are subject to change by legislative or regental action and become effective on the date e
- rank 4 | label=0 | score=1.6369 | msmarco_test_000412_neg_9 | The Texas Legislature does not set a specific amount for any particular student fee. The student fees assessed are authorized by state statute; however, the specific fee amounts and the determination to increase fees are
- rank 5 | label=0 | score=1.5402 | msmarco_test_000412_neg_7 | Tuition and Fees. Tuition and fees are subject to university approval and may change without notice by action of the Texas State Legislature and the University of Texas at El Paso Board of Trustees. Learn more here.

**MLP Jittor** positive rank: 7
- rank 1 | label=0 | score=0.1584 | msmarco_test_000412_neg_7 | Tuition and Fees. Tuition and fees are subject to university approval and may change without notice by action of the Texas State Legislature and the University of Texas at El Paso Board of Trustees. Learn more here.
- rank 2 | label=0 | score=-0.5563 | msmarco_test_000412_neg_6 | Tuition and Fees ‐ The average cost of tuition and fees for a typical student based on enrolling for 30 credit hours per academic year including 6 hours in the summer semester.
- rank 3 | label=0 | score=-0.9068 | msmarco_test_000412_neg_1 | Compliance Checklist requirements are approximately $130. General Courses: In-state students will pay UTEP's standard in-state tuition rates, including applicable fees. This comes to approximately $275 per SCH for in-sta
- rank 4 | label=0 | score=-1.0141 | msmarco_test_000412_neg_2 | Course fees are authorized by the UTEP administration and The University of Texas System Board of Regents. Policies governing payment or refund of tuition, fees and other charges are approved by the Board of Regents of T
- rank 5 | label=0 | score=-1.1262 | msmarco_test_000412_neg_4 | Pursuant to state law, The University of Texas System Board of Regents (the Board) is authorized to set tuition. Tuition and fees are subject to change by legislative or regental action and become effective on the date e

**TextCNN Jittor** positive rank: 6
- rank 1 | label=0 | score=7.1456 | msmarco_test_000412_neg_6 | Tuition and Fees ‐ The average cost of tuition and fees for a typical student based on enrolling for 30 credit hours per academic year including 6 hours in the summer semester.
- rank 2 | label=0 | score=5.9431 | msmarco_test_000412_neg_4 | Pursuant to state law, The University of Texas System Board of Regents (the Board) is authorized to set tuition. Tuition and fees are subject to change by legislative or regental action and become effective on the date e
- rank 3 | label=0 | score=5.9318 | msmarco_test_000412_neg_1 | Compliance Checklist requirements are approximately $130. General Courses: In-state students will pay UTEP's standard in-state tuition rates, including applicable fees. This comes to approximately $275 per SCH for in-sta
- rank 4 | label=0 | score=5.0678 | msmarco_test_000412_neg_9 | The Texas Legislature does not set a specific amount for any particular student fee. The student fees assessed are authorized by state statute; however, the specific fee amounts and the determination to increase fees are
- rank 5 | label=0 | score=4.8081 | msmarco_test_000412_neg_3 | 228.63. 235.46. 1 Resident undergraduate tuition as established by the Texas Legislature and the Board of Regents is $168.78/semester credit hours (SCH); non-residents undergraduate tuition is $478.78/SCH. 2 Required fee

**Cross-Encoder** positive rank: 6
- rank 1 | label=0 | score=7.5911 | msmarco_test_000412_neg_1 | Compliance Checklist requirements are approximately $130. General Courses: In-state students will pay UTEP's standard in-state tuition rates, including applicable fees. This comes to approximately $275 per SCH for in-sta
- rank 2 | label=0 | score=7.5846 | msmarco_test_000412_neg_2 | Course fees are authorized by the UTEP administration and The University of Texas System Board of Regents. Policies governing payment or refund of tuition, fees and other charges are approved by the Board of Regents of T
- rank 3 | label=0 | score=7.2859 | msmarco_test_000412_neg_5 | Course fees are authorized by the UTEP administration and The University of Texas System Board of Regents. Policies governing payment or refund of tuition, fees and other charges are approved by the Board of Regents of T
- rank 4 | label=0 | score=6.5360 | msmarco_test_000412_neg_4 | Pursuant to state law, The University of Texas System Board of Regents (the Board) is authorized to set tuition. Tuition and fees are subject to change by legislative or regental action and become effective on the date e
- rank 5 | label=0 | score=6.4601 | msmarco_test_000412_neg_9 | The Texas Legislature does not set a specific amount for any particular student fee. The student fees assessed are authorized by state statute; however, the specific fee amounts and the determination to increase fees are

## Jittor lightweight models lag Cross-Encoder

### msmarco_test_000001

Query: how long does it take to drive from st louis to detroit

**BM25** positive rank: 1
- rank 1 | label=1 | score=6.2511 | msmarco_test_000001_pos_8 | How long is the drive from Detroit, MI to Saint Louis, MO? The total driving time is 8 hours, 43 minutes.
- rank 2 | label=0 | score=3.8144 | msmarco_test_000001_neg_2 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa Claus and to the Ho
- rank 3 | label=0 | score=3.1887 | msmarco_test_000001_neg_3 | Your trip begins in Saint Louis, Missouri. It ends in Detroit, Michigan. If you're planning a road trip, you might be interested in seeing the total driving distance from Saint Louis, MO to Detroit, MI.
- rank 4 | label=0 | score=2.9807 | msmarco_test_000001_neg_1 | Since this is a long drive, you might want to stop halfway and stay overnight in a hotel. You can find the city that is halfway between Saint Louis, MO and Detroit, MI.
- rank 5 | label=0 | score=2.5266 | msmarco_test_000001_neg_4 | Merge onto I-71 south and stay on it for approximately 1 hour and 10 minutes, then take Exit 1 to I-64 west. Continue on I-64 west for approximately 1 hour then take Exit 63. Turn left (south) onto Highway 162 and drive 

**MLP Jittor** positive rank: 9
- rank 1 | label=0 | score=-3.5292 | msmarco_test_000001_neg_2 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa Claus and to the Ho
- rank 2 | label=0 | score=-5.6620 | msmarco_test_000001_neg_9 | The distance between Missouri, USA, and Detroit, MI, is 660 miles and will take about 10 Hours 20 Minutes of driving time.
- rank 3 | label=0 | score=-6.4313 | msmarco_test_000001_neg_6 | You can also calculate the cost to drive from Saint Louis, MO to Detroit, MI based on current local gas prices and an estimate of your car's best gas mileage.
- rank 4 | label=0 | score=-6.6795 | msmarco_test_000001_neg_4 | Merge onto I-71 south and stay on it for approximately 1 hour and 10 minutes, then take Exit 1 to I-64 west. Continue on I-64 west for approximately 1 hour then take Exit 63. Turn left (south) onto Highway 162 and drive 
- rank 5 | label=0 | score=-6.6961 | msmarco_test_000001_neg_5 | – Take I-75 south to southwestern Ohio, then I-71 south into Louisville, KY. Take I-64 west into Indiana. Stay on I-64 west for nearly an hour, then take Exit 63. Turn left (south) onto Highway 162 and drive 7 miles unti

**TextCNN Jittor** positive rank: 7
- rank 1 | label=0 | score=7.2690 | msmarco_test_000001_neg_2 | From St. Louis, Missouri. – Drive on I-64 east for approximately 3 hours; take Exit #57-A South and head south on US 231 to the State Road 162 exit, which will take you directly into the town of Santa Claus and to the Ho
- rank 2 | label=0 | score=6.6562 | msmarco_test_000001_neg_3 | Your trip begins in Saint Louis, Missouri. It ends in Detroit, Michigan. If you're planning a road trip, you might be interested in seeing the total driving distance from Saint Louis, MO to Detroit, MI.
- rank 3 | label=0 | score=5.0788 | msmarco_test_000001_neg_9 | The distance between Missouri, USA, and Detroit, MI, is 660 miles and will take about 10 Hours 20 Minutes of driving time.
- rank 4 | label=0 | score=4.7150 | msmarco_test_000001_neg_5 | – Take I-75 south to southwestern Ohio, then I-71 south into Louisville, KY. Take I-64 west into Indiana. Stay on I-64 west for nearly an hour, then take Exit 63. Turn left (south) onto Highway 162 and drive 7 miles unti
- rank 5 | label=0 | score=3.7280 | msmarco_test_000001_neg_10 | The driving distance between Detroit, MI and Nashville, TN is approximately 535 miles. The driving time would be approximately 8 hours 30 minutes if you were to travel non-sto … p in good driving conditions.

**Cross-Encoder** positive rank: 1
- rank 1 | label=1 | score=9.5204 | msmarco_test_000001_pos_8 | How long is the drive from Detroit, MI to Saint Louis, MO? The total driving time is 8 hours, 43 minutes.
- rank 2 | label=0 | score=8.0273 | msmarco_test_000001_neg_7 | 552 mi (888km) - about 8 hours 39 mins Saint Louis, MO, USA to Detroit, MI, USA.
- rank 3 | label=0 | score=6.3973 | msmarco_test_000001_neg_9 | The distance between Missouri, USA, and Detroit, MI, is 660 miles and will take about 10 Hours 20 Minutes of driving time.
- rank 4 | label=0 | score=3.1113 | msmarco_test_000001_neg_10 | The driving distance between Detroit, MI and Nashville, TN is approximately 535 miles. The driving time would be approximately 8 hours 30 minutes if you were to travel non-sto … p in good driving conditions.
- rank 5 | label=0 | score=2.5533 | msmarco_test_000001_neg_3 | Your trip begins in Saint Louis, Missouri. It ends in Detroit, Michigan. If you're planning a road trip, you might be interested in seeing the total driving distance from Saint Louis, MO to Detroit, MI.

### msmarco_test_000005

Query: what is the cause of HPV

**BM25** positive rank: 1
- rank 1 | label=1 | score=2.0256 | msmarco_test_000005_pos_1 | Human papilloma virus (HPV) is the major cause of cervical cancer. There are many different types of HPV. Some types of HPV cause genital warts and are sometimes called the genital wart virus. The types of HPV that cause
- rank 2 | label=0 | score=1.8595 | msmarco_test_000005_neg_6 | 1. The strains of HPV known to cause genital warts are low-risk HPV 6 and 11, while the strains of HPV associated with cancer include high-risk HPV 16, 18, 31, 33, 45, 52 and 58. 2. HPV is a virus which is passed skin-to
- rank 3 | label=0 | score=1.4660 | msmarco_test_000005_neg_2 | There are more than 100 types of HPV. About 30 or so types can cause genital infections. Some can cause genital warts; other types can cause cervical or other genital cancers. The other 70 or so HPV types can cause infec
- rank 4 | label=0 | score=1.4395 | msmarco_test_000005_neg_5 | Other types of HPV can cause warts and verrucas. These types of HPV are sometimes called the wart virus or genital wart virus and they include types 6 and 11. Warts and verrucas are most common on the hands and feet, in 
- rank 5 | label=0 | score=1.3091 | msmarco_test_000005_neg_3 | HPV is a different virus than HIV and HSV (herpes). HPV is so common that nearly all sexually active men and women get it at some point in their lives. There are many different types of HPV. Some types can cause health p

**MLP Jittor** positive rank: 5
- rank 1 | label=0 | score=-2.0115 | msmarco_test_000005_neg_6 | 1. The strains of HPV known to cause genital warts are low-risk HPV 6 and 11, while the strains of HPV associated with cancer include high-risk HPV 16, 18, 31, 33, 45, 52 and 58. 2. HPV is a virus which is passed skin-to
- rank 2 | label=0 | score=-3.5243 | msmarco_test_000005_neg_3 | HPV is a different virus than HIV and HSV (herpes). HPV is so common that nearly all sexually active men and women get it at some point in their lives. There are many different types of HPV. Some types can cause health p
- rank 3 | label=0 | score=-3.6245 | msmarco_test_000005_neg_4 | Certain HPV types are classified as high-risk because they lead to abnormal cell changes and can cause genital cancers: cervical cancer as well as cancer of the vulva, anus, and penis. In fact, researchers say that virtu
- rank 4 | label=0 | score=-3.8375 | msmarco_test_000005_neg_7 | HPV stands for human papillomavirus. These viruses are the direct causes of HPV and infect the skin or mucous membranes (mucous membranes are the moist, inner lining of some organs and body cavities, such as the nose, mo
- rank 5 | label=1 | score=-3.9331 | msmarco_test_000005_pos_1 | Human papilloma virus (HPV) is the major cause of cervical cancer. There are many different types of HPV. Some types of HPV cause genital warts and are sometimes called the genital wart virus. The types of HPV that cause

**TextCNN Jittor** positive rank: 6
- rank 1 | label=0 | score=7.7862 | msmarco_test_000005_neg_5 | Other types of HPV can cause warts and verrucas. These types of HPV are sometimes called the wart virus or genital wart virus and they include types 6 and 11. Warts and verrucas are most common on the hands and feet, in 
- rank 2 | label=0 | score=4.9942 | msmarco_test_000005_neg_2 | There are more than 100 types of HPV. About 30 or so types can cause genital infections. Some can cause genital warts; other types can cause cervical or other genital cancers. The other 70 or so HPV types can cause infec
- rank 3 | label=0 | score=4.7021 | msmarco_test_000005_neg_4 | Certain HPV types are classified as high-risk because they lead to abnormal cell changes and can cause genital cancers: cervical cancer as well as cancer of the vulva, anus, and penis. In fact, researchers say that virtu
- rank 4 | label=0 | score=4.6173 | msmarco_test_000005_neg_6 | 1. The strains of HPV known to cause genital warts are low-risk HPV 6 and 11, while the strains of HPV associated with cancer include high-risk HPV 16, 18, 31, 33, 45, 52 and 58. 2. HPV is a virus which is passed skin-to
- rank 5 | label=0 | score=3.7907 | msmarco_test_000005_neg_7 | HPV stands for human papillomavirus. These viruses are the direct causes of HPV and infect the skin or mucous membranes (mucous membranes are the moist, inner lining of some organs and body cavities, such as the nose, mo

**Cross-Encoder** positive rank: 1
- rank 1 | label=1 | score=6.7990 | msmarco_test_000005_pos_1 | Human papilloma virus (HPV) is the major cause of cervical cancer. There are many different types of HPV. Some types of HPV cause genital warts and are sometimes called the genital wart virus. The types of HPV that cause
- rank 2 | label=0 | score=6.6955 | msmarco_test_000005_neg_7 | HPV stands for human papillomavirus. These viruses are the direct causes of HPV and infect the skin or mucous membranes (mucous membranes are the moist, inner lining of some organs and body cavities, such as the nose, mo
- rank 3 | label=0 | score=4.9885 | msmarco_test_000005_neg_4 | Certain HPV types are classified as high-risk because they lead to abnormal cell changes and can cause genital cancers: cervical cancer as well as cancer of the vulva, anus, and penis. In fact, researchers say that virtu
- rank 4 | label=0 | score=4.8016 | msmarco_test_000005_neg_6 | 1. The strains of HPV known to cause genital warts are low-risk HPV 6 and 11, while the strains of HPV associated with cancer include high-risk HPV 16, 18, 31, 33, 45, 52 and 58. 2. HPV is a virus which is passed skin-to
- rank 5 | label=0 | score=3.7631 | msmarco_test_000005_neg_2 | There are more than 100 types of HPV. About 30 or so types can cause genital infections. Some can cause genital warts; other types can cause cervical or other genital cancers. The other 70 or so HPV types can cause infec

### msmarco_test_000008

Query: how tall was lebron james parents

**BM25** positive rank: 1
- rank 1 | label=1 | score=2.9148 | msmarco_test_000008_pos_4 | I'm always interested in the height of really tall people's parents because I'm wondering if their height is genetic. Kobe Bryant's father's height is 6'9 so his height is obviously genetic. I heard that Lebron James' da
- rank 2 | label=0 | score=2.1265 | msmarco_test_000008_neg_1 | How tall was LeBron James at the age of 14? lebron James was 6 feet getting into the 8th grade.i am 13 years  old right now, and on my 13th birthday, i was 5'6 1/2'', my 14th  birthday is in July this year, and i am a … 
- rank 3 | label=0 | score=1.1160 | msmarco_test_000008_neg_2 | Lebron James is 6 feet 8 inches tall. For those that use the metric system he is 2.03 meters tall. 2.03 meters is equal to 203 centimeters.
- rank 4 | label=0 | score=0.9258 | msmarco_test_000008_neg_7 | (Click here for a complete listing of today's sports birthdays.) His mother, Gloria James, was only 16 at the time. His biological father, Anthony McClelland, was an ex-con uninterested in being a parent. Gloria raised L
- rank 5 | label=0 | score=0.8279 | msmarco_test_000008_neg_5 | TMZ has learned ... LeBron James and his mother Gloria James are being sued for millions by a man who claims he tried to prove he's the NBA star's biological father -- but LeBron and his mom tampered with the evidence in

**MLP Jittor** positive rank: 7
- rank 1 | label=0 | score=-2.3794 | msmarco_test_000008_neg_7 | (Click here for a complete listing of today's sports birthdays.) His mother, Gloria James, was only 16 at the time. His biological father, Anthony McClelland, was an ex-con uninterested in being a parent. Gloria raised L
- rank 2 | label=0 | score=-2.8341 | msmarco_test_000008_neg_5 | TMZ has learned ... LeBron James and his mother Gloria James are being sued for millions by a man who claims he tried to prove he's the NBA star's biological father -- but LeBron and his mom tampered with the evidence in
- rank 3 | label=0 | score=-4.0528 | msmarco_test_000008_neg_3 | LeBron James is currently in 2010, 6 foot 9 inches. He has grown with his age, when he first came to the NBA he was 6 foot 8 as most people recall and 240 pounds but now he is … 6 foot 9 280 pounds. He is a player with a
- rank 4 | label=0 | score=-4.1221 | msmarco_test_000008_neg_6 | But Gloria is not the only mother he will be celebrating this Mother's Day. James shares his life with his high school sweetheart, Savannah Brinson, the mother of his two sons, LeBron Jr., 5, and Bryce Maximus, 2. The im
- rank 5 | label=0 | score=-5.6505 | msmarco_test_000008_neg_1 | How tall was LeBron James at the age of 14? lebron James was 6 feet getting into the 8th grade.i am 13 years  old right now, and on my 13th birthday, i was 5'6 1/2'', my 14th  birthday is in July this year, and i am a … 

**TextCNN Jittor** positive rank: 6
- rank 1 | label=0 | score=7.2194 | msmarco_test_000008_neg_7 | (Click here for a complete listing of today's sports birthdays.) His mother, Gloria James, was only 16 at the time. His biological father, Anthony McClelland, was an ex-con uninterested in being a parent. Gloria raised L
- rank 2 | label=0 | score=7.0063 | msmarco_test_000008_neg_5 | TMZ has learned ... LeBron James and his mother Gloria James are being sued for millions by a man who claims he tried to prove he's the NBA star's biological father -- but LeBron and his mom tampered with the evidence in
- rank 3 | label=0 | score=5.3541 | msmarco_test_000008_neg_3 | LeBron James is currently in 2010, 6 foot 9 inches. He has grown with his age, when he first came to the NBA he was 6 foot 8 as most people recall and 240 pounds but now he is … 6 foot 9 280 pounds. He is a player with a
- rank 4 | label=0 | score=4.3092 | msmarco_test_000008_neg_6 | But Gloria is not the only mother he will be celebrating this Mother's Day. James shares his life with his high school sweetheart, Savannah Brinson, the mother of his two sons, LeBron Jr., 5, and Bryce Maximus, 2. The im
- rank 5 | label=0 | score=3.9721 | msmarco_test_000008_neg_2 | Lebron James is 6 feet 8 inches tall. For those that use the metric system he is 2.03 meters tall. 2.03 meters is equal to 203 centimeters.

**Cross-Encoder** positive rank: 1
- rank 1 | label=1 | score=8.3550 | msmarco_test_000008_pos_4 | I'm always interested in the height of really tall people's parents because I'm wondering if their height is genetic. Kobe Bryant's father's height is 6'9 so his height is obviously genetic. I heard that Lebron James' da
- rank 2 | label=0 | score=6.2085 | msmarco_test_000008_neg_2 | Lebron James is 6 feet 8 inches tall. For those that use the metric system he is 2.03 meters tall. 2.03 meters is equal to 203 centimeters.
- rank 3 | label=0 | score=5.7981 | msmarco_test_000008_neg_1 | How tall was LeBron James at the age of 14? lebron James was 6 feet getting into the 8th grade.i am 13 years  old right now, and on my 13th birthday, i was 5'6 1/2'', my 14th  birthday is in July this year, and i am a … 
- rank 4 | label=0 | score=5.0897 | msmarco_test_000008_neg_3 | LeBron James is currently in 2010, 6 foot 9 inches. He has grown with his age, when he first came to the NBA he was 6 foot 8 as most people recall and 240 pounds but now he is … 6 foot 9 280 pounds. He is a player with a
- rank 5 | label=0 | score=3.1262 | msmarco_test_000008_neg_7 | (Click here for a complete listing of today's sports birthdays.) His mother, Gloria James, was only 16 at the time. His biological father, Anthony McClelland, was an ex-con uninterested in being a parent. Gloria raised L
