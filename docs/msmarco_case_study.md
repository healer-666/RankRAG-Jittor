# MS MARCO Case Study and Error Analysis

These examples are selected from the small MS MARCO subset and use automatic rule-based notes. They should be read as qualitative diagnostics, not proof of generalization.

## Success Cases

### msmarco_test_000001

Query: how much does an lpn make an hour in illinois

**TF-IDF** positive rank: 3

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (an, hour, in, lpn, make), which can help lexical baselines.

- rank 1 | msmarco_test_000001_neg_2 | label=0 | score=0.1533 | An LPN, or licensed practical nurse makes, on average $19.89 an hour in Chicago, IL. That means they earn an annually an income of roughly $41,370. From what I have been told, they make about $20 hr. 
- rank 2 | msmarco_test_000001_neg_4 | label=0 | score=0.1089 | Top Cities in Illinois for LPNs: Chicago. Illinois has a growing number of opportunities in the nursing field. Within the state, LPNs make up 17 % of nurses in the state. The Illinois LPN comfort scor
- rank 3 | msmarco_test_000001_pos_1 | label=1 | score=0.0604 | Bureau of Labor Statistics information for LPNs is classified under Licensed Practical & Licensed Vocational Nurses. A very popular profession that is always in the need of more workers had a median s
- rank 4 | msmarco_test_000001_neg_3 | label=0 | score=0.0419 | As you can see from our salary by state LPN wages listed below a licensed practical nurse can expect to make as much $51,000 a year if getting employed in one of the highest paying states. The total n
- rank 5 | msmarco_test_000001_neg_5 | label=0 | score=0.0155 | LPN Salary. According to information taken from the Bureau of Labor Statistics (BLS), the median annual LPN salary is $39,030. The middle 50% of LPNs make wages of $33,360 and $46,710 annually. The hi

**BM25** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (illinois, in, lpn, make), which can help lexical baselines.

- rank 1 | msmarco_test_000001_neg_4 | label=0 | score=2.8253 | Top Cities in Illinois for LPNs: Chicago. Illinois has a growing number of opportunities in the nursing field. Within the state, LPNs make up 17 % of nurses in the state. The Illinois LPN comfort scor
- rank 2 | msmarco_test_000001_neg_2 | label=0 | score=2.1749 | An LPN, or licensed practical nurse makes, on average $19.89 an hour in Chicago, IL. That means they earn an annually an income of roughly $41,370. From what I have been told, they make about $20 hr. 
- rank 3 | msmarco_test_000001_neg_3 | label=0 | score=1.6440 | As you can see from our salary by state LPN wages listed below a licensed practical nurse can expect to make as much $51,000 a year if getting employed in one of the highest paying states. The total n
- rank 4 | msmarco_test_000001_pos_1 | label=1 | score=1.3494 | Bureau of Labor Statistics information for LPNs is classified under Licensed Practical & Licensed Vocational Nurses. A very popular profession that is always in the need of more workers had a median s
- rank 5 | msmarco_test_000001_neg_5 | label=0 | score=0.4481 | LPN Salary. According to information taken from the Bureau of Labor Statistics (BLS), the median annual LPN salary is $39,030. The middle 50% of LPNs make wages of $33,360 and $46,710 annually. The hi

**TextCNN Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (an, hour, in, lpn), which can help lexical baselines.

- rank 1 | msmarco_test_000001_pos_1 | label=1 | score=6.4522 | Bureau of Labor Statistics information for LPNs is classified under Licensed Practical & Licensed Vocational Nurses. A very popular profession that is always in the need of more workers had a median s
- rank 2 | msmarco_test_000001_neg_3 | label=0 | score=6.3413 | As you can see from our salary by state LPN wages listed below a licensed practical nurse can expect to make as much $51,000 a year if getting employed in one of the highest paying states. The total n
- rank 3 | msmarco_test_000001_neg_5 | label=0 | score=5.7516 | LPN Salary. According to information taken from the Bureau of Labor Statistics (BLS), the median annual LPN salary is $39,030. The middle 50% of LPNs make wages of $33,360 and $46,710 annually. The hi
- rank 4 | msmarco_test_000001_neg_2 | label=0 | score=4.9187 | An LPN, or licensed practical nurse makes, on average $19.89 an hour in Chicago, IL. That means they earn an annually an income of roughly $41,370. From what I have been told, they make about $20 hr. 
- rank 5 | msmarco_test_000001_neg_4 | label=0 | score=4.7144 | Top Cities in Illinois for LPNs: Chicago. Illinois has a growing number of opportunities in the nursing field. Within the state, LPNs make up 17 % of nurses in the state. The Illinois LPN comfort scor

### msmarco_test_000007

Query: what is stone color

**TF-IDF** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is limited (color, is), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000007_neg_4 | label=0 | score=0.1006 | Pure topaz, like pure corundum, is transparent and colorless. The wide range of topaz colors that are available is due to either natural trace impurities or defects in the crystal structure. Color div
- rank 2 | msmarco_test_000007_neg_5 | label=0 | score=0.0520 | 1 Goes Great With: See colors that combine well with your chosen color. 2  More Color Combos: Get more color combinations for your color. 3  See More Shades: View different shades of your selected col
- rank 3 | msmarco_test_000007_neg_3 | label=0 | score=0.0503 | 1 More Color Combos: Get more color combinations for your color. 2  See More Shades: View different shades of your selected color. 3  View Similar Colors: See colors resembling the one you've chosen. 
- rank 4 | msmarco_test_000007_pos_1 | label=1 | score=0.0403 | Skip the talk, take me to the stone colors See also Birthstone Colors. This page will help locate jewelry made with gemstones in a specific color. For instance, if you are looking for blue, green, pin
- rank 5 | msmarco_test_000007_neg_2 | label=0 | score=0.0362 | 1 See More Shades: View different shades of your selected color. 2  View Similar Colors: See colors resembling the one you've chosen. 3  Share: Share colors with your friends on social networks. 4  Se

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is limited (color, is), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000007_neg_4 | label=0 | score=2.1444 | Pure topaz, like pure corundum, is transparent and colorless. The wide range of topaz colors that are available is due to either natural trace impurities or defects in the crystal structure. Color div
- rank 2 | msmarco_test_000007_pos_1 | label=1 | score=1.0816 | Skip the talk, take me to the stone colors See also Birthstone Colors. This page will help locate jewelry made with gemstones in a specific color. For instance, if you are looking for blue, green, pin
- rank 3 | msmarco_test_000007_neg_5 | label=0 | score=0.2957 | 1 Goes Great With: See colors that combine well with your chosen color. 2  More Color Combos: Get more color combinations for your color. 3  See More Shades: View different shades of your selected col
- rank 4 | msmarco_test_000007_neg_3 | label=0 | score=0.2888 | 1 More Color Combos: Get more color combinations for your color. 2  See More Shades: View different shades of your selected color. 3  View Similar Colors: See colors resembling the one you've chosen. 
- rank 5 | msmarco_test_000007_neg_2 | label=0 | score=0.2720 | 1 See More Shades: View different shades of your selected color. 2  View Similar Colors: See colors resembling the one you've chosen. 3  Share: Share colors with your friends on social networks. 4  Se

**TextCNN Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is limited (color, stone), so the ranker needs more than surface matching.

- rank 1 | msmarco_test_000007_pos_1 | label=1 | score=4.4442 | Skip the talk, take me to the stone colors See also Birthstone Colors. This page will help locate jewelry made with gemstones in a specific color. For instance, if you are looking for blue, green, pin
- rank 2 | msmarco_test_000007_neg_5 | label=0 | score=4.3277 | 1 Goes Great With: See colors that combine well with your chosen color. 2  More Color Combos: Get more color combinations for your color. 3  See More Shades: View different shades of your selected col
- rank 3 | msmarco_test_000007_neg_2 | label=0 | score=4.3151 | 1 See More Shades: View different shades of your selected color. 2  View Similar Colors: See colors resembling the one you've chosen. 3  Share: Share colors with your friends on social networks. 4  Se
- rank 4 | msmarco_test_000007_neg_4 | label=0 | score=4.2970 | Pure topaz, like pure corundum, is transparent and colorless. The wide range of topaz colors that are available is due to either natural trace impurities or defects in the crystal structure. Color div
- rank 5 | msmarco_test_000007_neg_3 | label=0 | score=4.1015 | 1 More Color Combos: Get more color combinations for your color. 2  See More Shades: View different shades of your selected color. 3  View Similar Colors: See colors resembling the one you've chosen. 

### msmarco_test_000009

Query: what are environmental impacts of hydropower

**TF-IDF** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (environmental, hydropower, impacts, of), which can help lexical baselines.

- rank 1 | msmarco_test_000009_neg_3 | label=0 | score=0.1397 | Environmental impacts of hydroelectric power. 1  1. ENVIRONMENTAL IMPACTS OF HYDROPOWER. 2  2. About hydropower The water behind the dam flows through an intake and pushes against blades in a turbine,
- rank 2 | msmarco_test_000009_pos_2 | label=1 | score=0.1105 | Environmental impacts. Hydropower is emission-free, renewable energy which does not cause any emissions into the air, water, or soil. The construction of hydropower plants changes waterways and their 
- rank 3 | msmarco_test_000009_neg_1 | label=0 | score=0.0851 | 1 1. ENVIRONMENTAL IMPACTS OF HYDROPOWER. 2  2. About hydropower The water behind the dam flows through an intake and pushes against blades in a turbine, causing them to turn. 3  The turbine spins a g
- rank 4 | msmarco_test_000009_neg_4 | label=0 | score=0.0650 | A belief that 'small' hydropower systems are a source of clean energy with little or no environmental problems is driving the growing interest in mini, micro, and pico hydro systems that generate from
- rank 5 | msmarco_test_000009_neg_5 | label=0 | score=0.0380 | Hydroelectric power includes both massive hydroelectric dams and small run-of-the-river plants. Large-scale hydroelectric dams continue to be built in many parts of the world (including China and Braz

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (are, environmental, hydropower, of), which can help lexical baselines.

- rank 1 | msmarco_test_000009_neg_4 | label=0 | score=1.1270 | A belief that 'small' hydropower systems are a source of clean energy with little or no environmental problems is driving the growing interest in mini, micro, and pico hydro systems that generate from
- rank 2 | msmarco_test_000009_pos_2 | label=1 | score=1.1162 | Environmental impacts. Hydropower is emission-free, renewable energy which does not cause any emissions into the air, water, or soil. The construction of hydropower plants changes waterways and their 
- rank 3 | msmarco_test_000009_neg_3 | label=0 | score=1.0949 | Environmental impacts of hydroelectric power. 1  1. ENVIRONMENTAL IMPACTS OF HYDROPOWER. 2  2. About hydropower The water behind the dam flows through an intake and pushes against blades in a turbine,
- rank 4 | msmarco_test_000009_neg_1 | label=0 | score=0.8873 | 1 1. ENVIRONMENTAL IMPACTS OF HYDROPOWER. 2  2. About hydropower The water behind the dam flows through an intake and pushes against blades in a turbine, causing them to turn. 3  The turbine spins a g
- rank 5 | msmarco_test_000009_neg_5 | label=0 | score=0.7722 | Hydroelectric power includes both massive hydroelectric dams and small run-of-the-river plants. Large-scale hydroelectric dams continue to be built in many parts of the world (including China and Braz

**TextCNN Jittor** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (environmental, hydropower, impacts, of), which can help lexical baselines.

- rank 1 | msmarco_test_000009_pos_2 | label=1 | score=4.1978 | Environmental impacts. Hydropower is emission-free, renewable energy which does not cause any emissions into the air, water, or soil. The construction of hydropower plants changes waterways and their 
- rank 2 | msmarco_test_000009_neg_4 | label=0 | score=4.0371 | A belief that 'small' hydropower systems are a source of clean energy with little or no environmental problems is driving the growing interest in mini, micro, and pico hydro systems that generate from
- rank 3 | msmarco_test_000009_neg_5 | label=0 | score=3.7185 | Hydroelectric power includes both massive hydroelectric dams and small run-of-the-river plants. Large-scale hydroelectric dams continue to be built in many parts of the world (including China and Braz
- rank 4 | msmarco_test_000009_neg_3 | label=0 | score=2.8449 | Environmental impacts of hydroelectric power. 1  1. ENVIRONMENTAL IMPACTS OF HYDROPOWER. 2  2. About hydropower The water behind the dam flows through an intake and pushes against blades in a turbine,
- rank 5 | msmarco_test_000009_neg_1 | label=0 | score=2.4808 | 1 1. ENVIRONMENTAL IMPACTS OF HYDROPOWER. 2  2. About hydropower The water behind the dam flows through an intake and pushes against blades in a turbine, causing them to turn. 3  The turbine spins a g

## Middle Cases

### msmarco_test_000003

Query: common extensor tendon origin and insertion

**TF-IDF** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (and, common, extensor, origin, tendon), which can help lexical baselines.

- rank 1 | msmarco_test_000003_neg_5 | label=0 | score=0.2760 | Tennis elbow. Tendon pathology involving the common extensor tendon origin is the most common cause of elbow pain and/or disability, and is much more common than tendinous injury of the common flexor 
- rank 2 | msmarco_test_000003_neg_2 | label=0 | score=0.1762 | Origin: Common extensor tendon from latearl epicondyle of humerus and deep antebrachial fascia Insertion: into expansion of little finger with extensor digitorum tendon Action: Extend MCP joints and I
- rank 3 | msmarco_test_000003_neg_1 | label=0 | score=0.1430 | Origin: Common extensor tendon from lateral epicondyle of humerus, and deep antebrachial fascia. Insertion: By four tendons, each penetrating a membranous expansion of the dorsum of the second to fift
- rank 4 | msmarco_test_000003_pos_4 | label=1 | score=0.1159 | ORIGIN Common extensor origin on anterior aspect of lateral epicondyle of humerus. INSERTION Extensor expansion of little finger-usually two tendons which are joined by a slip from extensor digitorum 
- rank 5 | msmarco_test_000003_neg_3 | label=0 | score=0.0598 | Extensor tendinopathy or as it is most commonly know “tennis elbow” is pain in the outside of the elbow that comes from inflammation and degeneration of the tendons on the outside of the elbow. The te

**BM25** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (and, common, extensor, insertion, origin, tendon), which can help lexical baselines.

- rank 1 | msmarco_test_000003_neg_2 | label=0 | score=1.5158 | Origin: Common extensor tendon from latearl epicondyle of humerus and deep antebrachial fascia Insertion: into expansion of little finger with extensor digitorum tendon Action: Extend MCP joints and I
- rank 2 | msmarco_test_000003_neg_5 | label=0 | score=1.3901 | Tennis elbow. Tendon pathology involving the common extensor tendon origin is the most common cause of elbow pain and/or disability, and is much more common than tendinous injury of the common flexor 
- rank 3 | msmarco_test_000003_neg_1 | label=0 | score=1.2377 | Origin: Common extensor tendon from lateral epicondyle of humerus, and deep antebrachial fascia. Insertion: By four tendons, each penetrating a membranous expansion of the dorsum of the second to fift
- rank 4 | msmarco_test_000003_pos_4 | label=1 | score=1.0057 | ORIGIN Common extensor origin on anterior aspect of lateral epicondyle of humerus. INSERTION Extensor expansion of little finger-usually two tendons which are joined by a slip from extensor digitorum 
- rank 5 | msmarco_test_000003_neg_3 | label=0 | score=0.9288 | Extensor tendinopathy or as it is most commonly know “tennis elbow” is pain in the outside of the elbow that comes from inflammation and degeneration of the tendons on the outside of the elbow. The te

**TextCNN Jittor** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (and, common, extensor, origin, tendon), which can help lexical baselines.

- rank 1 | msmarco_test_000003_neg_3 | label=0 | score=6.3023 | Extensor tendinopathy or as it is most commonly know “tennis elbow” is pain in the outside of the elbow that comes from inflammation and degeneration of the tendons on the outside of the elbow. The te
- rank 2 | msmarco_test_000003_pos_4 | label=1 | score=5.9616 | ORIGIN Common extensor origin on anterior aspect of lateral epicondyle of humerus. INSERTION Extensor expansion of little finger-usually two tendons which are joined by a slip from extensor digitorum 
- rank 3 | msmarco_test_000003_neg_1 | label=0 | score=5.5453 | Origin: Common extensor tendon from lateral epicondyle of humerus, and deep antebrachial fascia. Insertion: By four tendons, each penetrating a membranous expansion of the dorsum of the second to fift
- rank 4 | msmarco_test_000003_neg_5 | label=0 | score=5.3424 | Tennis elbow. Tendon pathology involving the common extensor tendon origin is the most common cause of elbow pain and/or disability, and is much more common than tendinous injury of the common flexor 
- rank 5 | msmarco_test_000003_neg_2 | label=0 | score=4.8857 | Origin: Common extensor tendon from latearl epicondyle of humerus and deep antebrachial fascia Insertion: into expansion of little finger with extensor digitorum tendon Action: Extend MCP joints and I

### msmarco_test_000004

Query: when was the death penalty abolished

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (abolished, death, penalty, the, was), which can help lexical baselines.

- rank 1 | msmarco_test_000004_pos_3 | label=1 | score=0.2008 | The last executions UK took place in 1964, and the death penalty was abolished in 1998. In 2004 the UK became a party to the 13th Protocol to the European Convention on Human Rights and prohibited the
- rank 2 | msmarco_test_000004_neg_1 | label=0 | score=0.1631 | Why the Death Penalty should be abolished. The risk of executing innocent people exists in any justice system There have been and always will be cases of executions of innocent people. No matter how d
- rank 3 | msmarco_test_000004_neg_4 | label=0 | score=0.1566 | In 1846, Michigan became the first state to abolish the death penalty for all crimes except treason. Later, Rhode Island and Wisconsin abolished the death penalty for all crimes. By the end of the cen
- rank 4 | msmarco_test_000004_neg_2 | label=0 | score=0.1422 | Since 2007, six states have abolished the death penalty: Maryland, Connecticut, Illinois, New Mexico and New Jersey. According to the Death Penalty Information Center, a research group opposed to the 
- rank 5 | msmarco_test_000004_neg_5 | label=0 | score=0.1107 | California currently spends $184 million on the death penalty each year and is on track to spend $1 billion in the next five years. There are many reasons the death penalty should be abolished. It is 

**BM25** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (abolished, death, penalty, the, was), which can help lexical baselines.

- rank 1 | msmarco_test_000004_pos_3 | label=1 | score=1.8736 | The last executions UK took place in 1964, and the death penalty was abolished in 1998. In 2004 the UK became a party to the 13th Protocol to the European Convention on Human Rights and prohibited the
- rank 2 | msmarco_test_000004_neg_4 | label=0 | score=1.6951 | In 1846, Michigan became the first state to abolish the death penalty for all crimes except treason. Later, Rhode Island and Wisconsin abolished the death penalty for all crimes. By the end of the cen
- rank 3 | msmarco_test_000004_neg_1 | label=0 | score=1.4394 | Why the Death Penalty should be abolished. The risk of executing innocent people exists in any justice system There have been and always will be cases of executions of innocent people. No matter how d
- rank 4 | msmarco_test_000004_neg_2 | label=0 | score=1.4127 | Since 2007, six states have abolished the death penalty: Maryland, Connecticut, Illinois, New Mexico and New Jersey. According to the Death Penalty Information Center, a research group opposed to the 
- rank 5 | msmarco_test_000004_neg_5 | label=0 | score=1.3722 | California currently spends $184 million on the death penalty each year and is on track to spend $1 billion in the next five years. There are many reasons the death penalty should be abolished. It is 

**TextCNN Jittor** positive rank: 3

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (abolished, death, penalty, the, was), which can help lexical baselines.

- rank 1 | msmarco_test_000004_neg_4 | label=0 | score=5.6577 | In 1846, Michigan became the first state to abolish the death penalty for all crimes except treason. Later, Rhode Island and Wisconsin abolished the death penalty for all crimes. By the end of the cen
- rank 2 | msmarco_test_000004_neg_1 | label=0 | score=4.9863 | Why the Death Penalty should be abolished. The risk of executing innocent people exists in any justice system There have been and always will be cases of executions of innocent people. No matter how d
- rank 3 | msmarco_test_000004_pos_3 | label=1 | score=4.8606 | The last executions UK took place in 1964, and the death penalty was abolished in 1998. In 2004 the UK became a party to the 13th Protocol to the European Convention on Human Rights and prohibited the
- rank 4 | msmarco_test_000004_neg_5 | label=0 | score=4.6394 | California currently spends $184 million on the death penalty each year and is on track to spend $1 billion in the next five years. There are many reasons the death penalty should be abolished. It is 
- rank 5 | msmarco_test_000004_neg_2 | label=0 | score=4.2154 | Since 2007, six states have abolished the death penalty: Maryland, Connecticut, Illinois, New Mexico and New Jersey. According to the Death Penalty Information Center, a research group opposed to the 

### msmarco_test_000008

Query: what does a k in a mustang's vin code mean

**TF-IDF** positive rank: 1

The positive passage is ranked first. Keyword overlap is visible (a, code, in, k, mustang, vin), which can help lexical baselines.

- rank 1 | msmarco_test_000008_pos_1 | label=1 | score=0.0638 | The “K” represented the engine code on the VIN number of these Mustangs. The K-Code engine was first introduced by Ford in 1963, and was featured in cars such as the Fairlane and the Comet. Each K-Cod
- rank 2 | msmarco_test_000008_neg_2 | label=0 | score=0.0549 | Answer: Ah, the popular K-code Mustang question. Well, the simple answer is the K-Code was a 1965-1967 Mustang that came from the factory with a special 289 High Performance cubic-inch engine beneath 
- rank 3 | msmarco_test_000008_neg_4 | label=0 | score=0.0512 | So with that said let start with the facts for both years: (I will also identify specific things with each year). 1. The VIN needs to have either an A code or K code in the 5th position. Transmission 
- rank 4 | msmarco_test_000008_neg_5 | label=0 | score=0.0449 | Ford 289 ‘K Code’ engine in 1968 Mustang GT350 (credit: Stephen Foskett). The code refers to the fifth character of the car’s VIN (vehicle identification number), and denotes the engine the car was fi
- rank 5 | msmarco_test_000008_neg_3 | label=0 | score=0.0190 | Foxbody Mustang Vehicle Identification Number. The VIN on any vehicle is a very useful line of characters and numbers. It tells us the year, assembly plant, body type and engine option that the car wa

**BM25** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (a, code, in, k, mustang, s), which can help lexical baselines.

- rank 1 | msmarco_test_000008_neg_5 | label=0 | score=2.5876 | Ford 289 ‘K Code’ engine in 1968 Mustang GT350 (credit: Stephen Foskett). The code refers to the fifth character of the car’s VIN (vehicle identification number), and denotes the engine the car was fi
- rank 2 | msmarco_test_000008_pos_1 | label=1 | score=1.5201 | The “K” represented the engine code on the VIN number of these Mustangs. The K-Code engine was first introduced by Ford in 1963, and was featured in cars such as the Fairlane and the Comet. Each K-Cod
- rank 3 | msmarco_test_000008_neg_2 | label=0 | score=1.4541 | Answer: Ah, the popular K-code Mustang question. Well, the simple answer is the K-Code was a 1965-1967 Mustang that came from the factory with a special 289 High Performance cubic-inch engine beneath 
- rank 4 | msmarco_test_000008_neg_4 | label=0 | score=1.2468 | So with that said let start with the facts for both years: (I will also identify specific things with each year). 1. The VIN needs to have either an A code or K code in the 5th position. Transmission 
- rank 5 | msmarco_test_000008_neg_3 | label=0 | score=0.7794 | Foxbody Mustang Vehicle Identification Number. The VIN on any vehicle is a very useful line of characters and numbers. It tells us the year, assembly plant, body type and engine option that the car wa

**TextCNN Jittor** positive rank: 2

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (a, code, in, k, mustang), which can help lexical baselines.

- rank 1 | msmarco_test_000008_neg_2 | label=0 | score=5.4254 | Answer: Ah, the popular K-code Mustang question. Well, the simple answer is the K-Code was a 1965-1967 Mustang that came from the factory with a special 289 High Performance cubic-inch engine beneath 
- rank 2 | msmarco_test_000008_pos_1 | label=1 | score=5.2202 | The “K” represented the engine code on the VIN number of these Mustangs. The K-Code engine was first introduced by Ford in 1963, and was featured in cars such as the Fairlane and the Comet. Each K-Cod
- rank 3 | msmarco_test_000008_neg_3 | label=0 | score=5.1588 | Foxbody Mustang Vehicle Identification Number. The VIN on any vehicle is a very useful line of characters and numbers. It tells us the year, assembly plant, body type and engine option that the car wa
- rank 4 | msmarco_test_000008_neg_4 | label=0 | score=4.4404 | So with that said let start with the facts for both years: (I will also identify specific things with each year). 1. The VIN needs to have either an A code or K code in the 5th position. Transmission 
- rank 5 | msmarco_test_000008_neg_5 | label=0 | score=4.2247 | Ford 289 ‘K Code’ engine in 1968 Mustang GT350 (credit: Stephen Foskett). The code refers to the fifth character of the car’s VIN (vehicle identification number), and denotes the engine the car was fi

## Failure Cases

### msmarco_test_000002

Query: what does the aortic valve do

**TF-IDF** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (aortic, the, valve), which can help lexical baselines.

- rank 1 | msmarco_test_000002_neg_2 | label=0 | score=0.1928 | The aortic valve is one of the two semilunar valves of the heart, the other being the pulmonary valve. The heart has four valves and the other two are the mitral and the tricuspid valves. When the pre
- rank 2 | msmarco_test_000002_neg_3 | label=0 | score=0.1917 | There are four valves in your heart including the mitral, tricuspid, aortic and pulmonic valves. The aortic valve is located between the left ventricle (lower heart pumping chamber) and the aorta, whi
- rank 3 | msmarco_test_000002_neg_4 | label=0 | score=0.1544 | During ventricular systole, pressure rises in the left ventricle. When the pressure in the left ventricle rises above the pressure in the aorta, the aortic valve opens, allowing blood to exit the left
- rank 4 | msmarco_test_000002_pos_1 | label=1 | score=0.1313 | The aortic valve opens to allow blood to flow into the aorta and then closes to prevent blood from flowing back into the heart. This animation shows the aortic valve in action.
- rank 5 | msmarco_test_000002_neg_5 | label=0 | score=0.1255 | All rights reserved. The aorta is the largest artery in the body. The aorta begins at the top of the left ventricle, the heart's muscular pumping chamber. The heart pumps blood from the left ventricle

**BM25** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (aortic, does, the, valve), which can help lexical baselines.

- rank 1 | msmarco_test_000002_neg_3 | label=0 | score=1.8174 | There are four valves in your heart including the mitral, tricuspid, aortic and pulmonic valves. The aortic valve is located between the left ventricle (lower heart pumping chamber) and the aorta, whi
- rank 2 | msmarco_test_000002_neg_2 | label=0 | score=0.8671 | The aortic valve is one of the two semilunar valves of the heart, the other being the pulmonary valve. The heart has four valves and the other two are the mitral and the tricuspid valves. When the pre
- rank 3 | msmarco_test_000002_neg_4 | label=0 | score=0.8221 | During ventricular systole, pressure rises in the left ventricle. When the pressure in the left ventricle rises above the pressure in the aorta, the aortic valve opens, allowing blood to exit the left
- rank 4 | msmarco_test_000002_pos_1 | label=1 | score=0.8048 | The aortic valve opens to allow blood to flow into the aorta and then closes to prevent blood from flowing back into the heart. This animation shows the aortic valve in action.
- rank 5 | msmarco_test_000002_neg_5 | label=0 | score=0.7189 | All rights reserved. The aorta is the largest artery in the body. The aorta begins at the top of the left ventricle, the heart's muscular pumping chamber. The heart pumps blood from the left ventricle

**TextCNN Jittor** positive rank: 5

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (aortic, does, the, valve), which can help lexical baselines.

- rank 1 | msmarco_test_000002_neg_3 | label=0 | score=4.5243 | There are four valves in your heart including the mitral, tricuspid, aortic and pulmonic valves. The aortic valve is located between the left ventricle (lower heart pumping chamber) and the aorta, whi
- rank 2 | msmarco_test_000002_neg_5 | label=0 | score=4.1690 | All rights reserved. The aorta is the largest artery in the body. The aorta begins at the top of the left ventricle, the heart's muscular pumping chamber. The heart pumps blood from the left ventricle
- rank 3 | msmarco_test_000002_neg_2 | label=0 | score=4.0345 | The aortic valve is one of the two semilunar valves of the heart, the other being the pulmonary valve. The heart has four valves and the other two are the mitral and the tricuspid valves. When the pre
- rank 4 | msmarco_test_000002_neg_4 | label=0 | score=2.2713 | During ventricular systole, pressure rises in the left ventricle. When the pressure in the left ventricle rises above the pressure in the aorta, the aortic valve opens, allowing blood to exit the left
- rank 5 | msmarco_test_000002_pos_1 | label=1 | score=1.8405 | The aortic valve opens to allow blood to flow into the aorta and then closes to prevent blood from flowing back into the heart. This animation shows the aortic valve in action.

### msmarco_test_000005

Query: what types of movements are observed in transport of bedload

**TF-IDF** positive rank: 3

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (in, of, transport), which can help lexical baselines.

- rank 1 | msmarco_test_000005_neg_1 | label=0 | score=0.1008 | Sediment transport due to fluid motion occurs in rivers, oceans, lakes, seas, and other bodies of water due to currents and tides. Transport is also caused by glaciers as they flow, and on terrestrial
- rank 2 | msmarco_test_000005_neg_4 | label=0 | score=0.0750 | Rapids are sections of a river where the gradient of the river bed is relatively steep resulting in an increase in the river’s turbulence and velocity. They form where the gradient of the river is ste
- rank 3 | msmarco_test_000005_pos_3 | label=1 | score=0.0582 | On Fraser River at Mission, for example, about 1 million tonnes of the total 22 million tonnes of total sediment transport moves as bedload (about 5%). Equation (4.2) and others of its type utilize a 
- rank 4 | msmarco_test_000005_neg_2 | label=0 | score=0.0543 | Sediment transport is the movement of organic and inorganic particles by water 10. In general, the greater the flow, the more sediment that will be conveyed. Water flow can be strong enough to suspend
- rank 5 | msmarco_test_000005_neg_5 | label=0 | score=0.0367 | Sediment is a naturally occurring material that is broken down by processes of weathering and erosion, and is subsequently transported by the action of wind, water, or ice, and/or by the force of grav

**BM25** positive rank: 3

The positive passage appears in the middle ranks. Hard negatives share enough words or answer style to compete with the positive passage. Keyword overlap is visible (are, bedload, in, of), which can help lexical baselines.

- rank 1 | msmarco_test_000005_neg_4 | label=0 | score=2.5859 | Rapids are sections of a river where the gradient of the river bed is relatively steep resulting in an increase in the river’s turbulence and velocity. They form where the gradient of the river is ste
- rank 2 | msmarco_test_000005_neg_1 | label=0 | score=1.5876 | Sediment transport due to fluid motion occurs in rivers, oceans, lakes, seas, and other bodies of water due to currents and tides. Transport is also caused by glaciers as they flow, and on terrestrial
- rank 3 | msmarco_test_000005_pos_3 | label=1 | score=1.5093 | On Fraser River at Mission, for example, about 1 million tonnes of the total 22 million tonnes of total sediment transport moves as bedload (about 5%). Equation (4.2) and others of its type utilize a 
- rank 4 | msmarco_test_000005_neg_2 | label=0 | score=1.3176 | Sediment transport is the movement of organic and inorganic particles by water 10. In general, the greater the flow, the more sediment that will be conveyed. Water flow can be strong enough to suspend
- rank 5 | msmarco_test_000005_neg_5 | label=0 | score=0.7837 | Sediment is a naturally occurring material that is broken down by processes of weathering and erosion, and is subsequently transported by the action of wind, water, or ice, and/or by the force of grav

**TextCNN Jittor** positive rank: 4

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (in, of, transport), which can help lexical baselines.

- rank 1 | msmarco_test_000005_neg_2 | label=0 | score=5.9333 | Sediment transport is the movement of organic and inorganic particles by water 10. In general, the greater the flow, the more sediment that will be conveyed. Water flow can be strong enough to suspend
- rank 2 | msmarco_test_000005_neg_5 | label=0 | score=5.3210 | Sediment is a naturally occurring material that is broken down by processes of weathering and erosion, and is subsequently transported by the action of wind, water, or ice, and/or by the force of grav
- rank 3 | msmarco_test_000005_neg_1 | label=0 | score=5.2329 | Sediment transport due to fluid motion occurs in rivers, oceans, lakes, seas, and other bodies of water due to currents and tides. Transport is also caused by glaciers as they flow, and on terrestrial
- rank 4 | msmarco_test_000005_pos_3 | label=1 | score=4.2631 | On Fraser River at Mission, for example, about 1 million tonnes of the total 22 million tonnes of total sediment transport moves as bedload (about 5%). Equation (4.2) and others of its type utilize a 
- rank 5 | msmarco_test_000005_neg_4 | label=0 | score=3.9516 | Rapids are sections of a river where the gradient of the river bed is relatively steep resulting in an increase in the river’s turbulence and velocity. They form where the gradient of the river is ste

### msmarco_test_000006

Query: what is the average bookkeeper hourly rate

**TF-IDF** positive rank: 5

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (average, bookkeeper, hourly, is, the), which can help lexical baselines.

- rank 1 | msmarco_test_000006_neg_5 | label=0 | score=0.0926 | Hourly Wages. The average hourly wage for bookkeepers is $16.71, according to 2009 data from the Bureau of Labor Statistics. This wage translates to an annual salary of $34,750 for those working full-
- rank 2 | msmarco_test_000006_neg_3 | label=0 | score=0.0771 | For example the median expected hourly pay for a typical Bookkeeper in the United States is $18 an hour, so 50% of the people who perform the job of Bookkeeper in the United States are expected to mak
- rank 3 | msmarco_test_000006_neg_4 | label=0 | score=0.0756 | Bookkeeping service fee charged by the hour. Some bookkeepers charge out their bookkeeping service at a basic hourly fee with a set minimum charge. The average rate for general bookkeeping services ri
- rank 4 | msmarco_test_000006_neg_2 | label=0 | score=0.0575 | 1 A typical salary for an in-house bookkeeper in the United States runs $30,400-$39,898 according to Salary.com. 2  The average annual pay for a bookkeeping account executive is $68,294, or $57,600 fo
- rank 5 | msmarco_test_000006_pos_1 | label=1 | score=0.0528 | A bookkeeper usually earns an hourly rate between $15 and $20. Entry level bookkeepers can expect starting pay between $12 and $15 an hour while senior bookkeepers can expect to earn a little more tha

**BM25** positive rank: 5

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (average, bookkeeper, hourly, is, the), which can help lexical baselines.

- rank 1 | msmarco_test_000006_neg_5 | label=0 | score=1.4597 | Hourly Wages. The average hourly wage for bookkeepers is $16.71, according to 2009 data from the Bureau of Labor Statistics. This wage translates to an annual salary of $34,750 for those working full-
- rank 2 | msmarco_test_000006_neg_4 | label=0 | score=1.3072 | Bookkeeping service fee charged by the hour. Some bookkeepers charge out their bookkeeping service at a basic hourly fee with a set minimum charge. The average rate for general bookkeeping services ri
- rank 3 | msmarco_test_000006_neg_3 | label=0 | score=1.1664 | For example the median expected hourly pay for a typical Bookkeeper in the United States is $18 an hour, so 50% of the people who perform the job of Bookkeeper in the United States are expected to mak
- rank 4 | msmarco_test_000006_neg_2 | label=0 | score=1.0013 | 1 A typical salary for an in-house bookkeeper in the United States runs $30,400-$39,898 according to Salary.com. 2  The average annual pay for a bookkeeping account executive is $68,294, or $57,600 fo
- rank 5 | msmarco_test_000006_pos_1 | label=1 | score=0.8139 | A bookkeeper usually earns an hourly rate between $15 and $20. Entry level bookkeepers can expect starting pay between $12 and $15 an hour while senior bookkeepers can expect to earn a little more tha

**TextCNN Jittor** positive rank: 5

The positive passage is ranked low. This suggests the lightweight reranker is still sensitive to lexical shortcuts and lacks the deeper semantic judgment of pretrained encoder or LLM-style rerankers. Keyword overlap is visible (bookkeeper, hourly, is, the), which can help lexical baselines.

- rank 1 | msmarco_test_000006_neg_3 | label=0 | score=6.2356 | For example the median expected hourly pay for a typical Bookkeeper in the United States is $18 an hour, so 50% of the people who perform the job of Bookkeeper in the United States are expected to mak
- rank 2 | msmarco_test_000006_neg_4 | label=0 | score=6.0169 | Bookkeeping service fee charged by the hour. Some bookkeepers charge out their bookkeeping service at a basic hourly fee with a set minimum charge. The average rate for general bookkeeping services ri
- rank 3 | msmarco_test_000006_neg_5 | label=0 | score=5.9516 | Hourly Wages. The average hourly wage for bookkeepers is $16.71, according to 2009 data from the Bureau of Labor Statistics. This wage translates to an annual salary of $34,750 for those working full-
- rank 4 | msmarco_test_000006_neg_2 | label=0 | score=4.6501 | 1 A typical salary for an in-house bookkeeper in the United States runs $30,400-$39,898 according to Salary.com. 2  The average annual pay for a bookkeeping account executive is $68,294, or $57,600 fo
- rank 5 | msmarco_test_000006_pos_1 | label=1 | score=3.4570 | A bookkeeper usually earns an hourly rate between $15 and $20. Entry level bookkeepers can expect starting pay between $12 and $15 an hour while senior bookkeepers can expect to earn a little more tha
