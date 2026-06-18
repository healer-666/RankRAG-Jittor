"""Generate synthetic ranking data with easy and hard negatives."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_PATH = ROOT / "data" / "academic_demo.json"


TOPICS = [
    {
        "name": "non-stationary reinforcement learning",
        "query": "How is UCB used in non-stationary reinforcement learning?",
        "positive": "Value-based reinforcement learning can use UCB-style optimism by adding confidence bonuses to action values and updating those bonuses as non-stationary dynamics shift over time.",
        "keywords": ["UCB", "exploration", "non-stationary", "value-based", "reinforcement learning"],
        "hard_negatives": [
            "UCB algorithms choose bandit arms with upper confidence bounds and are often used as a foundation for exploration under uncertainty.",
            "Policy gradient reinforcement learning can encourage exploration with entropy bonuses and stochastic policies during online training.",
            "Value-based reinforcement learning in stationary Markov decision processes often balances exploration and exploitation through optimistic initial values.",
            "Exploration bonuses in deep reinforcement learning provide additional rewards for uncertain or novel transitions across many algorithms.",
            "Non-stationary reinforcement learning benchmarks evaluate adaptation when transition dynamics and rewards change over time.",
        ],
        "eval_hard_negatives": [
            "A non-stationary reinforcement learning benchmark paper compares value-based agents, UCB exploration, and reward drift as background factors.",
            "An exploration study applies UCB-style confidence intervals to reinforcement learning experiments and reports value estimates across changing tasks.",
            "A value-based reinforcement learning ablation uses non-stationary environments and UCB-inspired exploration bonuses as one experimental setting.",
            "A survey section connects UCB, non-stationary reinforcement learning, value functions, and exploration bonuses across several research threads.",
        ],
    },
    {
        "name": "retrieval augmented generation",
        "query": "How can context ranking improve retrieval augmented generation?",
        "positive": "Context ranking improves retrieval augmented generation by scoring retrieved passages, selecting the most relevant evidence, and passing the best contexts to answer generation.",
        "keywords": ["ranking", "retrieval", "context", "generation", "RAG"],
        "hard_negatives": [
            "Dense retrieval for RAG maps questions and documents into an embedding space and returns passages with high vector similarity.",
            "Answer generation prompts for RAG combine a user question with retrieved context and instruct the language model to produce a response.",
            "Document ranking for web search estimates relevance between a query and candidate pages using lexical and neural features.",
            "A survey of RAG systems compares retrieval, context windows, ranking modules, and generation metrics across benchmarks.",
            "Long-context generation evaluates how language models use many input passages when producing an answer.",
        ],
        "eval_hard_negatives": [
            "A RAG benchmark reports retrieved context ranking scores beside generation quality metrics for several question answering datasets.",
            "A retrieval augmented generation system uses passage scores, context windows, and answer generation prompts in a single pipeline.",
            "A neural ranking model orders retrieved documents and compares its outputs with RAG answer generation results.",
            "A long-context RAG analysis discusses context ranking, retrieval recall, and generated answer faithfulness as evaluation dimensions.",
        ],
    },
    {
        "name": "academic paper recommendation",
        "query": "How should academic search systems rank candidate paper abstracts?",
        "positive": "Academic search systems can rank candidate paper abstracts by computing query-abstract semantic matching features and scoring each candidate for relevance.",
        "keywords": ["academic search", "paper", "abstract", "ranking", "semantic"],
        "hard_negatives": [
            "Academic collaborator recommendation uses citation graphs, publication venues, and author embeddings to suggest related researchers.",
            "Semantic embedding of paper abstracts maps titles and abstracts into vector space for clustering and nearest-neighbor lookup.",
            "General search ranking combines lexical matching, semantic similarity, and click feedback to order candidate documents.",
            "Paper recommendation datasets collect queries, abstracts, citations, and relevance judgments for evaluating academic search.",
            "Topic clustering groups paper abstracts by semantic themes and helps users browse research areas.",
        ],
        "eval_hard_negatives": [
            "An academic search benchmark stores paper abstracts, semantic query features, and ranking labels for offline experiments.",
            "A paper recommendation system scores candidate abstracts using citation overlap, semantic similarity, and venue priors.",
            "A scholarly retrieval model embeds queries and abstracts and displays ranked paper candidates in an academic search interface.",
            "A survey of academic ranking datasets compares abstract matching, citation recommendation, and semantic search tasks.",
        ],
    },
    {
        "name": "uncertainty estimation",
        "query": "Why are uncertainty estimates useful for exploration in value learning?",
        "positive": "Uncertainty estimates are useful in value learning because they create exploration bonuses for actions whose value estimates are uncertain, improving state-action coverage.",
        "keywords": ["uncertainty", "bonus", "exploration", "value", "learning"],
        "hard_negatives": [
            "Uncertainty estimation for supervised calibration measures whether predicted probabilities match observed frequencies.",
            "Exploration schedules in reinforcement learning adjust random action rates during training and can improve coverage.",
            "Value learning objectives minimize temporal-difference errors between predicted values and bootstrapped targets.",
            "A survey of uncertainty, exploration, and learning reviews Bayesian models, ensembles, bonuses, and intrinsic motivation.",
            "Intrinsic reward bonuses for novelty encourage agents to visit unfamiliar states during reinforcement learning.",
        ],
        "eval_hard_negatives": [
            "A reinforcement learning paper compares uncertainty estimates, exploration bonuses, value learning, and intrinsic rewards across tasks.",
            "A value learning benchmark reports how uncertainty signals correlate with exploration coverage during training.",
            "A survey chapter links uncertainty estimation, exploration bonuses, value functions, and optimistic learning algorithms.",
            "An exploration method combines novelty bonuses with value learning updates and evaluates uncertain state-action regions.",
        ],
    },
]


EASY_NEGATIVE_POOL = [
    "An unrelated document about large-scale image classification with convolutional networks.",
    "A tutorial on general deep learning optimization and batch normalization.",
    "An article about database indexing for transaction processing systems.",
    "A study of language model tokenization for multilingual text.",
    "A report on climate forecasting with remote sensing imagery.",
    "A document about recommender system matrix factorization for movie ratings.",
    "A note about classical graph algorithms and shortest path search.",
    "A finance report about portfolio allocation and risk control.",
]


def make_record(split: str, index: int, rng: random.Random) -> dict:
    """Create one query with 1 positive, 2 hard negatives, and 2 easy negatives."""

    topic = rng.choice(TOPICS)
    query_id = f"{split}_q_{index:04d}"
    positive_text = topic["positive"]

    candidates = [
        {
            "doc_id": f"{query_id}_pos",
            "text": positive_text,
            "label": 1,
        }
    ]

    hard_pool_name = "hard_negatives" if split == "train" else "eval_hard_negatives"
    for neg_idx, text in enumerate(rng.sample(topic[hard_pool_name], 2), start=1):
        candidates.append(
            {
                "doc_id": f"{query_id}_hard_neg_{neg_idx}",
                "text": text,
                "label": 0,
            }
        )

    for neg_idx, text in enumerate(rng.sample(EASY_NEGATIVE_POOL, 2), start=1):
        candidates.append(
            {
                "doc_id": f"{query_id}_easy_neg_{neg_idx}",
                "text": text,
                "label": 0,
            }
        )

    rng.shuffle(candidates)
    return {"query_id": query_id, "query": topic["query"], "candidates": candidates}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_demo() -> None:
    """Write a human-readable academic search demo with explicit hard negatives."""

    demo = {
        "query_id": "academic_demo_001",
        "query": "非平稳强化学习中，value-based 方法如何使用 UCB 思想进行探索？",
        "candidates": [
            {
                "doc_id": "paper_ucb_nonstationary_value_rl",
                "title": "UCB Bonuses for Non-Stationary Value-Based Reinforcement Learning",
                "text": "This paper studies value-based reinforcement learning in non-stationary environments and uses UCB-style exploration bonuses to balance uncertainty and reward estimation.",
                "label": 1,
            },
            {
                "doc_id": "hard_ucb_bandit_foundation",
                "title": "Finite-time Analysis of the Multiarmed Bandit Problem",
                "text": "A foundational UCB bandit paper that introduces upper confidence bounds for exploration under uncertainty.",
                "label": 0,
            },
            {
                "doc_id": "hard_policy_gradient_exploration",
                "title": "Policy Gradient Exploration with Entropy Regularization",
                "text": "This work discusses exploration in reinforcement learning through policy gradient entropy objectives and stochastic policy updates.",
                "label": 0,
            },
            {
                "doc_id": "hard_stationary_value_rl",
                "title": "Value-Based Reinforcement Learning in Stationary MDPs",
                "text": "This paper studies value-based reinforcement learning and exploration in stationary Markov decision processes.",
                "label": 0,
            },
            {
                "doc_id": "hard_general_exploration_bonus",
                "title": "A Survey of Exploration Bonuses in Deep RL",
                "text": "This survey reviews exploration bonuses, uncertainty, UCB-inspired optimism, and deep reinforcement learning.",
                "label": 0,
            },
            {
                "doc_id": "easy_image_classification",
                "title": "Large-Scale Image Classification with Convolutional Networks",
                "text": "This paper focuses on supervised visual recognition and image classification benchmarks.",
                "label": 0,
            },
            {
                "doc_id": "easy_database_indexing",
                "title": "Database Indexing for Transaction Processing",
                "text": "This article discusses B-tree and LSM-tree indexing strategies for transactional database systems.",
                "label": 0,
            },
        ],
    }
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_PATH.write_text(json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(42)
    split_sizes = {"train": 200, "valid": 50, "test": 50}

    for split, size in split_sizes.items():
        records = [make_record(split, index, rng) for index in range(size)]
        write_jsonl(PROCESSED_DIR / f"{split}.jsonl", records)

    write_demo()
    print(f"Wrote synthetic data with hard negatives to {PROCESSED_DIR}")
    print(f"Wrote academic demo to {DEMO_PATH}")


if __name__ == "__main__":
    main()
