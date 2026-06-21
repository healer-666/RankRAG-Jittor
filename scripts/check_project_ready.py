"""Check whether the lightweight RankRAG reproduction is ready to present."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "docs/method_summary.md",
    "docs/jittor_setup.md",
    "configs/default.yaml",
    "src/model_torch.py",
    "src/model_jittor.py",
    "src/train_torch.py",
    "src/train_jittor.py",
    "src/eval_torch.py",
    "src/eval_jittor.py",
    "outputs/loss_curve.png",
    "outputs/metrics_compare.png",
    "outputs/torch_metrics.json",
    "outputs/demo_ranking_result_torch.json",
]

MSMARCO_FILES = {
    "MS MARCO data": [
        "data/processed/msmarco/train.jsonl",
        "data/processed/msmarco/valid.jsonl",
        "data/processed/msmarco/test.jsonl",
    ],
    "MS MARCO PyTorch": [
        "outputs/msmarco_torch_metrics.json",
    ],
    "MS MARCO Jittor": [
        "outputs/msmarco_jittor_metrics.json",
    ],
    "MS MARCO visualization": [
        "outputs/msmarco_metrics_compare.md",
        "outputs/msmarco_loss_curve.png",
        "outputs/msmarco_metrics_compare.png",
    ],
}

L2_FILES = {
    "Retrieval baselines": [
        "outputs/msmarco_retrieval_baseline_metrics.json",
        "outputs/msmarco_retrieval_baseline_rankings.json",
    ],
    "TextCNN PyTorch": [
        "outputs/msmarco_textcnn_torch_metrics.json",
        "logs/msmarco_textcnn_torch_train.log",
    ],
    "TextCNN Jittor": [
        "outputs/msmarco_textcnn_jittor_metrics.json",
        "logs/msmarco_textcnn_jittor_train.log",
    ],
    "L2 aggregate results": [
        "outputs/l2_msmarco_results.md",
        "outputs/l2_msmarco_results.png",
    ],
    "Case study": [
        "docs/msmarco_case_study.md",
        "outputs/msmarco_case_study.json",
    ],
    "Hardware report": [
        "docs/hardware_report.md",
    ],
}

MSMARCO_MEDIUM_FILES = {
    "MS MARCO medium data": [
        "data/processed/msmarco_medium/train.jsonl",
        "data/processed/msmarco_medium/valid.jsonl",
        "data/processed/msmarco_medium/test.jsonl",
    ],
    "MS MARCO medium retrieval baselines": [
        "outputs/msmarco_medium_retrieval_baseline_metrics.json",
    ],
    "MS MARCO medium MLP PyTorch/Jittor": [
        "outputs/msmarco_medium_torch_metrics.json",
        "outputs/msmarco_medium_jittor_metrics.json",
    ],
    "MS MARCO medium TextCNN PyTorch/Jittor": [
        "outputs/msmarco_medium_textcnn_torch_metrics.json",
        "outputs/msmarco_medium_textcnn_jittor_metrics.json",
    ],
    "MS MARCO medium aggregate results": [
        "outputs/l2_msmarco_medium_results.md",
        "outputs/l2_msmarco_medium_results.png",
    ],
    "MS MARCO medium case study": [
        "docs/msmarco_medium_case_study.md",
        "outputs/msmarco_medium_case_study.json",
    ],
}

L25_FILES = {
    "External Cross-Encoder reference": [
        "outputs/msmarco_medium_cross_encoder_metrics.json",
        "outputs/msmarco_medium_cross_encoder_rankings.json",
    ],
    "L2.5 aggregate results": [
        "outputs/l25_msmarco_medium_results.md",
        "outputs/l25_msmarco_medium_results.png",
    ],
    "Cross-Encoder case study": [
        "docs/msmarco_medium_cross_encoder_case_study.md",
    ],
}

LORA_DEBUG_FILES = {
    "LoRA reranker plan": [
        "docs/lora_reranker_plan.md",
        "docs/lora_debug_report.md",
    ],
    "LoRA debug data": [
        "data/processed/lora_debug/train_pairs.jsonl",
        "data/processed/lora_debug/valid_pairs.jsonl",
        "data/processed/lora_debug/test_queries.jsonl",
    ],
    "LoRA debug evaluation": [
        "outputs/lora_debug/qwen_0_5b_lora_metrics.json",
        "outputs/lora_debug/qwen_0_5b_lora_rankings.json",
    ],
}

LORA_FORMAL_FILES = {
    "LoRA 1.5B configs": [
        "configs/lora_qwen_1_5b_formal.yaml",
        "configs/lora_qwen_1_5b_lr5e5_s500.yaml",
        "configs/lora_qwen_1_5b_10k_lr1e4_s800.yaml",
    ],
    "LoRA 1.5B data cards": [
        "data/processed/lora_qwen_1_5b_formal/data_card.md",
        "data/processed/lora_qwen_1_5b_10k/data_card.md",
    ],
    "LoRA 1.5B formal v3 result": [
        "outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_metrics.json",
        "outputs/lora_qwen_1_5b_10k_lr1e4_s800/train_summary.json",
        "outputs/lora_qwen_1_5b_10k_lr1e4_s800/loss_curve.png",
    ],
    "LoRA 1.5B tuning summaries": [
        "outputs/lora_qwen_1_5b_formal/qwen_1_5b_lora_metrics.json",
        "outputs/lora_qwen_1_5b_lr5e5_s500/qwen_1_5b_lora_metrics.json",
        "outputs/lora_qwen2_1_5b_tuning_summary.md",
    ],
    "LoRA 1.5B comparison": [
        "outputs/lora_qwen2_1_5b_comparison.json",
        "outputs/lora_qwen2_1_5b_comparison.md",
        "docs/lora_qwen2_1_5b_results.md",
    ],
}

DOWNSTREAM_RAG_FILES = {
    "Downstream RAG config and subset": [
        "configs/downstream_rag_50q.yaml",
        "data/processed/msmarco_downstream_qa_50.jsonl",
        "outputs/downstream_rag_eval/qa_subset_manifest.json",
    ],
    "Downstream RAG generated answers": [
        "outputs/downstream_rag_eval/bm25_top3_answers.jsonl",
        "outputs/downstream_rag_eval/lora_v3_top3_answers.jsonl",
        "outputs/downstream_rag_eval/cross_encoder_top3_answers.jsonl",
    ],
    "Downstream RAG aggregate results": [
        "outputs/downstream_rag_eval/downstream_rag_eval_results.json",
        "outputs/downstream_rag_eval/downstream_rag_eval_results.md",
        "outputs/downstream_rag_eval/downstream_rag_eval_results.png",
    ],
    "Downstream RAG validation and docs": [
        "outputs/downstream_rag_eval/validation.json",
        "outputs/downstream_rag_eval/downstream_rag_case_study.json",
        "docs/downstream_rag_data_audit.md",
        "docs/downstream_rag_subset_report.md",
        "docs/downstream_rag_analysis.md",
    ],
}

JITTORLLM_ZEROSHOT_FILES = [
    "src/jittorllm_reranker/evaluate_jittorllm_zeroshot.py",
    "src/jittorllm_reranker/prompt_utils.py",
    "configs/jittorllm_zeroshot_medium.yaml",
    "scripts/run_jittorllm_zeroshot_smoke.sh",
    "scripts/run_jittorllm_zeroshot_medium.sh",
    "docs/jittorllm_zeroshot_report.md",
]

JITTORLLM_QWEN2_FILES = [
    "src/jittorllm_reranker/backend_qwen2_jittor.py",
    "src/jittorllm_reranker/evaluate_qwen2_jittor.py",
    "scripts/smoke_jittorllms_qwen2.py",
    "scripts/patch_jittorllms_qwen2_fp32_attention.py",
    "scripts/run_jittorllm_qwen2_0_5b_smoke.sh",
    "scripts/run_jittorllm_qwen2_0_5b_20q.sh",
    "configs/jittorllm_qwen2_0_5b_smoke.yaml",
    "configs/jittorllm_qwen2_0_5b_20q.yaml",
    "configs/jittorllm_qwen2_1_5b_smoke.yaml",
    "configs/jittorllm_qwen2_1_5b_full.yaml",
    "docs/qwen2_1_5b_fp32_attention.patch",
    "outputs/jittorllm_qwen2_1_5b_full/metrics.json",
    "outputs/jittorllm_qwen2_1_5b_full/validation.json",
]


def exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def load_torch_metrics() -> dict:
    metrics_path = ROOT / "outputs" / "torch_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def status_for(paths: list[str]) -> str:
    return "ready" if all(exists(path) for path in paths) else "pending"


def jittorllm_status() -> str:
    if not all(exists(path) for path in JITTORLLM_ZEROSHOT_FILES):
        return "missing"
    if all(exists(path) for path in JITTORLLM_QWEN2_FILES):
        qwen_metrics = [
            ROOT / "outputs" / "jittorllm_qwen2_0_5b_smoke" / "metrics.json",
            ROOT / "outputs" / "jittorllm_qwen2_0_5b_20q" / "metrics.json",
        ]
        if all(path.exists() for path in qwen_metrics):
            try:
                statuses = [
                    json.loads(path.read_text(encoding="utf-8")).get("status")
                    for path in qwen_metrics
                ]
            except Exception:
                return "partial"
            if all(status == "ready" for status in statuses):
                return "ready"
    metrics_path = ROOT / "outputs" / "jittorllm_zeroshot_medium" / "smoke_metrics.json"
    if not metrics_path.exists():
        return "partial"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return "partial"
    status = payload.get("status")
    if status == "ready":
        return "ready"
    if status == "blocked":
        return "blocked"
    return "partial"


def lora_formal_status() -> str:
    return "ready" if all(status_for(paths) == "ready" for paths in LORA_FORMAL_FILES.values()) else "pending"


def downstream_rag_status() -> str:
    return "ready" if all(status_for(paths) == "ready" for paths in DOWNSTREAM_RAG_FILES.values()) else "pending"


def main() -> None:
    print("Project readiness check")
    print("=" * 24)

    missing = []
    for relative_path in REQUIRED_FILES:
        ok = exists(relative_path)
        print(f"{'[OK]' if ok else '[MISSING]'} {relative_path}")
        if not ok:
            missing.append(relative_path)

    jittor_metrics_ready = exists("outputs/jittor_metrics.json")
    if jittor_metrics_ready:
        print("[OK] outputs/jittor_metrics.json")
    else:
        print("[PENDING] outputs/jittor_metrics.json - Jittor result pending")

    metrics = load_torch_metrics()
    if metrics:
        print("\nPyTorch metrics")
        for name in ["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr", "pairwise_accuracy"]:
            if name in metrics:
                print(f"- {name}: {metrics[name]:.4f}")

    print("\nProject status summary")
    torch_ready = all(
        exists(path)
        for path in [
            "outputs/torch_metrics.json",
            "outputs/demo_ranking_result_torch.json",
            "logs/torch_train.log",
        ]
    )
    print(f"PyTorch baseline: {'ready' if torch_ready else 'missing'}")
    print("Jittor skeleton: ready")
    print(f"Jittor training: {'ready' if jittor_metrics_ready else 'pending'}")
    print(f"Visualization: {'ready' if exists('outputs/loss_curve.png') and exists('outputs/metrics_compare.png') else 'missing'}")
    print(f"README: {'ready' if exists('README.md') else 'missing'}")

    print("\nMS MARCO status summary")
    for name, paths in MSMARCO_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nL2 status summary")
    for name, paths in L2_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nMS MARCO medium status summary")
    for name, paths in MSMARCO_MEDIUM_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nL2.5 external reranker status summary")
    for name, paths in L25_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nLoRA debug status summary")
    lora_ready = all(status_for(paths) == "ready" for paths in LORA_DEBUG_FILES.values())
    print(f"LoRA reranker debug: {'ready' if lora_ready else 'pending'}")
    for name, paths in LORA_DEBUG_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nL3 LoRA formal status summary")
    print(f"Qwen2.5-1.5B LoRA formal: {lora_formal_status()}")
    for name, paths in LORA_FORMAL_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nStage D downstream RAG status summary")
    print(f"Downstream RAG answer generation: {downstream_rag_status()}")
    for name, paths in DOWNSTREAM_RAG_FILES.items():
        print(f"{name}: {status_for(paths)}")

    print("\nJittorLLM zero-shot status summary")
    print(f"JittorLLM zero-shot: {jittorllm_status()}")

    if missing:
        print("\nMissing required files:")
        for relative_path in missing:
            print(f"- {relative_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
