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


def exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def load_torch_metrics() -> dict:
    metrics_path = ROOT / "outputs" / "torch_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def status_for(paths: list[str]) -> str:
    return "ready" if all(exists(path) for path in paths) else "pending"


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
    print(f"PyTorch baseline: {'ready' if exists('outputs/torch_metrics.json') and exists('outputs/torch_model.pt') else 'missing'}")
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

    if missing:
        print("\nMissing required files:")
        for relative_path in missing:
            print(f"- {relative_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
