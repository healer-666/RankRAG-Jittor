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


def exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def load_torch_metrics() -> dict:
    metrics_path = ROOT / "outputs" / "torch_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


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

    if missing:
        print("\nMissing required files:")
        for relative_path in missing:
            print(f"- {relative_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
