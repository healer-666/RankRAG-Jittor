"""Aggregate formal E1 LoRA data-size ablation outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


RUNS = [
    {
        "experiment_name": "1k",
        "training_pairs": 1000,
        "effective_epochs": 6.4,
        "output_dir": Path("outputs/lora_ablation_data_1k"),
        "adapter_dir": Path("outputs/lora_adapters/qwen_1_5b_1k_lr1e4_s800"),
        "gpu_csv": Path("logs/e1_autodl_4090d/1k_gpu.csv"),
    },
    {
        "experiment_name": "3k",
        "training_pairs": 3000,
        "effective_epochs": 2.1333333333333333,
        "output_dir": Path("outputs/lora_ablation_data_3k"),
        "adapter_dir": Path("outputs/lora_adapters/qwen_1_5b_3k_lr1e4_s800"),
        "gpu_csv": Path("logs/e1_autodl_4090d/3k_gpu.csv"),
    },
    {
        "experiment_name": "10k-rerun",
        "training_pairs": 10000,
        "effective_epochs": 0.64,
        "output_dir": Path("outputs/lora_ablation_data_10k_rerun"),
        "adapter_dir": Path("outputs/lora_adapters/qwen_1_5b_10k_lr1e4_s800_rerun"),
        "gpu_csv": Path("logs/e1_autodl_4090d/10k_rerun_gpu.csv"),
    },
]

HISTORICAL_10K = Path("outputs/lora_qwen_1_5b_10k_lr1e4_s800")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_train_log_last_step(path: Path) -> int | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return None if last is None else int(last.get("step", 0))


def path_text(path: Path) -> str:
    return path.as_posix()


def gpu_stats(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"avg_gpu_utilization": None, "max_gpu_temperature": None, "avg_power_draw": None, "max_power_draw": None}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return {"avg_gpu_utilization": None, "max_gpu_temperature": None, "avg_power_draw": None, "max_power_draw": None}

    def values(key: str) -> list[float]:
        out = []
        for row in rows:
            text = str(row.get(key, "")).strip()
            if text and text != "[N/A]":
                try:
                    out.append(float(text))
                except ValueError:
                    pass
        return out

    util = values("utilization.gpu [%]")
    temp = values("temperature.gpu")
    power = values("power.draw [W]")
    return {
        "avg_gpu_utilization": sum(util) / len(util) if util else None,
        "max_gpu_temperature": max(temp) if temp else None,
        "avg_power_draw": sum(power) / len(power) if power else None,
        "max_power_draw": max(power) if power else None,
    }


def normalize_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}recall_at_1": metrics.get("recall@1"),
        f"{prefix}recall_at_3": metrics.get("recall@3"),
        f"{prefix}recall_at_5": metrics.get("recall@5"),
        f"{prefix}ndcg_at_1": metrics.get("ndcg@1"),
        f"{prefix}ndcg_at_3": metrics.get("ndcg@3"),
        f"{prefix}ndcg_at_5": metrics.get("ndcg@5"),
        f"{prefix}mrr": metrics.get("mrr"),
        f"{prefix}pairwise_accuracy": metrics.get("pairwise_accuracy"),
    }


def build_row(run: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    output_dir = run["output_dir"]
    summary_path = output_dir / "train_summary.json"
    metrics_path = output_dir / "qwen_1_5b_lora_metrics.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    last_step = read_train_log_last_step(output_dir / "train_log.jsonl")
    complete = bool(summary and metrics and summary.get("steps") == 800 and last_step == 800)
    row = {
        "experiment_name": run["experiment_name"],
        "training_pairs": run["training_pairs"],
        "effective_epochs": run["effective_epochs"],
        "max_steps": 800,
        "per_device_train_batch_size": summary.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": summary.get("gradient_accumulation_steps"),
        "global_batch_size": summary.get("global_batch_size"),
        "seed": 42,
        "git_commit": summary.get("git_commit") or metrics.get("git_commit") or env.get("git_commit"),
        "hardware": summary.get("device_name") or metrics.get("device_name") or env.get("gpu_name"),
        "model_path": summary.get("model_load_name") or metrics.get("model_load_name") or env.get("model_path"),
        **normalize_metrics(metrics),
        "final_train_loss": summary.get("loss_end"),
        "validation_metric": summary.get("valid_loss_end"),
        "train_runtime_sec": summary.get("runtime_sec"),
        "eval_runtime_sec": metrics.get("inference_time_sec"),
        "peak_gpu_memory_mib": (summary.get("peak_gpu_memory_gib") or 0) * 1024 if summary.get("peak_gpu_memory_gib") is not None else None,
        "output_dir": path_text(output_dir),
        "adapter_dir": path_text(run["adapter_dir"]),
        "last_step": last_step,
        "status": "ready" if complete else "failed",
        **gpu_stats(run["gpu_csv"]),
    }
    return row


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    if any(row["status"] != "ready" for row in rows):
        errors.append("not all E1 runs are complete ready results")
    commits = {row.get("git_commit") for row in rows}
    if len(commits) != 1:
        errors.append(f"E1 runs do not share one commit: {sorted(commits)}")
    hardware = {row.get("hardware") for row in rows}
    if len(hardware) != 1:
        errors.append(f"E1 runs do not share one hardware value: {sorted(hardware)}")
    for row in rows:
        if row.get("last_step") != 800:
            errors.append(f"{row['experiment_name']}: last step is not 800")
        for key in ["recall_at_1", "recall_at_3", "recall_at_5", "ndcg_at_5", "mrr", "pairwise_accuracy"]:
            value = row.get(key)
            if value is None or not (0.0 <= float(value) <= 1.0):
                errors.append(f"{row['experiment_name']}: invalid {key}={value}")
    return errors


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    return "" if value is None else str(value)


def write_markdown(rows: list[dict[str, Any]], errors: list[str], historical: dict[str, Any] | None) -> None:
    lines = [
        "# LoRA Data-Size Ablation Results",
        "",
        "Status: " + ("ready" if not errors else "failed"),
        "",
        "| Run | Pairs | Micro batch | Grad accum | Global batch | Eff. epochs | R@1 | R@3 | R@5 | NDCG@5 | MRR | Pairwise Acc. | Train s | Eval s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {row['training_pairs']} | "
            f"{fmt(row.get('per_device_train_batch_size'))} | {fmt(row.get('gradient_accumulation_steps'))} | "
            f"{fmt(row.get('global_batch_size'))} | {fmt(row.get('effective_epochs'))} | "
            f"{fmt(row.get('recall_at_1'))} | {fmt(row.get('recall_at_3'))} | {fmt(row.get('recall_at_5'))} | "
            f"{fmt(row.get('ndcg_at_5'))} | {fmt(row.get('mrr'))} | {fmt(row.get('pairwise_accuracy'))} | "
            f"{fmt(row.get('train_runtime_sec'))} | {fmt(row.get('eval_runtime_sec'))} |"
        )
    if historical:
        lines += [
            "",
            "Historical 10k reference, not part of the E1 trend line:",
            "",
            f"- R@1: {fmt(historical.get('recall@1'))}; R@5: {fmt(historical.get('recall@5'))}; "
            f"NDCG@5: {fmt(historical.get('ndcg@5'))}; MRR: {fmt(historical.get('mrr'))}",
        ]
    if errors:
        lines += ["", "## Validation Errors", ""]
        lines += [f"- {error}" for error in errors]
    Path("outputs/lora_ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["experiment_name"] for row in rows]
    metrics = [("R@1", "recall_at_1"), ("MRR", "mrr"), ("NDCG@5", "ndcg_at_5")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(labels))
    width = 0.24
    for offset, (label, key) in enumerate(metrics):
        positions = [value + (offset - 1) * width for value in x]
        ax.bar(positions, [row[key] for row in rows], width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric")
    ax.set_title("Qwen2.5-1.5B LoRA data-size ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/lora_ablation_results.png", dpi=220)
    plt.close(fig)


def write_analysis(rows: list[dict[str, Any]], historical: dict[str, Any] | None) -> None:
    by_name = {row["experiment_name"]: row for row in rows}
    def delta(a: str, b: str, key: str) -> str:
        if by_name[a].get(key) is None or by_name[b].get(key) is None:
            return "n/a"
        return f"{float(by_name[b][key]) - float(by_name[a][key]):+.4f}"
    lines = [
        "# Stage E1: LoRA Training Data-Size Ablation",
        "",
        "E1 fixes the optimizer-step budget at 800 steps. Smaller datasets are revisited more often, while larger datasets expose the model to more distinct training pairs.",
        "",
        "All runs keep the same global batch size and the same 6400 training samples seen at 800 optimizer steps. The 10k-rerun uses a larger micro-batch on the AutoDL RTX 4090 D (`per_device_train_batch_size=8`, `gradient_accumulation_steps=1`) to reduce rented-GPU time; 1k and 3k use the original `1 x 8` accumulation schedule.",
        "",
        "## Marginal Changes",
        "",
        f"- 1k to 3k R@1: {delta('1k', '3k', 'recall_at_1')}; MRR: {delta('1k', '3k', 'mrr')}; NDCG@5: {delta('1k', '3k', 'ndcg_at_5')}.",
        f"- 3k to 10k-rerun R@1: {delta('3k', '10k-rerun', 'recall_at_1')}; MRR: {delta('3k', '10k-rerun', 'mrr')}; NDCG@5: {delta('3k', '10k-rerun', 'ndcg_at_5')}.",
        "",
        "Pairwise accuracy and ranking metrics should be read together; disagreement indicates that better pair-level separation did not necessarily improve top-ranked evidence.",
        "",
        "## E3 Scope Control",
        "",
        "Stage E deliberately skips a new top-k downstream RAG ablation. The project already includes BM25 / LoRA / Cross-Encoder downstream answer comparisons and a Qwen2.5-1.5B / 7B by Original / Strict Prompt 2x2 experiment. Adding more top-k sweeps on the same 50 questions would increase repeated test-set use with limited incremental value, so Stage E focuses on reranker-side E1 and E2.",
    ]
    if historical:
        lines += [
            "",
            "## Historical 10k Reference",
            "",
            "The historical 10k result is shown only as a reproducibility reference. It does not replace the 10k-rerun point in the E1 trend line, and runtime is not directly comparable because the historical adapter metadata and environment are incomplete.",
        ]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/ablation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    env_path = Path("outputs/e1_autodl_environment.json")
    env = read_json(env_path) if env_path.exists() else {}
    rows = [build_row(run, env) for run in RUNS]
    errors = validate(rows)
    historical = read_json(HISTORICAL_10K / "qwen_1_5b_lora_metrics.json") if (HISTORICAL_10K / "qwen_1_5b_lora_metrics.json").exists() else None
    payload = {"status": "ready" if not errors else "failed", "runs": rows, "validation_errors": errors, "historical_10k_reference": historical}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/lora_ablation_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, Path("outputs/lora_ablation_results.csv"))
    write_markdown(rows, errors, historical)
    write_plot(rows)
    write_analysis(rows, historical)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
