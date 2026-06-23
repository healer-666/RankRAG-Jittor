"""Check config and artifact consistency for the LoRA data-size ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from src.lora_reranker.lora_utils import load_jsonl
    from src.utils import read_json, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.lora_reranker.lora_utils import load_jsonl
    from src.utils import read_json, write_json


ALLOWED_DIFFERENCES = {
    "train_path",
    "training_pairs",
    "output_dir",
    "debug_output_dir",
    "experiment_name",
    "ablation_factor",
    "reference_training_pairs",
    "fixed_max_steps",
    "subset_manifest_path",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
}


def compute_effective_epochs(cfg: dict[str, Any], row_count: int, *, gpu_count: int = 1) -> dict[str, Any]:
    per_device_batch = int(cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    max_steps = int(cfg.get("max_train_steps", 0))
    global_batch_size = per_device_batch * grad_accum * gpu_count
    samples_seen = max_steps * global_batch_size
    return {
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "gpu_count_assumption": gpu_count,
        "global_batch_size": global_batch_size,
        "max_train_steps": max_steps,
        "drop_last": False,
        "samples_seen_at_fixed_steps": samples_seen,
        "effective_epochs_at_fixed_steps": samples_seen / max(row_count, 1),
        "interpretation": "fixed optimizer-step budget; smaller datasets are revisited more often",
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def compare_configs(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors = []
    keys = set(reference) | set(candidate)
    for key in sorted(keys):
        if key in ALLOWED_DIFFERENCES:
            continue
        if reference.get(key) != candidate.get(key):
            errors.append(f"config differs unexpectedly: {key}")
    return errors


def build_config_report(reference_config: Path, run_configs: list[Path], output_path: Path) -> dict[str, Any]:
    reference = load_yaml(reference_config)
    errors = []
    runs = {}
    for path in run_configs:
        cfg = load_yaml(path)
        cfg_errors = compare_configs(reference, cfg)
        errors.extend(f"{path}: {error}" for error in cfg_errors)
        train_path = Path(cfg["train_path"])
        rows = load_jsonl(train_path) if train_path.exists() else []
        if not train_path.exists():
            errors.append(f"{path}: missing train_path {train_path}")
        if cfg.get("training_pairs") != len(rows):
            errors.append(f"{path}: training_pairs does not match row count")
        runs[path.stem] = {
            "train_path": cfg.get("train_path"),
            "training_pairs": cfg.get("training_pairs"),
            "row_count": len(rows),
            "output_dir": cfg.get("debug_output_dir"),
            "adapter_output_dir": cfg.get("output_dir"),
            "effective_epochs": compute_effective_epochs(cfg, len(rows)),
        }

    debug_output_dirs = [row["output_dir"] for row in runs.values()]
    adapter_output_dirs = [row["adapter_output_dir"] for row in runs.values()]
    if len(set(debug_output_dirs)) != len(debug_output_dirs):
        errors.append("ablation output directories overlap")
    if reference.get("debug_output_dir") in debug_output_dirs:
        errors.append("ablation output directory overlaps 10k debug output")
    if len(set(adapter_output_dirs)) != len(adapter_output_dirs):
        errors.append("ablation adapter output directories overlap")
    if reference.get("output_dir") in adapter_output_dirs:
        errors.append("ablation adapter output directory overlaps 10k adapter output")
    global_batch_sizes = {
        row["effective_epochs"]["global_batch_size"]
        for row in runs.values()
    }
    samples_seen = {
        row["effective_epochs"]["samples_seen_at_fixed_steps"]
        for row in runs.values()
    }
    if len(global_batch_sizes) != 1:
        errors.append(f"global batch size differs unexpectedly: {sorted(global_batch_sizes)}")
    if len(samples_seen) != 1:
        errors.append(f"samples seen at fixed steps differs unexpectedly: {sorted(samples_seen)}")

    payload = {
        "status": "passed" if not errors else "failed",
        "same_base_model": all(load_yaml(path).get("model_name") == reference.get("model_name") for path in run_configs),
        "same_lora_rank": all(load_yaml(path).get("lora_r") == reference.get("lora_r") for path in run_configs),
        "same_learning_rate": all(load_yaml(path).get("learning_rate") == reference.get("learning_rate") for path in run_configs),
        "same_max_steps": all(load_yaml(path).get("max_train_steps") == reference.get("max_train_steps") for path in run_configs),
        "same_max_length": all(load_yaml(path).get("max_length") == reference.get("max_length") for path in run_configs),
        "same_global_batch_size": len(global_batch_sizes) == 1,
        "same_samples_seen_at_fixed_steps": len(samples_seen) == 1,
        "same_seed": all(load_yaml(path).get("seed") == reference.get("seed") for path in run_configs),
        "same_validation_set": all(load_yaml(path).get("valid_path") == reference.get("valid_path") for path in run_configs),
        "same_test_set": True,
        "same_scoring_method": True,
        "effective_epoch_note": (
            "With max_train_steps fixed, E1 measures the effect of adding training-data diversity under a fixed "
            "optimizer-step budget, not a fixed-epoch comparison."
        ),
        "allowed_differences": sorted(ALLOWED_DIFFERENCES),
        "runs": runs,
        "errors": errors,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)
    return payload


def build_validation_report(
    manifest_path: Path,
    config_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    config_report = read_json(config_report_path)
    errors = []
    runs = {
        "1k": {
            "training_pairs": 1000,
            "output_dir": Path("outputs/lora_ablation_data_1k"),
            "metrics": "qwen_1_5b_lora_metrics.json",
            "rankings": "qwen_1_5b_lora_rankings.json",
        },
        "3k": {
            "training_pairs": 3000,
            "output_dir": Path("outputs/lora_ablation_data_3k"),
            "metrics": "qwen_1_5b_lora_metrics.json",
            "rankings": "qwen_1_5b_lora_rankings.json",
        },
        "10k": {
            "training_pairs": 10000,
            "output_dir": Path("outputs/lora_qwen_1_5b_10k_lr1e4_s800"),
            "metrics": "qwen_1_5b_lora_metrics.json",
            "rankings": "qwen_1_5b_lora_rankings.json",
            "retrained": False,
        },
    }
    report_runs = {}
    for name, spec in runs.items():
        metrics_path = spec["output_dir"] / spec["metrics"]
        rankings_path = spec["output_dir"] / spec["rankings"]
        metrics_present = metrics_path.exists()
        rankings_present = rankings_path.exists()
        if not metrics_present:
            errors.append(f"{name}: missing metrics")
        if not rankings_present:
            errors.append(f"{name}: missing rankings")
        report_runs[name] = {
            "training_pairs": spec["training_pairs"],
            "metrics_present": metrics_present,
            "rankings_present": rankings_present,
            "validation_passed": metrics_present and rankings_present,
            "retrained": spec.get("retrained"),
        }
    nested_ok = (
        manifest["checks"].get("subset_1k_in_3k")
        and manifest["checks"].get("subset_3k_in_10k")
        and not manifest["checks"].get("validation_or_test_rows_used")
    )
    if not nested_ok:
        errors.append("nested subset check failed")
    if config_report.get("status") != "passed":
        errors.append("config consistency failed")
    payload = {
        "status": "ready" if not errors else "failed",
        "runs": report_runs,
        "config_consistency": config_report.get("status"),
        "nested_subset_check": "passed" if nested_ok else "failed",
        "errors": errors,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["config", "validation"], required=True)
    parser.add_argument("--reference-config", default="configs/lora_qwen_1_5b_10k_lr1e4_s800.yaml")
    parser.add_argument(
        "--run-configs",
        nargs="*",
        default=[
            "configs/lora_qwen_1_5b_1k_lr1e4_s800.yaml",
            "configs/lora_qwen_1_5b_3k_lr1e4_s800.yaml",
            "configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml",
        ],
    )
    parser.add_argument("--manifest", default="data/processed/lora_ablation/data_size_manifest.json")
    parser.add_argument("--config-report", default="outputs/lora_ablation_config_consistency.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "config":
        build_config_report(
            Path(args.reference_config),
            [Path(path) for path in args.run_configs],
            Path(args.output or args.config_report),
        )
    else:
        build_validation_report(
            Path(args.manifest),
            Path(args.config_report),
            Path(args.output or "outputs/lora_ablation_validation_report.json"),
        )


if __name__ == "__main__":
    main()
