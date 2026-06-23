"""Build the Stage F resource-effectiveness profile from committed artifacts.

This script only reads existing metrics, rankings, logs, summaries, and
environment files. It does not import model code, load weights, run inference,
or touch GPU state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


NOT_RECORDED = "not recorded"
METHOD_ORDER = [
    "BM25",
    "Jittor MLP",
    "Jittor TextCNN",
    "Qwen2.5-1.5B zero-shot reranker",
    "Qwen2.5-1.5B LoRA 10k-rerun",
    "Cross-Encoder",
]
METRIC_KEYS = ["recall_at_1", "recall_at_3", "recall_at_5", "ndcg_at_5", "mrr", "pairwise_accuracy"]
CSV_FIELDS = [
    "method",
    "category",
    "framework",
    "implementation_role",
    "pretrained_semantic_ability",
    "local_task_training",
    "training_type",
    "base_model",
    "parameter_count",
    "parameter_scale",
    "trainable_parameter_count",
    "trainable_parameter_ratio",
    "training_hardware",
    "evaluation_hardware",
    "python_version",
    "framework_version",
    "cuda_version",
    "training_runtime_sec",
    "evaluation_runtime_sec",
    "runtime_scope",
    "peak_gpu_memory_mib",
    "peak_cpu_memory_mib",
    "query_count",
    "pair_count",
    "pairs_per_second",
    "milliseconds_per_pair",
    *METRIC_KEYS,
    "effectiveness_comparable",
    "resource_comparability_group",
    "resource_strictly_comparable",
    "measurement_status",
    "evidence_paths",
    "notes",
]
ALLOWED_RUNTIME_SCOPES = {"training_only", "scoring_only", "end_to_end_evaluation", "index_building", "unknown"}
ALLOWED_MEASUREMENT_STATUS = {"measured", "derived_from_measured", "metadata_only", "not_recorded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json", default="outputs/cost_effectiveness_table.json")
    parser.add_argument("--output-csv", default="outputs/cost_effectiveness_table.csv")
    parser.add_argument("--output-markdown", default="outputs/cost_effectiveness_table.md")
    parser.add_argument("--output-audit", default="outputs/cost_effectiveness_input_audit.json")
    parser.add_argument("--output-comparability", default="outputs/cost_effectiveness_comparability.json")
    parser.add_argument("--output-doc", default="docs/cost_effectiveness_analysis.md")
    parser.add_argument("--output-figure", default="docs/figures/cost_effectiveness_tradeoff.png")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(root: Path, path: Path | None) -> str:
    if path is None:
        return NOT_RECORDED
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def existing_paths(root: Path, paths: list[str | None]) -> list[str]:
    out = []
    for item in paths:
        if not item or item == NOT_RECORDED:
            continue
        path = root / item
        if path.exists():
            out.append(item)
    return out


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    aliases = {
        "recall_at_1": ["recall_at_1", "recall@1"],
        "recall_at_3": ["recall_at_3", "recall@3"],
        "recall_at_5": ["recall_at_5", "recall@5"],
        "ndcg_at_5": ["ndcg_at_5", "ndcg@5"],
        "mrr": ["mrr"],
        "pairwise_accuracy": ["pairwise_accuracy"],
    }
    for alias in aliases[key]:
        if alias in metrics:
            return metrics[alias]
    return NOT_RECORDED


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metric_value(metrics, key) for key in METRIC_KEYS}


def ranking_rows(payload: Any, method_key: str | None = None) -> list[dict[str, Any]]:
    if method_key:
        return list(payload[method_key])
    return list(payload)


def candidate_id(candidate: dict[str, Any]) -> str:
    for key in ("doc_id", "candidate_id", "passage_id"):
        if key in candidate:
            return str(candidate[key])
    raise KeyError(f"candidate id missing in keys={sorted(candidate)}")


def ranking_signature(rows: list[dict[str, Any]]) -> dict[str, tuple[tuple[str, int], ...]]:
    signature: dict[str, tuple[tuple[str, int], ...]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        candidates = row.get("ranking") or row.get("candidates") or []
        signature[query_id] = tuple(sorted((candidate_id(item), int(item.get("label", 0))) for item in candidates))
    return signature


def count_rankings(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return len(rows), sum(len(row.get("ranking") or row.get("candidates") or []) for row in rows)


def test_file_signature(path: Path) -> dict[str, tuple[tuple[str, int], ...]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows.append({"query_id": item["query_id"], "candidates": item.get("candidates", [])})
    return ranking_signature(rows)


def file_record(root: Path, path_text: str | None) -> dict[str, Any]:
    if not path_text or path_text == NOT_RECORDED:
        return {"path": NOT_RECORDED, "exists": False, "sha256": NOT_RECORDED}
    path = root / path_text
    if not path.exists():
        return {"path": path_text, "exists": False, "sha256": NOT_RECORDED}
    return {"path": path_text, "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def load_lora_params_from_log(path: Path) -> tuple[Any, Any, Any]:
    if not path.exists():
        return NOT_RECORDED, NOT_RECORDED, NOT_RECORDED
    pattern = re.compile(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(2).replace(",", "")), int(match.group(1).replace(",", "")), float(match.group(3))
    return NOT_RECORDED, NOT_RECORDED, NOT_RECORDED


def textcnn_param_count(root: Path) -> tuple[int, str]:
    # Static calculation from configs/msmarco_medium_textcnn.yaml and src/model_textcnn_jittor.py.
    vocab = read_json(root / "outputs/msmarco_medium_textcnn_vocab.json")
    vocab_size = len(vocab)
    embed_dim = 128
    num_filters = 128
    kernel_sizes = [3, 4, 5]
    embedding = vocab_size * embed_dim
    convs = sum(num_filters * embed_dim * kernel + num_filters for kernel in kernel_sizes)
    fc = num_filters * len(kernel_sizes) + 1
    return embedding + convs + fc, (
        "static formula: vocab_size*128 + sum(128*128*k+128 for k in [3,4,5]) + 128*3+1; "
        f"vocab_size={vocab_size}"
    )


def mlp_param_count() -> tuple[int, str]:
    input_dim = 1537
    hidden_dim = 256
    return input_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1, (
        "static formula: 1537*256+256+256*1+1 from configs/msmarco_medium.yaml and src/model_jittor.py"
    )


def maybe_rate(pair_count: int, runtime: Any) -> tuple[Any, Any]:
    if isinstance(runtime, (int, float)) and runtime > 0:
        return pair_count / runtime, runtime * 1000.0 / pair_count
    return NOT_RECORDED, NOT_RECORDED


def audit_entry(
    root: Path,
    method_name: str,
    paths: dict[str, str | None],
    query_count: int,
    pair_count: int,
    available_fields: list[str],
    missing_fields: list[str],
    is_formal: bool = True,
    is_debug: bool = False,
) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "effectiveness_metrics_path": paths.get("metrics", NOT_RECORDED),
        "rankings_path": paths.get("rankings", NOT_RECORDED),
        "train_summary_path": paths.get("train_summary", NOT_RECORDED),
        "eval_summary_path": paths.get("eval_summary", NOT_RECORDED),
        "environment_path": paths.get("environment", NOT_RECORDED),
        "gpu_monitor_path": paths.get("gpu_monitor", NOT_RECORDED),
        "config_path": paths.get("config", NOT_RECORDED),
        "query_count": query_count,
        "pair_count": pair_count,
        "available_fields": sorted(set(available_fields)),
        "missing_fields": sorted(set(missing_fields)),
        "file_sha256": {key: file_record(root, value) for key, value in paths.items()},
        "is_formal_result": is_formal,
        "is_debug_result": is_debug,
    }


def compact(value: Any) -> str:
    if value == NOT_RECORDED:
        return NOT_RECORDED
    if value == "N/A":
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_fmt(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def seconds_fmt(value: Any, scope: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}s ({scope})"
    return NOT_RECORDED


def mib_fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f} MiB"
    return NOT_RECORDED


def build_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Method",
        "R@1",
        "MRR",
        "NDCG@5",
        "Training",
        "Pretrained semantics",
        "Model scale",
        "Eval runtime",
        "Peak VRAM",
        "Hardware",
        "Resource comparability",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    metric_fmt(row["recall_at_1"]),
                    metric_fmt(row["mrr"]),
                    metric_fmt(row["ndcg_at_5"]),
                    row["local_task_training"],
                    compact(row["pretrained_semantic_ability"]),
                    compact(row["parameter_scale"]),
                    seconds_fmt(row["evaluation_runtime_sec"], row["runtime_scope"]),
                    mib_fmt(row["peak_gpu_memory_mib"]),
                    compact(row["evaluation_hardware"]),
                    f"{row['resource_comparability_group']}; strict={str(row['resource_strictly_comparable']).lower()}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def make_figure(path: Path, rows: list[dict[str, Any]], figure_type: str) -> None:
    import matplotlib.pyplot as plt

    names = ["BM25", "MLP", "TextCNN", "Qwen zero-shot", "LoRA 10k", "Cross-Encoder"]
    values = [float(row["ndcg_at_5"]) for row in rows]
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#b279a2", "#e45756"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(names, values, color=colors, edgecolor="#2f2f2f", linewidth=0.7)
    ax.set_title("Resource and Effectiveness Profile", fontsize=14, pad=12)
    ax.set_ylabel("NDCG@5 on MS MARCO medium")
    ax.set_ylim(0, max(values) + 0.12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    notes = [
        "No train\nNo pretrained\nresource N/R",
        "From scratch\nJittor\nresource N/R",
        "From scratch\nJittor\nresource N/R",
        "Zero-shot\npretrained\nruntime recorded",
        "LoRA train\n4090 D\nruntime recorded",
        "Pretrained\nCUDA\nruntime recorded",
    ]
    for bar, value, note in zip(bars, values, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.018, note, ha="center", va="bottom", fontsize=8, color="#1f1f1f")
    ax.text(
        0.5,
        -0.20,
        "Runtime and memory are not strictly comparable across methods. "
        "Effectiveness uses the same 500-query / 4044-pair candidate pool.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.text(
        0.99,
        0.98,
        f"Figure type: {figure_type}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_analysis_doc(
    rows: list[dict[str, Any]],
    scoring_runs: list[dict[str, Any]],
    historical: dict[str, Any],
    figure_type: str,
    test_sha: str,
) -> str:
    lookup = {row["method"]: row for row in rows}
    md_table = build_markdown(rows)
    historical_metrics = normalize_metrics(historical)
    e2_lines = []
    for run in scoring_runs:
        e2_lines.append(
            f"| {run['scoring_method']} | {run['recall_at_1']:.4f} | {run['ndcg_at_5']:.4f} | "
            f"{run['mrr']:.4f} | {run['inference_runtime_sec']:.3f} | {run['parse_failure_rate']:.4f} | "
            f"{run['query_tie_rate']:.4f} |"
        )
    return f"""# Resource and Effectiveness Profile

## 1. Purpose

Stage F builds a resource and effectiveness profile for six reranking methods in this repository. It is not a strict hardware-normalized benchmark and does not claim a full RankRAG reproduction. The project remains a Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking.

## 2. Methods

- BM25: lexical retrieval baseline without neural model training.
- Jittor MLP: lightweight Jittor neural ranking baseline trained from scratch on handcrafted features.
- Jittor TextCNN: lightweight Jittor neural ranking baseline trained from scratch on local text patterns.
- Qwen2.5-1.5B zero-shot reranker: pretrained LLM relevance scoring without local task training.
- Qwen2.5-1.5B LoRA 10k-rerun: the Stage E 10k-rerun LoRA adapter using `logprob_margin` scoring.
- Cross-Encoder: external pretrained semantic reranker reference, not the Jittor reproduction body.

## 3. Measurement Protocol

The effectiveness metrics come from existing committed metrics files. The six main methods all contain 500 queries and 4044 query-passage pairs, and the candidate-pool signature matches across methods. The MS MARCO medium test split SHA256 is `{test_sha}`. No model was retrained, no model inference was rerun, and missing resource fields are recorded as `not recorded`.

Resource records come from heterogeneous artifacts: local Jittor logs, JittorLLM metrics, Cross-Encoder metrics, and AutoDL RTX 4090 D LoRA logs. Because hardware, software, batching, and runtime scopes differ, resource fields form a profile rather than a fair speed ranking.

## 4. Effectiveness Profile

{md_table}
BM25 reaches R@1={lookup['BM25']['recall_at_1']:.4f} and NDCG@5={lookup['BM25']['ndcg_at_5']:.4f}. The lightweight Jittor MLP is close on R@1 but lower on NDCG@5, while Jittor TextCNN is weaker on this candidate pool. Qwen2.5-1.5B zero-shot improves NDCG@5 over the lightweight Jittor models, showing the value of pretrained semantic ability even without task training. The Stage E LoRA 10k-rerun raises R@1 to {lookup['Qwen2.5-1.5B LoRA 10k-rerun']['recall_at_1']:.4f} and NDCG@5 to {lookup['Qwen2.5-1.5B LoRA 10k-rerun']['ndcg_at_5']:.4f}. The Cross-Encoder remains the strongest ranking reference in this set, with R@1={lookup['Cross-Encoder']['recall_at_1']:.4f} and NDCG@5={lookup['Cross-Encoder']['ndcg_at_5']:.4f}.

## 5. Resource Profile

BM25 has no neural parameter count and no recorded runtime or hardware in the committed metrics. Jittor MLP and Jittor TextCNN have static parameter counts derived from source/configuration, but their training and evaluation runtime summaries, hardware, and peak memory were not recorded. Qwen zero-shot records scoring runtime and observed GPU memory in its metrics, but no explicit GPU model or software environment. The LoRA 10k-rerun records AutoDL NVIDIA GeForce RTX 4090 D hardware, Python/Torch/CUDA environment, training runtime, scoring runtime, and peak GPU memory. Cross-Encoder records CUDA device, batch size, evaluation runtime, query count, and pair count, but not the exact GPU model or peak memory.

No pair of methods is marked as strictly resource comparable in the main table. The strict flag requires the same hardware, software environment, query and pair count, runtime scope, batch or serial path, and loader scope. The current artifacts satisfy the effectiveness comparability requirement, but not the strict resource comparability requirement.

## 6. Internal LoRA Scoring Cost

The E2 scoring ablation uses the same 10k-rerun LoRA adapter. These rows are not separate main methods; they isolate scoring-rule cost and behavior.

| Scoring method | R@1 | NDCG@5 | MRR | Runtime sec | Parse failure rate | Query tie rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(e2_lines)}

`generate_parse` calls generation for each pair and is much slower while producing many ties. `relevant_logprob` scores the `Relevant` label sequence only. `logprob_margin` scores both `Relevant` and `Irrelevant` label sequences and uses their difference; it is the Stage E default used in the main table.

## 7. Main Trade-off

The practical ladder is lexical low-resource retrieval, lightweight Jittor neural baselines, zero-shot LLM reranking, task-adapted LoRA reranking, and a mature Cross-Encoder reference. BM25 is cheap and simple but semantically limited. Jittor MLP/TextCNN are useful for framework alignment and lightweight neural comparisons, not for maximum ranking quality. Qwen zero-shot adds pretrained semantic ability without task training. LoRA adds task-specific adaptation and produces a clear effectiveness gain over zero-shot in this setup. Cross-Encoder gives the strongest ranking metrics, while LoRA is most relevant to the RankRAG-style LLM reranking reproduction objective.

## 8. Limitations

- Runtime and memory were collected under different hardware and implementation paths.
- Some methods lack recorded training runtime, evaluation runtime, GPU model, or peak memory.
- Runtime scopes differ and may or may not include loader overhead.
- Batch sizes and serial/parallel scoring paths differ.
- The current table is not a strict speed benchmark or a hardware-normalized efficiency ranking.
- Parameter scale and observed latency are not linearly related.
- The conclusions are limited to the current MS MARCO medium subset.

## Historical 10k Reproducibility Reference

The historical 10k LoRA directory is retained only as a reproducibility reference: `outputs/lora_qwen_1_5b_10k_lr1e4_s800/`. The main Stage F table uses `outputs/lora_ablation_data_10k_rerun/` instead. Historical metrics are R@1={historical_metrics['recall_at_1']:.4f}, R@3={historical_metrics['recall_at_3']:.4f}, R@5={historical_metrics['recall_at_5']:.4f}, NDCG@5={historical_metrics['ndcg_at_5']:.4f}, MRR={historical_metrics['mrr']:.4f}, pairwise accuracy={historical_metrics['pairwise_accuracy']:.4f}. Training runtime and memory should not be strictly compared with the rerun because the exact environment and code version are not fully identical.

## Figure

`docs/figures/cost_effectiveness_tradeoff.png` is a non-strict resource profile figure because fewer than three methods satisfy strict resource comparability. It plots NDCG@5 and annotates training/pretraining/hardware context rather than drawing a runtime scatter.
"""


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    out_json = root / args.output_json
    out_csv = root / args.output_csv
    out_md = root / args.output_markdown
    out_audit = root / args.output_audit
    out_comp = root / args.output_comparability
    out_doc = root / args.output_doc
    out_fig = root / args.output_figure

    test_signature = test_file_signature(root / "data/processed/msmarco_medium/test.jsonl")
    lora_test_signature = test_file_signature(root / "data/processed/lora_qwen_1_5b_10k/test_queries.jsonl")
    if test_signature != lora_test_signature:
        raise SystemExit("MS MARCO medium and LoRA test query candidate signatures differ")
    test_sha = sha256_file(root / "data/processed/msmarco_medium/test.jsonl")

    specs: list[dict[str, Any]] = []
    bm25_payload = read_json(root / "outputs/msmarco_medium_retrieval_baseline_metrics.json")
    bm25_metrics = bm25_payload["bm25"]["metrics"]
    bm25_rankings = ranking_rows(read_json(root / "outputs/msmarco_medium_retrieval_baseline_rankings.json"), "bm25")
    specs.append(
        {
            "method": "BM25",
            "metrics": bm25_metrics,
            "rankings": bm25_rankings,
            "paths": {
                "metrics": "outputs/msmarco_medium_retrieval_baseline_metrics.json",
                "rankings": "outputs/msmarco_medium_retrieval_baseline_rankings.json",
                "config": NOT_RECORDED,
            },
            "fields": {
                "category": "lexical retrieval baseline",
                "framework": "rank_bm25",
                "implementation_role": "lexical baseline",
                "pretrained_semantic_ability": False,
                "local_task_training": "none",
                "training_type": "none",
                "base_model": "N/A",
                "parameter_count": "N/A",
                "parameter_scale": "N/A",
                "trainable_parameter_count": "N/A",
                "trainable_parameter_ratio": "N/A",
                "training_hardware": NOT_RECORDED,
                "evaluation_hardware": NOT_RECORDED,
                "runtime_scope": "index_building",
                "resource_comparability_group": "cpu_local_retrieval",
                "measurement_status": "metadata_only",
                "notes": "BM25 parameter_count is not applicable; runtime/hardware were not recorded in the committed metrics.",
            },
        }
    )

    mlp_count, mlp_note = mlp_param_count()
    mlp_metrics = read_json(root / "outputs/msmarco_medium_jittor_metrics.json")
    mlp_rankings = ranking_rows(read_json(root / "outputs/msmarco_medium_jittor_test_rankings.json"))
    specs.append(
        {
            "method": "Jittor MLP",
            "metrics": mlp_metrics,
            "rankings": mlp_rankings,
            "paths": {
                "metrics": "outputs/msmarco_medium_jittor_metrics.json",
                "rankings": "outputs/msmarco_medium_jittor_test_rankings.json",
                "config": "configs/msmarco_medium.yaml",
                "train_log": "logs/msmarco_medium_jittor_train.log",
            },
            "fields": {
                "category": "lightweight neural reranker",
                "framework": "Jittor",
                "implementation_role": "Jittor lightweight reproduction baseline",
                "pretrained_semantic_ability": False,
                "local_task_training": "from_scratch",
                "training_type": "pairwise ranking",
                "base_model": "small MLP scorer",
                "parameter_count": mlp_count,
                "parameter_scale": "0.394M static",
                "trainable_parameter_count": mlp_count,
                "trainable_parameter_ratio": 1.0,
                "training_hardware": NOT_RECORDED,
                "evaluation_hardware": NOT_RECORDED,
                "runtime_scope": "unknown",
                "resource_comparability_group": "jittor_local_training",
                "measurement_status": "metadata_only",
                "notes": f"Runtime, hardware, and memory are not recorded. Parameter count uses {mlp_note}.",
            },
        }
    )

    textcnn_count, textcnn_note = textcnn_param_count(root)
    textcnn_metrics = read_json(root / "outputs/msmarco_medium_textcnn_jittor_metrics.json")
    textcnn_rankings = ranking_rows(read_json(root / "outputs/msmarco_medium_textcnn_jittor_test_rankings.json"))
    specs.append(
        {
            "method": "Jittor TextCNN",
            "metrics": textcnn_metrics,
            "rankings": textcnn_rankings,
            "paths": {
                "metrics": "outputs/msmarco_medium_textcnn_jittor_metrics.json",
                "rankings": "outputs/msmarco_medium_textcnn_jittor_test_rankings.json",
                "config": "configs/msmarco_medium_textcnn.yaml",
                "train_log": "logs/msmarco_medium_textcnn_jittor_train.log",
            },
            "fields": {
                "category": "lightweight neural reranker",
                "framework": "Jittor",
                "implementation_role": "Jittor lightweight reproduction baseline",
                "pretrained_semantic_ability": False,
                "local_task_training": "from_scratch",
                "training_type": "pairwise ranking",
                "base_model": "small TextCNN scorer",
                "parameter_count": textcnn_count,
                "parameter_scale": "3.899M static",
                "trainable_parameter_count": textcnn_count,
                "trainable_parameter_ratio": 1.0,
                "training_hardware": NOT_RECORDED,
                "evaluation_hardware": NOT_RECORDED,
                "runtime_scope": "unknown",
                "resource_comparability_group": "jittor_local_training",
                "measurement_status": "metadata_only",
                "notes": f"Runtime, hardware, and memory are not recorded. Parameter count uses {textcnn_note}.",
            },
        }
    )

    qwen_metrics = read_json(root / "outputs/jittorllm_qwen2_1_5b_full/metrics.json")
    qwen_rankings = ranking_rows(read_json(root / "outputs/jittorllm_qwen2_1_5b_full/rankings.json"))
    specs.append(
        {
            "method": "Qwen2.5-1.5B zero-shot reranker",
            "metrics": qwen_metrics,
            "rankings": qwen_rankings,
            "paths": {
                "metrics": "outputs/jittorllm_qwen2_1_5b_full/metrics.json",
                "rankings": "outputs/jittorllm_qwen2_1_5b_full/rankings.json",
                "config": "configs/jittorllm_qwen2_1_5b_full.yaml",
            },
            "fields": {
                "category": "LLM zero-shot reranker",
                "framework": "JittorLLM",
                "implementation_role": "LLM inference supplement",
                "pretrained_semantic_ability": True,
                "local_task_training": "none",
                "training_type": "zero_shot",
                "base_model": "Qwen2.5-1.5B-Instruct",
                "parameter_count": NOT_RECORDED,
                "parameter_scale": "1.5B class",
                "trainable_parameter_count": "N/A",
                "trainable_parameter_ratio": "N/A",
                "training_hardware": "N/A",
                "evaluation_hardware": NOT_RECORDED,
                "evaluation_runtime_sec": qwen_metrics["inference_time_sec"],
                "peak_gpu_memory_mib": qwen_metrics["gpu_memory_peak_observed_mb"],
                "runtime_scope": "scoring_only",
                "resource_comparability_group": "qwen_zero_shot_recorded_environment",
                "measurement_status": "measured",
                "notes": "Metrics record JittorLLM backend, label-token logit-margin scoring, inference time, and observed GPU memory, but not the GPU model.",
            },
        }
    )

    lora_metrics = read_json(root / "outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_metrics.json")
    lora_summary = read_json(root / "outputs/lora_ablation_data_10k_rerun/train_summary.json")
    lora_rankings = ranking_rows(read_json(root / "outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_rankings.json"))
    env = read_json(root / "outputs/e1_autodl_environment.json")
    all_params, trainable_params, trainable_ratio = load_lora_params_from_log(root / "logs/e1_autodl_4090d/10k_rerun_train.log")
    specs.append(
        {
            "method": "Qwen2.5-1.5B LoRA 10k-rerun",
            "metrics": lora_metrics,
            "rankings": lora_rankings,
            "paths": {
                "metrics": "outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_metrics.json",
                "rankings": "outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_rankings.json",
                "train_summary": "outputs/lora_ablation_data_10k_rerun/train_summary.json",
                "environment": "outputs/e1_autodl_environment.json",
                "gpu_monitor": "logs/e1_autodl_4090d/10k_rerun_gpu.csv",
                "eval_gpu_monitor": "logs/e1_autodl_4090d/10k_rerun_eval_gpu.csv",
                "config": "configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml",
                "train_log": "logs/e1_autodl_4090d/10k_rerun_train.log",
                "eval_log": "logs/e1_autodl_4090d/10k_rerun_eval.log",
            },
            "fields": {
                "category": "LLM LoRA reranker",
                "framework": "PyTorch + Transformers + PEFT",
                "implementation_role": "RankRAG-style LLM reranking reproduction target",
                "pretrained_semantic_ability": True,
                "local_task_training": "lora_finetuning",
                "training_type": "LoRA fine-tuning",
                "base_model": lora_summary["model_name"],
                "parameter_count": all_params,
                "parameter_scale": "1.5B class",
                "trainable_parameter_count": trainable_params,
                "trainable_parameter_ratio": trainable_ratio,
                "training_hardware": lora_summary["device_name"],
                "evaluation_hardware": lora_metrics["device_name"],
                "python_version": env.get("python_version", NOT_RECORDED),
                "framework_version": f"torch {env.get('torch_version', NOT_RECORDED)}; transformers {env.get('transformers_version', NOT_RECORDED)}; peft {env.get('peft_version', NOT_RECORDED)}",
                "cuda_version": f"torch_cuda {env.get('torch_cuda', NOT_RECORDED)}; nvidia_smi CUDA 13.0",
                "training_runtime_sec": lora_summary["runtime_sec"],
                "evaluation_runtime_sec": lora_metrics["inference_time_sec"],
                "peak_gpu_memory_mib": lora_summary["peak_gpu_memory_gib"] * 1024.0,
                "runtime_scope": "scoring_only",
                "resource_comparability_group": "autodl_4090d_lora",
                "measurement_status": "measured",
                "notes": "Main LoRA row uses Stage E 10k-rerun and logprob_margin scoring. Peak GPU memory is the training peak from train_summary; eval peak is lower and recorded separately in metrics.",
            },
        }
    )

    cross_metrics = read_json(root / "outputs/msmarco_medium_cross_encoder_metrics.json")
    cross_rankings = ranking_rows(read_json(root / "outputs/msmarco_medium_cross_encoder_rankings.json"))
    specs.append(
        {
            "method": "Cross-Encoder",
            "metrics": cross_metrics,
            "rankings": cross_rankings,
            "paths": {
                "metrics": "outputs/msmarco_medium_cross_encoder_metrics.json",
                "rankings": "outputs/msmarco_medium_cross_encoder_rankings.json",
                "config": NOT_RECORDED,
                "script": "src/pretrained_cross_encoder_reference.py",
            },
            "fields": {
                "category": "pretrained semantic reranker",
                "framework": "sentence-transformers / PyTorch",
                "implementation_role": "external pretrained semantic reranker reference",
                "pretrained_semantic_ability": True,
                "local_task_training": "none_in_this_project",
                "training_type": "external pretrained/fine-tuned checkpoint",
                "base_model": cross_metrics["model_name"],
                "parameter_count": NOT_RECORDED,
                "parameter_scale": "MiniLM-L6 cross-encoder class",
                "trainable_parameter_count": "N/A",
                "trainable_parameter_ratio": "N/A",
                "training_hardware": "N/A",
                "evaluation_hardware": cross_metrics["device"],
                "evaluation_runtime_sec": cross_metrics["total_time_sec"],
                "runtime_scope": "scoring_only",
                "resource_comparability_group": "pretrained_cross_encoder_unknown_hardware",
                "measurement_status": "measured",
                "notes": "Cross-Encoder metrics record CUDA device and batch size 32, but not exact GPU model or peak memory.",
            },
        }
    )

    baseline_signature = ranking_signature(bm25_rankings)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for spec in specs:
        method = spec["method"]
        query_count, pair_count = count_rankings(spec["rankings"])
        sig = ranking_signature(spec["rankings"])
        comparable = sig == baseline_signature and query_count == 500 and pair_count == 4044
        fields = {
            "method": method,
            "training_runtime_sec": NOT_RECORDED,
            "evaluation_runtime_sec": NOT_RECORDED,
            "peak_gpu_memory_mib": NOT_RECORDED,
            "peak_cpu_memory_mib": NOT_RECORDED,
            "pairs_per_second": NOT_RECORDED,
            "milliseconds_per_pair": NOT_RECORDED,
            "python_version": NOT_RECORDED,
            "framework_version": NOT_RECORDED,
            "cuda_version": NOT_RECORDED,
            **spec["fields"],
            **normalize_metrics(spec["metrics"]),
            "query_count": query_count,
            "pair_count": pair_count,
            "effectiveness_comparable": comparable,
            "resource_strictly_comparable": False,
            "evidence_paths": existing_paths(root, list(spec["paths"].values())),
        }
        if fields["runtime_scope"] not in ALLOWED_RUNTIME_SCOPES:
            raise SystemExit(f"{method}: invalid runtime_scope {fields['runtime_scope']}")
        if fields["measurement_status"] not in ALLOWED_MEASUREMENT_STATUS:
            raise SystemExit(f"{method}: invalid measurement_status {fields['measurement_status']}")
        pps, mpp = maybe_rate(pair_count, fields["evaluation_runtime_sec"])
        if fields["pairs_per_second"] == NOT_RECORDED:
            fields["pairs_per_second"] = pps
        if fields["milliseconds_per_pair"] == NOT_RECORDED:
            fields["milliseconds_per_pair"] = mpp
        rows.append(fields)

        available = [key for key, value in fields.items() if value != NOT_RECORDED]
        missing = [key for key, value in fields.items() if value == NOT_RECORDED]
        audit.append(audit_entry(root, method, spec["paths"], query_count, pair_count, available, missing))

    strict_count = sum(1 for row in rows if row["resource_strictly_comparable"])
    figure_type = "strict_runtime_scatter" if strict_count >= 3 else "non_strict_resource_profile"

    comparability = {
        "status": "ready",
        "test_query_file": "data/processed/msmarco_medium/test.jsonl",
        "test_query_sha256": test_sha,
        "lora_test_query_file": "data/processed/lora_qwen_1_5b_10k/test_queries.jsonl",
        "lora_test_query_sha256": sha256_file(root / "data/processed/lora_qwen_1_5b_10k/test_queries.jsonl"),
        "method_count": len(rows),
        "effectiveness_comparable_methods": [row["method"] for row in rows if row["effectiveness_comparable"]],
        "strictly_resource_comparable_methods": [row["method"] for row in rows if row["resource_strictly_comparable"]],
        "strictly_comparable_method_count": strict_count,
        "figure_type": figure_type,
        "reason": "Effectiveness is comparable on the same candidate pool; resource fields use heterogeneous hardware/software/runtime scopes.",
    }

    table_payload = {
        "status": "ready",
        "title": "Resource and Effectiveness Profile",
        "generated_from_existing_artifacts_only": True,
        "methods": rows,
        "comparability": comparability,
    }
    write_json(out_json, table_payload)
    write_json(out_audit, {"status": "ready", "methods": audit})
    write_json(out_comp, comparability)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key, NOT_RECORDED) for key in CSV_FIELDS}
            csv_row["evidence_paths"] = "; ".join(row["evidence_paths"])
            writer.writerow(csv_row)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("# Resource and Effectiveness Profile\n\n" + build_markdown(rows), encoding="utf-8")
    make_figure(out_fig, rows, figure_type)
    scoring_runs = read_json(root / "outputs/lora_scoring_ablation_results.json")["runs"]
    historical = read_json(root / "outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_metrics.json")
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text(build_analysis_doc(rows, scoring_runs, historical, figure_type, test_sha), encoding="utf-8")
    print(json.dumps({"status": "ready", "methods": METHOD_ORDER, "figure_type": figure_type}, indent=2))


if __name__ == "__main__":
    main()
