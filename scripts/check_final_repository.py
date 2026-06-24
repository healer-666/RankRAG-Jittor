"""Final read-only repository audit for documentation release."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "README.md",
    "README.zh-CN.md",
    "docs/final_results.md",
    "docs/reproduction.md",
    "docs/figures/project_pipeline.mmd",
    "docs/figures/rankrag_jittor_overview.excalidraw",
    "docs/figures/rankrag_jittor_overview.svg",
    "docs/figures/rankrag_jittor_overview.png",
    "docs/figures/main_reranking_results.svg",
    "docs/figures/main_reranking_results.png",
    "docs/figures/pytorch_jittor_alignment.svg",
    "docs/figures/pytorch_jittor_alignment.png",
    "docs/figures/readme_lora_ablation.svg",
    "docs/figures/readme_lora_ablation.png",
    "docs/figures/readme_error_taxonomy.svg",
    "docs/figures/readme_error_taxonomy.png",
    "docs/figures/readme_resource_profile.svg",
    "docs/figures/readme_resource_profile.png",
    "outputs/final_results_summary.json",
    "outputs/final_results_summary.csv",
    "scripts/build_final_project_summary.py",
    "scripts/build_readme_figures.py",
    "scripts/check_final_repository.py",
]
FORBIDDEN_WEIGHT_PATTERNS = [
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    "checkpoint",
    "adapter_model",
    "optimizer",
    "scheduler",
    "rng_state",
    "huggingface",
    "hf_cache",
]
SECRET_PATTERNS = [
    ("api_key", re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}")),
    ("access_token", re.compile(r"(?i)(access[_-]?token|bearer)\s*[:= ]\s*['\"]?[A-Za-z0-9_\-.]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password", re.compile(r"(?i)(password|passwd)\s*[:=]\s*['\"]?[^'\"\s]{8,}")),
]
HARDCODED_PATH_PATTERNS = [
    ("windows_drive", re.compile(r"[A-Za-z]:\\[^\\\r\n\"']+\\")),
    ("windows_user", re.compile(r"C:\\Users\\")),
    ("autodl_tmp", re.compile(r"/root/autodl-tmp")),
]
MARKDOWN_TARGETS = ["README.md", "README.zh-CN.md"]
POSITIONING_EN = "A Jittor-based lightweight reproduction and empirical analysis of RankRAG-style LLM reranking."
POSITIONING_ZH = "基于 Jittor 的 RankRAG 风格大模型重排序轻量复现与实证分析。"
FORBIDDEN_CLAIMS = [
    "Full reproduction of RankRAG",
    "Complete RankRAG implementation",
    "Reproduced all RankRAG results",
    "State-of-the-art",
    "Jittor implementation of the full RankRAG model",
]
STAGE_CHECKS = [
    ("lora_data_config", [sys.executable, "scripts/check_lora_data_ablation.py", "--mode", "config", "--output", os.devnull]),
    ("lora_data_validation", [sys.executable, "scripts/check_lora_data_ablation.py", "--mode", "validation", "--output", os.devnull]),
    ("cost_effectiveness", [sys.executable, "scripts/check_cost_effectiveness_outputs.py"]),
    ("error_taxonomy", [sys.executable, "scripts/check_error_taxonomy_outputs.py"]),
]
SAFE_HELP_SCRIPTS = [
    "scripts/prepare_msmarco_subset.py",
    "scripts/run_downstream_rag_eval.py",
    "scripts/validate_downstream_rag_results.py",
    "scripts/check_downstream_rag_protocol_consistency.py",
    "scripts/validate_jittorllm_qwen2_results.py",
    "src/baseline_retrieval.py",
    "src/pretrained_cross_encoder_reference.py",
    "src/aggregate_l2_results.py",
    "src/audit_downstream_rag_data.py",
    "src/build_downstream_qa_subset.py",
    "src/aggregate_downstream_rag_results.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json", default="outputs/final_repository_audit.json")
    parser.add_argument("--output-md", default="docs/final_repository_audit.md")
    return parser.parse_args()


def run(root: Path, command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }
    except Exception as exc:
        return {"command": command, "returncode": -1, "error": str(exc)}


def git_lines(root: Path, *args: str) -> list[str]:
    out = subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL)
    return [line for line in out.splitlines() if line.strip()]


def is_text(path: Path) -> bool:
    try:
        data = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\0" not in data


def tracked_files(root: Path) -> list[Path]:
    return [root / line for line in git_lines(root, "ls-files")]


def scan_large_files(root: Path, files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings = []
    failures = []
    for path in files:
        size = path.stat().st_size
        record = {"path": path.relative_to(root).as_posix(), "size_bytes": size}
        if size > 100 * 1024 * 1024:
            failures.append(record)
        elif size > 50 * 1024 * 1024:
            warnings.append(record)
    return warnings, failures


def scan_forbidden_weights(root: Path, files: list[Path]) -> list[str]:
    findings = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        lowered = rel.lower()
        if any(pattern in lowered for pattern in FORBIDDEN_WEIGHT_PATTERNS):
            if rel.startswith("outputs/") and rel.endswith((".json", ".md", ".csv", ".png")):
                continue
            findings.append(rel)
    return findings


def scan_secrets(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    findings = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if not is_text(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for kind, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"path": rel, "line": line, "type": kind})
    return findings


def scan_hardcoded_paths(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    findings = []
    allowed_windows_example = "docs/reproduction.md"
    for path in files:
        rel = path.relative_to(root).as_posix()
        if not is_text(path):
            continue
        if not rel.endswith((".md", ".py", ".yaml", ".yml", ".json", ".sh", ".ps1")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for kind, pattern in HARDCODED_PATH_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                severity = "warning"
                if rel in {"README.md", "README.zh-CN.md"}:
                    severity = "failure"
                elif rel == allowed_windows_example and kind == "windows_drive":
                    severity = "allowed_windows_example"
                elif rel.startswith("configs/") or rel.startswith("src/"):
                    severity = "failure"
                findings.append({"path": rel, "line": line, "type": kind, "severity": severity})
    return findings


def markdown_files(root: Path) -> list[Path]:
    paths = [root / item for item in MARKDOWN_TARGETS]
    paths.extend(sorted((root / "docs").glob("*.md")))
    return [path for path in paths if path.exists()]


def check_links(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    broken = []
    link_re = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in link_re.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")) or "://" in target:
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                broken.append({"path": path.relative_to(root).as_posix(), "target": target})
    return broken


def check_readme_consistency(root: Path) -> dict[str, Any]:
    en = (root / "README.md").read_text(encoding="utf-8")
    zh = (root / "README.zh-CN.md").read_text(encoding="utf-8")
    checks = {
        "english_links_chinese": "README.zh-CN.md" in en and "简体中文" in en,
        "chinese_links_english": "[English](README.md)" in zh,
        "english_positioning": POSITIONING_EN in en,
        "chinese_positioning": POSITIONING_ZH in zh,
        "no_forbidden_claims": not any(claim in en or claim in zh for claim in FORBIDDEN_CLAIMS),
        "overview_figure": "docs/figures/rankrag_jittor_overview.svg" in en
        and "docs/figures/rankrag_jittor_overview.svg" in zh,
        "main_result_figure": "docs/figures/main_reranking_results.svg" in en
        and "docs/figures/main_reranking_results.svg" in zh,
        "alignment_figure": "docs/figures/pytorch_jittor_alignment.svg" in en
        and "docs/figures/pytorch_jittor_alignment.svg" in zh,
        "readme_analysis_figures": "docs/figures/readme_lora_ablation.svg" in en
        and "docs/figures/readme_error_taxonomy.svg" in en
        and "docs/figures/readme_resource_profile.svg" in en
        and "docs/figures/readme_lora_ablation.svg" in zh
        and "docs/figures/readme_error_taxonomy.svg" in zh
        and "docs/figures/readme_resource_profile.svg" in zh,
        "mentions_formal_lora_rerun": "10k LoRA rerun completed in a unified RTX 4090 D environment" in en
        and "统一的 RTX 4090 D 环境下完成的 10k LoRA 重跑结果" in zh,
        "mentions_500_4044": "500 queries" in en and "4,044" in en and "500 个 query" in zh and "4,044" in zh,
        "cross_encoder_reference": "Cross-Encoder remains the strongest external effectiveness reference" in en
        and "Cross-Encoder 是外部预训练效果参照" in zh,
        "mlp_textcnn_alignment": "lightweight alignment baselines" in en and "轻量对齐基线" in zh,
        "public_lora_label": "Qwen2.5-1.5B LoRA (10k pairs)" in en
        and "Qwen2.5-1.5B LoRA (10k pairs)" in zh,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def check_result_consistency(root: Path) -> dict[str, Any]:
    final = json.loads((root / "outputs/final_results_summary.json").read_text(encoding="utf-8"))
    cost = json.loads((root / "outputs/cost_effectiveness_table.json").read_text(encoding="utf-8"))
    e1 = json.loads((root / "outputs/lora_ablation_results.json").read_text(encoding="utf-8"))
    e2 = json.loads((root / "outputs/lora_scoring_ablation_results.json").read_text(encoding="utf-8"))
    err = json.loads((root / "outputs/error_taxonomy_summary.json").read_text(encoding="utf-8"))
    failures = []
    final_main = {row["method"]: row for row in final["main_reranking_results"]}
    cost_main = {row["method"]: row for row in cost["methods"]}
    for method in ["BM25", "Jittor MLP", "Jittor TextCNN", "Qwen2.5-1.5B LoRA 10k-rerun", "Cross-Encoder"]:
        for metric in ["recall_at_1", "recall_at_3", "recall_at_5", "ndcg_at_5", "mrr", "pairwise_accuracy"]:
            if abs(float(final_main[method][metric]) - float(cost_main[method][metric])) > 1e-12:
                failures.append(f"{method} {metric} mismatch")
    if final["test_query_count"] != 500 or final["pair_count"] != 4044:
        failures.append("test query or pair count mismatch")
    if len(final["lora_data_size_ablation"]) != len(e1["runs"]):
        failures.append("E1 row count mismatch")
    if len(final["scoring_method_ablation"]) != len(e2["runs"]):
        failures.append("E2 row count mismatch")
    if final["error_analysis_summary"]["case_count"] != err["case_count"]:
        failures.append("error summary case count mismatch")
    return {"status": "passed" if not failures else "failed", "failures": failures}


def validate_commands(root: Path) -> dict[str, Any]:
    docs = [(root / "README.md").read_text(encoding="utf-8"), (root / "README.zh-CN.md").read_text(encoding="utf-8")]
    docs.append((root / "docs/reproduction.md").read_text(encoding="utf-8"))
    text = "\n".join(docs)
    paths = sorted(set(re.findall(r"(?<![\w/-])(?:scripts|src)/[A-Za-z0-9_./-]+\.py", text)))
    configs = sorted(set(re.findall(r"configs/[A-Za-z0-9_./-]+\.ya?ml", text)))
    outputs = sorted(set(re.findall(r"(?:outputs|docs|data)/[A-Za-z0-9_./*-]+", text)))
    missing = []
    for item in paths + configs:
        if not (root / item).exists():
            missing.append(item)
    output_warnings = []
    for item in outputs:
        clean = item.rstrip(".,);")
        if "*" in clean:
            continue
        if clean.endswith((".json", ".md", ".csv", ".mmd", ".png", ".jsonl")) and not (root / clean).exists():
            output_warnings.append(clean)
    help_results = {}
    for script in SAFE_HELP_SCRIPTS:
        if script in paths and (root / script).exists():
            help_results[script] = run(root, [sys.executable, script, "--help"], timeout=30)
    help_failures = [script for script, result in help_results.items() if result.get("returncode") != 0]
    return {
        "status": "passed" if not missing and not help_failures else "failed",
        "script_paths": paths,
        "config_paths": configs,
        "missing_paths": missing,
        "output_warnings": output_warnings,
        "help_results": help_results,
        "help_failures": help_failures,
    }


def run_stage_checks(root: Path) -> dict[str, Any]:
    results = {}
    for name, command in STAGE_CHECKS:
        if (root / command[1]).exists():
            results[name] = run(root, command, timeout=120)
        else:
            results[name] = {"returncode": -1, "error": f"missing {command[1]}"}
    return {
        "status": "passed" if all(item.get("returncode") == 0 for item in results.values()) else "failed",
        "results": results,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Final Repository Audit",
        "",
        f"- Status: {audit['status']}",
        f"- Checked commit: `{audit['checked_commit']}`",
        f"- Tracked files: {audit['tracked_file_count']}",
        f"- Large file warnings: {len(audit['large_file_warnings'])}",
        f"- Large file failures: {len(audit['large_file_failures'])}",
        f"- Forbidden weight files: {len(audit['forbidden_weight_files'])}",
        f"- Suspected secrets: {len(audit['suspected_secrets'])}",
        f"- Broken Markdown links: {len(audit['broken_links'])}",
        f"- Missing required files: {len(audit['missing_files'])}",
        f"- Warnings: {len(audit['warnings'])}",
        f"- Failures: {len(audit['failures'])}",
        "",
        "## Stage Checks",
        "",
    ]
    for name, result in audit["stage_checks"]["results"].items():
        lines.append(f"- {name}: returncode={result.get('returncode')}")
    lines.extend(
        [
            "",
            "## README Consistency",
            "",
            f"- Status: {audit['readme_consistency']['status']}",
            "",
            "## Result Consistency",
            "",
            f"- Status: {audit['result_consistency']['status']}",
            "",
            "## Command Validation",
            "",
            f"- Status: {audit['command_validation']['status']}",
            f"- Script paths checked: {len(audit['command_validation']['script_paths'])}",
            f"- Config paths checked: {len(audit['command_validation']['config_paths'])}",
            f"- Help checks run: {len(audit['command_validation']['help_results'])}",
            "",
            "Warnings and detailed findings are in `outputs/final_repository_audit.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    files = tracked_files(root)
    warnings: list[str] = []
    failures: list[str] = []

    large_warnings, large_failures = scan_large_files(root, files)
    forbidden_weights = scan_forbidden_weights(root, files)
    secrets = scan_secrets(root, files)
    hardcoded = scan_hardcoded_paths(root, files)
    broken_links = check_links(root, markdown_files(root))
    missing_files = [item for item in REQUIRED_FILES if not (root / item).exists()]
    readme = check_readme_consistency(root)
    result_consistency = check_result_consistency(root)
    command_validation = validate_commands(root)
    stage_checks = run_stage_checks(root)

    if large_warnings:
        warnings.append("tracked files larger than 50 MiB are present")
    if any(item["severity"] in {"warning", "allowed_windows_example"} for item in hardcoded):
        warnings.append("historical or documented local paths are present as non-fatal findings")
    if command_validation["output_warnings"]:
        warnings.append("some documented output paths use globs or are produced only after full reproduction")
    if large_failures:
        failures.append("tracked files larger than 100 MiB are present")
    if forbidden_weights:
        failures.append("forbidden tracked model/checkpoint files are present")
    if secrets:
        failures.append("suspected secrets are present in tracked text files")
    if any(item["severity"] == "failure" for item in hardcoded):
        failures.append("hardcoded nonportable paths are present in portable code/docs")
    if broken_links:
        failures.append("broken Markdown links are present")
    if missing_files:
        failures.append("required final files are missing")
    if readme["status"] != "passed":
        failures.append("README consistency check failed")
    if result_consistency["status"] != "passed":
        failures.append("result consistency check failed")
    if command_validation["status"] != "passed":
        failures.append("command validation failed")
    if stage_checks["status"] != "passed":
        failures.append("stage check failed")

    audit = {
        "status": "passed" if not failures else "failed",
        "checked_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "tracked_file_count": len(files),
        "large_file_warnings": large_warnings,
        "large_file_failures": large_failures,
        "forbidden_weight_files": forbidden_weights,
        "suspected_secrets": secrets,
        "hardcoded_path_findings": hardcoded,
        "broken_links": broken_links,
        "readme_consistency": readme,
        "result_consistency": result_consistency,
        "command_validation": command_validation,
        "stage_checks": stage_checks,
        "missing_files": missing_files,
        "warnings": warnings,
        "failures": failures,
    }
    output_json = root / args.output_json
    output_md = root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output_md, audit)
    print(json.dumps({"status": audit["status"], "warnings": len(warnings), "failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
