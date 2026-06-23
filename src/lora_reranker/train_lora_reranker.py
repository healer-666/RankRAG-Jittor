"""Train a Qwen LoRA relevance classifier for reranking debug runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from src.lora_reranker.lora_utils import build_prompt, load_jsonl, plot_loss_curve, set_seed
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.lora_reranker.lora_utils import build_prompt, load_jsonl, plot_loss_curve, set_seed


DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt = build_prompt(row["query"], row["passage"])
        answer = str(row["label"])
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if len(input_ids) > self.max_length:
            keep_prompt = max(1, self.max_length - len(answer_ids))
            prompt_ids = prompt_ids[:keep_prompt]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def collate_batch(batch: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def available_linear_module_suffixes(model) -> list[str]:
    suffixes = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffixes.add(name.split(".")[-1])
    return sorted(suffixes)


def choose_target_modules(model, requested: list[str]) -> list[str]:
    available = available_linear_module_suffixes(model)
    selected = [name for name in requested if name in available]
    if selected:
        return selected
    fallback = [name for name in ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"] if name in available]
    if fallback:
        print(f"Requested LoRA targets not found. Available linear suffixes: {available}")
        print(f"Using fallback LoRA targets: {fallback}")
        return fallback
    raise RuntimeError(f"No compatible LoRA target modules found. Available linear suffixes: {available}")


def evaluate_loss(model, loader, device: torch.device, max_batches: int = 10) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
            if idx + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / max(len(losses), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into non-empty output directories.")
    return parser.parse_args()


def ensure_output_dir_safe(path: Path, *, label: str, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"{label} already exists and is not empty: {path}. "
            "Use a fresh config output path or pass --overwrite intentionally."
        )


def resolve_model_name(config_model_name: str) -> tuple[str, bool]:
    local_path = os.environ.get("QWEN_LORA_MODEL_PATH") or os.environ.get("QWEN_GENERATOR_MODEL_PATH")
    return (local_path, True) if local_path else (config_model_name, False)


def resolve_git_commit() -> str | None:
    env_commit = os.environ.get("RANKRAG_GIT_COMMIT")
    if env_commit:
        return env_commit
    commit_file = Path("GIT_COMMIT.txt")
    if commit_file.exists():
        return commit_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 42)))

    output_dir = Path(config["output_dir"])
    debug_dir = Path(config.get("debug_output_dir", "outputs/lora_debug"))
    ensure_output_dir_safe(output_dir, label="Adapter output directory", overwrite=args.overwrite)
    ensure_output_dir_safe(debug_dir, label="Debug output directory", overwrite=args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available. CPU LoRA debug training will be slow.")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_load_name, local_model_path_used = resolve_model_name(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_load_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_load_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.config.use_cache = False

    target_modules = choose_target_modules(model, config.get("target_modules", DEFAULT_TARGETS))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_jsonl(config["train_path"])
    valid_rows = load_jsonl(config["valid_path"])
    train_dataset = PairDataset(train_rows, tokenizer, int(config.get("max_length", 256)))
    valid_dataset = PairDataset(valid_rows, tokenizer, int(config.get("max_length", 256)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("per_device_train_batch_size", 1)),
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_batch(batch, tokenizer))

    grad_accum = int(config.get("gradient_accumulation_steps", 4))
    epochs = int(config.get("num_train_epochs", 1))
    max_train_steps = int(config.get("max_train_steps", 0))
    estimated_steps = (len(train_loader) * epochs + grad_accum - 1) // grad_accum
    total_steps = max_train_steps if max_train_steps else estimated_steps
    total_steps = max(total_steps, 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.03)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log_rows = []
    global_step = 0
    samples_processed = 0
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="LoRA debug training")
    epoch_index = 0
    while global_step < total_steps:
        epoch_index += 1
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            samples_processed += int(batch["input_ids"].shape[0])
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            if (batch_index + 1) % grad_accum != 0:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

            raw_loss = float(loss.detach().cpu()) * grad_accum
            row = {"step": global_step, "loss": raw_loss, "lr": scheduler.get_last_lr()[0]}
            if global_step % int(config.get("eval_steps", 50)) == 0 or global_step == 1:
                row["valid_loss"] = evaluate_loss(model, valid_loader, device)
            log_rows.append(row)
            if global_step % int(config.get("logging_steps", 10)) == 0 or global_step == 1:
                print(json.dumps(row, ensure_ascii=False))
            if global_step >= total_steps:
                break
    progress.close()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_log_path = debug_dir / "train_log.jsonl"
    with train_log_path.open("w", encoding="utf-8") as f:
        for row in log_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot_loss_curve(log_rows, debug_dir / "loss_curve.png")

    summary = {
        "model_name": config["model_name"],
        "model_load_name": model_load_name,
        "local_model_path_used": local_model_path_used,
        "git_commit": resolve_git_commit(),
        "output_dir": str(output_dir),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_dtype": str(dtype),
        "train_pairs": len(train_rows),
        "valid_pairs": len(valid_rows),
        "steps": global_step,
        "configured_epochs": epochs,
        "completed_dataset_passes": epoch_index,
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size": int(config.get("per_device_train_batch_size", 1)) * grad_accum,
        "samples_processed": samples_processed,
        "effective_epochs": samples_processed / max(len(train_rows), 1),
        "max_length": int(config.get("max_length", 256)),
        "lora_r": int(config.get("lora_r", 8)),
        "target_modules": target_modules,
        "loss_start": log_rows[0]["loss"] if log_rows else None,
        "loss_end": log_rows[-1]["loss"] if log_rows else None,
        "loss_decreased": bool(log_rows and log_rows[-1]["loss"] < log_rows[0]["loss"]),
        "valid_loss_start": next((row.get("valid_loss") for row in log_rows if "valid_loss" in row), None),
        "valid_loss_end": next((row.get("valid_loss") for row in reversed(log_rows) if "valid_loss" in row), None),
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        ),
        "runtime_sec": time.time() - start_time,
    }
    (debug_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
