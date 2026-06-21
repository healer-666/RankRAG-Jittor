#!/usr/bin/env python3
"""Patch JittorLLMs Qwen2 attention to compute SDPA in FP32.

The script accepts either a JittorLLMs root directory or the target
`modeling_qwen2.py` path. It is intentionally narrow: it checks for the
expected Qwen2 attention snippet, creates a backup, and exits safely if the
patch has already been applied.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


TARGET_RELATIVE = Path("models/qwen2/qwen2_jt/modeling_qwen2.py")
PATCH_MARKER = "Compute attention in FP32, then cast back."

ORIGINAL = """    attn_output = jt.attention.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
"""

PATCHED = """    # Qwen2.5-1.5B can produce non-finite FP16 attention outputs in
    # this JittorLLMs Qwen2 path on the tested environment. Compute
    # attention in FP32, then cast back.
    output_dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()

    attn_output = jt.attention.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.to(output_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        type=Path,
        help="JittorLLMs root directory or qwen2_jt/modeling_qwen2.py path",
    )
    return parser.parse_args()


def resolve_target(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / TARGET_RELATIVE
    if path.name != "modeling_qwen2.py":
        raise SystemExit(f"Expected modeling_qwen2.py or JittorLLMs root, got: {path}")
    if not path.exists():
        raise SystemExit(f"Target file does not exist: {path}")
    return path


def main() -> None:
    target = resolve_target(parse_args().target)
    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        print(f"Already patched: {target}")
        return
    if ORIGINAL not in text:
        raise SystemExit(
            "Target file does not match the expected Qwen2 attention snippet; "
            "no changes were made."
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = target.with_name(f"{target.name}.backup_before_fp32_attention_{timestamp}")
    shutil.copy2(target, backup)
    target.write_text(text.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
