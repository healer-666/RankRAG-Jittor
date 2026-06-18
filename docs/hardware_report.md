# Hardware and Environment Report

Date: 2026-06-18

## System

- OS: Ubuntu 22.04.5 LTS, Linux 6.8.0-124-generic, x86_64
- Python: 3.10.20
- Python path: `/home/dora/桌面/rankrag-jittor-reproduction/.venv-jittor/bin/python`
- pip: 26.1.1
- Project path: `/home/dora/桌面/rankrag-jittor-reproduction`
- Compiler: g++ 11.4.0

## CPU and Memory

- CPU: 12th Gen Intel(R) Core(TM) i9-12900H
- Cores/threads: 14 cores / 20 threads
- RAM: 15 GiB total
- Available RAM during check: 9.6 GiB
- Swap: 15 GiB total, unused during check
- Judgment: suitable for CPU-based feature extraction, BM25 / TF-IDF, MLP reranking, TextCNN-scale experiments, and small subset data processing. It is not ideal for full MS MARCO-scale preprocessing plus local LLM fine-tuning.

## Disk

- Root free space: 26 GiB free on `/`
- Home free space: 67 GiB free on `/home`
- Project size: 1.7 GiB
- Judgment: enough for the current RankRAG-style lightweight reproduction, small public subsets, logs, metrics, and visualizations. Avoid full MS MARCO downloads, multiple large checkpoints, or local multi-GB LLM weight caches.

## GPU

- GPU model: NVIDIA GeForce RTX 3060 Laptop GPU / GA106M RTX 3060 Mobile Max-Q
- VRAM: not confirmed because `nvidia-smi` cannot communicate with the driver in the current session
- Driver files/modules: NVIDIA open kernel module 590.48.01 is present; `lsmod` shows NVIDIA modules loaded
- Driver status: not healthy for CUDA workloads because `nvidia-smi` fails
- CUDA toolkit: `nvcc` is not installed or not on `PATH`
- CUDA environment:
  - `CUDA_HOME`: empty
  - `LD_LIBRARY_PATH`: empty
- `nvidia-smi` status: failed with "couldn't communicate with the NVIDIA driver"
- `ubuntu-drivers devices`: recommends `nvidia-driver-595-open`; multiple NVIDIA driver packages are available

## PyTorch CUDA

- torch version: 2.12.1+cpu
- cuda available: False
- torch CUDA version: None
- device count: 0
- device name: None
- VRAM detected: None
- Judgment: current PyTorch environment is CPU-only. It cannot run GPU LoRA or GPU training without reinstalling a CUDA-enabled PyTorch build and fixing the NVIDIA driver/runtime state.

## Jittor CUDA

- jittor version: 1.3.11.0
- CPU test: passed with `use_cuda=0` on a small 100 x 100 matrix multiplication
- GPU test: skipped
- use_cuda status during safe check: 0
- compiler: `/usr/bin/g++`
- CUDA package download risk: high if Jittor is imported or run with CUDA enabled in the current environment. Because `nvidia-smi` fails and `nvcc` is absent, GPU testing was intentionally skipped to avoid Jittor attempting to fetch a large CUDA/cuDNN package.
- Judgment: Jittor CPU is usable. Jittor GPU is not currently usable as a reliable experiment target.

## LLM Ecosystem Check

- transformers: missing
- datasets: installed, version 5.0.0
- accelerate: missing
- peft: missing
- bitsandbytes: missing
- sentencepiece: missing
- PyTorch LoRA readiness: not ready in the current environment
- Jittor LLM LoRA readiness: risky; Jittor itself works on CPU, but the local CUDA stack is not healthy and the surrounding LoRA ecosystem is not installed

## Feasibility Judgment

### L2 feasibility

BM25 / TF-IDF + MLP + TextCNN reranker:

- Status: feasible
- Reason: CPU, memory, and disk are sufficient for lexical baselines, feature-based reranking, TextCNN-scale neural reranking, case studies, and error analysis on small to medium subsets.

### L3 feasibility

Small-model LoRA / instruction-style ranking:

- Status: risky; not recommended as the main upgrade in the current environment
- Recommended model size if tried later: CPU-only is not recommended. After GPU driver and CUDA-enabled PyTorch are fixed, try at most a small model around 0.5B to 1.5B parameters, preferably with quantization if supported.
- Recommended sequence length: 128 to 256 tokens for early checks
- Recommended batch size: 1, with gradient accumulation if training is attempted later
- Expected risks: NVIDIA driver communication failure, no `nvcc`, CPU-only PyTorch, missing LoRA dependencies, possible Jittor CUDA package auto-download, limited local disk for model caches, and unclear Jittor-side LoRA tooling maturity.

## Recommendation

- Final recommendation: Stay at L2 as the main project upgrade.
- Optional L3: only try as a small exploratory appendix after fixing the NVIDIA driver/runtime and installing a CUDA-enabled PyTorch stack. Do not make L3 the main deliverable on the current environment.
- Recommended L2 next steps:
  - Add BM25 / TF-IDF baseline.
  - Add a TextCNN reranker.
  - Add case study examples for successful and failed rankings.
  - Add error analysis on hard negatives and lexical shortcuts.

## Short Summary

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU / RTX 3060 Mobile Max-Q; VRAM not confirmed because `nvidia-smi` fails.
- PyTorch GPU: not available; installed build is `2.12.1+cpu`.
- Jittor GPU: not available as a safe target now; CPU mode works.
- Disk and memory: enough for L2 and small benchmark work; avoid full large datasets and large model checkpoints.
- L3 recommendation: not recommended as the main upgrade right now.
- If L3 is attempted later: use a very small model, short sequences, batch size 1, and only after fixing NVIDIA driver/CUDA/PyTorch GPU support.
