# Jittor 运行环境说明

## Ubuntu 22.04 实际运行状态

本次在 Ubuntu 22.04 上完成了 Jittor 安装、验证、训练、评估和 PyTorch/Jittor 对齐实验。由于系统 Python 3.10 缺少 `python3.10-venv/ensurepip`，且 `sudo apt` 需要交互式密码，最终使用项目内 Conda 环境完成隔离：

```bash
conda create -y --override-channels -c defaults \
  -p /home/dora/桌面/rankrag-jittor-reproduction/.venv-jittor \
  python=3.10 pip setuptools wheel
conda activate /home/dora/桌面/rankrag-jittor-reproduction/.venv-jittor
pip install -r requirements.txt
```

实际版本：

- Python: `3.10.20`
- pip: `26.1.1`
- g++: `11.4.0`
- Jittor: `1.3.11.0`
- PyTorch: `2.12.1+cpu`

Jittor 已通过：

```bash
use_cuda=0 nvcc_path='' python -c "import jittor as jt; jt.flags.use_cuda=0; print(jt.__version__)"
use_cuda=0 nvcc_path='' python -m jittor.test.test_example
```

`test_example` 结果为 `Ran 1 test ... OK`。

## CPU / GPU 说明

本次实验使用 CPU。系统可检测到 CUDA driver，Jittor 默认 import 会尝试下载 `5.61GB` CUDA/cuDNN 包。该下载已中止，后续训练和评估均使用：

```bash
use_cuda=0 nvcc_path=''
```

这样可以强制 Jittor 使用 CPU runtime，避免下载大型 CUDA 组件，也符合本项目轻量复现定位。

## 推荐运行环境

建议在以下环境之一运行 Jittor：

- WSL Ubuntu
- Linux 本机
- Linux 服务器

建议使用单独 Python 虚拟环境，避免污染现有 PyTorch / Conda 环境。

## 本次实际命令

```bash
python scripts/prepare_data.py
use_cuda=0 nvcc_path='' bash scripts/run_train_jittor.sh
use_cuda=0 nvcc_path='' bash scripts/run_eval_jittor.sh
python src/compare_results.py
python src/plot_results.py
```

如果在其它机器上 `pip install jittor` 失败，请参考 Jittor 官方安装文档，并优先在 Linux / WSL 环境下运行。本项目优先验证 CPU 训练与评估流程，不要求 CUDA。

## 已生成产物

本次 Jittor 训练评估已生成：

- `logs/jittor_train.log`
- `outputs/jittor_model.pkl`
- `outputs/jittor_metrics.json`
- `outputs/demo_ranking_result_jittor.json`

并刷新：

- `outputs/metrics_compare.json`
- `outputs/metrics_compare.md`
- `outputs/loss_curve.png`
- `outputs/metrics_compare.png`

Jittor 测试指标与 PyTorch 一致，均为 `1.0000`。这只说明 synthetic benchmark 流程正确和两套实现对齐，不代表真实学术搜索泛化能力。
