# SFT：全量微调

## 框架和入口

主入口为 `bash scripts/sft.sh`，训练代码为 `scripts/train_sft.py`。
框架是 **PyTorch + Hugging Face Transformers Trainer + Accelerate**。
主入口固定选择 `--finetuning-type full`，不注入 LoRA adapter、不需要 PEFT、
不执行 adapter 合并。veRL 用于后续 GRPO，不负责此处 SFT。

## 数据、Python 和模型

先完成[教师采样与离线清洗](data-collection.md)，主入口默认读取：

- `outputs/sft-collection/clean/train.jsonl`
- `outputs/sft-collection/clean/validation.jsonl`

两份数据必须存在且非空。训练前检查训练/验证 task_id 不重叠，并排除
`data/evaluation/tasks.jsonl` 中的最终评测任务，不允许使用评测数据调参。

Python 选择顺序为 `PYTHON_BIN` 显式指定、当前 PATH 的 `python`、`python3`，
最后才是仓库 `.venv/bin/python` 回退。脚本会打印 `Using SFT Python: ...`，
自动加入仓库 `src/`，不安装或升级服务器环境。
当前环境需要兼容的 PyTorch、Transformers、Accelerate，以及 Ascend 上的
torch_npu/CANN。模型配置必须受当前 Transformers 支持。

基模默认目录是仓库旁的 `models/Qwen3.5-2B`，也可通过 `MODEL_ROOT` 指定
该目录的父目录。更换模型时用 `BASE_MODEL` 指定实际本地目录或模型 ID：

```bash
BASE_MODEL="$(pwd)/../models/Qwen3.5-4B" bash scripts/sft.sh
```

需要指定解释器或自定义清洗输出时：

```bash
PYTHON_BIN=/path/to/prepared/env/bin/python \
BASE_MODEL=/path/to/models/Qwen3.5-4B \
SFT_TRAIN_FILE=outputs/my-collection/clean/train.jsonl \
SFT_VALIDATION_FILE=outputs/my-collection/clean/validation.jsonl \
SFT_OUTPUT_DIR=outputs/models/sft-full \
  bash scripts/sft.sh --learning-rate 2e-5 --epochs 3
```

不要将 `SFT_TRAIN_FILE` 指向原始轨迹 `raw.jsonl`；应使用清洗后的 messages/tools 格式。
脚本不会自动读取 `.env`，变量需在当前 shell 导出或直接加在启动命令前。

## 训练方式

| 设置 | 主入口默认值 |
|---|---|
| 微调方式 | full，所有模型参数允许梯度 |
| 学习率 | `2e-5` |
| Epoch | 3 |
| 每设备 train/eval batch | 1 / 1 |
| 梯度累积 | 8 |
| 最大序列长度 | 24,576 |
| 精度 | auto：NPU/CUDA 支持时 BF16，否则 FP16；CPU FP32 |
| 优化器 | PyTorch AdamW |
| Gradient checkpointing | 开启，full 使用 `use_reentrant=false` |
| Attention | SDPA |
| 进程数 | 1；可显式设置单机多卡 |
| 保存与验证 | 每个 epoch；最多保留 3 个 checkpoint |

一个样本是一条完整购物轨迹。通过模型自身的 chat template 渲染，只对
assistant 回合计算 Loss，包括工具调用与保留的 assistant 文字；
system、user、tool observation 的 labels 为 `-100`。
超长样本整条丢弃并报告数量，不截断工具调用或购买终局。

Qwen3.5 使用 AutoProcessor 的模板与对应多模态模型类。
所有参数设置为可训练，但当前数据只有文本：没有被前向计算使用的视觉分支
不会获得梯度，因此本方案不构成视觉能力训练。DDP 对这类未使用参数启用检测，
并搭配非重入 gradient checkpointing。

## 单机多卡与显存

默认不自动占用全部设备。单机 8 卡显式启动：

```bash
SFT_NPROC_PER_NODE=8 \
BASE_MODEL="$(pwd)/../models/Qwen3.5-4B" \
  bash scripts/sft.sh --dtype bf16
```

脚本使用同一个 Python 的 `-m torch.distributed.run --standalone --nnodes=1`。
Ascend 上先导入 torch_npu、按 `LOCAL_RANK` 绑定 NPU，再检测 BF16；
分布式通信交给 Transformers/Accelerate。

普通 DDP 会在每卡保留完整模型及优化器状态，增加卡数不等于将显存合并。
全量微调的显存开销与 LoRA 不同，9B 模型和长序列是否能运行需在服务器验证；
本次代码验证没有实际加载 NPU 模型。
若服务器已有兼容的 DeepSpeed，可显式传入自己的配置：

```bash
SFT_NPROC_PER_NODE=8 bash scripts/sft.sh --deepspeed /path/to/validated-zero-config.json
```

脚本不安装 DeepSpeed，也不默认应用分片配置。ZeRO-3 配置必须设置
`zero_optimization.stage3_gather_16bit_weights_on_model_save=true`，确保保存可直接加载的完整权重。
不要将 LoRA 的显存记录当作全量配置的容量保证。

## 日志、恢复和产物

控制台日志默认只输出到终端；需要保存时：

```bash
set -o pipefail
mkdir -p outputs/sft
bash scripts/sft.sh 2>&1 | tee -a outputs/sft/train.log
```

日志包含实际 Python、微调类型、可训练参数数量、数据保留/剔除数量，
以及训练 loss、步耗时、设备峰值显存等。完整输出默认在：

```text
outputs/models/sft-full/
  config.json
  model*.safetensors          完整模型权重，可因大小分片保存
  ...                        tokenizer / processor / chat template 文件
  checkpoint-*/              Trainer checkpoint，可恢复训练
  train_summary.json         参数计数、模式、训练指标与参数
```

训练中断后的恢复示例：

```bash
bash scripts/sft.sh --resume-from-checkpoint outputs/models/sft-full/checkpoint-100
```

训练完成后无需 merge，直接用于模型服务和 GRPO：

```bash
bash scripts/serve_model.sh outputs/models/sft-full
bash scripts/evaluate.sh sft
```

GRPO 默认起点同步为 `outputs/models/sft-full`；若设置了其他 `SFT_OUTPUT_DIR`，
用 `bash scripts/grpo.sh --model /path/to/full-model` 显式传入。
验证 loss 是训练状态信号，最终购物效果需要后续独立评测；已有 LoRA 结果不代表本次全量训练结果。

## 可选 LoRA 入口

`scripts/train_sft.py` 仍支持显式 `--finetuning-type lora`，此时需要 PEFT，
默认学习率为 `1e-4`，产物为 adapter。`--qlora` 只允许和 lora 模式一起使用。
课程和 LoRA 消融入口已经显式传入该模式；`scripts/sft.sh` 固定为 full。
`scripts/sft_curriculum.sh` 仍按它自己的 Pure V4 数据和 manifest 训练，
不会自动使用这次新清洗数据。其最终模型需显式传给后续服务或 GRPO。
