# SFT：全量微调

## 框架和入口

主入口为 `bash scripts/sft.sh`，训练代码为 `scripts/train_sft.py`。
框架是 **PyTorch + Hugging Face Transformers Trainer + Accelerate**。
主入口固定选择 `--finetuning-type full`，不注入 LoRA adapter、不需要 PEFT、
不执行 adapter 合并。veRL 用于后续 GRPO，不负责此处 SFT。

## 数据、Python 和模型

主入口可以直接使用仓库中已经清洗好的数据，无须重新采样，默认读取：

- `data/sft/train.jsonl`：800 条训练轨迹。
- `data/sft/validation.jsonl`：200 条验证轨迹。

两份数据必须存在且非空。训练前检查训练/验证 task_id 不重叠，并排除
`data/evaluation/tasks.jsonl` 中的最终评测任务，不允许使用评测数据调参。

Python 选择顺序为 `PYTHON_BIN` 显式指定、当前 PATH 的 `python`、`python3`，
最后才是仓库 `.venv/bin/python` 回退。脚本会打印 `Using SFT Python: ...`，
自动加入仓库 `src/`，不安装或升级服务器环境。
当前环境需要兼容的 PyTorch、Transformers、Accelerate、DeepSpeed，以及
Ascend 上的 torch_npu/CANN。模型配置必须受当前 Transformers 支持。
veRL 已经跑通不代表这个环境中的 DeepSpeed 已配置好；脚本不会自动安装它。

基模默认目录是仓库旁的 `models/Qwen3.5-9B`，也可通过 `MODEL_ROOT` 指定
该目录的父目录。更换模型时用 `BASE_MODEL` 指定实际本地目录或模型 ID：

```bash
BASE_MODEL="$(pwd)/../models/Qwen3.5-9B" bash scripts/sft.sh
```

如果按[教师采样与离线清洗](data-collection.md)重新生成了数据，需显式选择
新数据。以下示例同时指定解释器：

```bash
PYTHON_BIN=/path/to/prepared/env/bin/python \
BASE_MODEL=/path/to/models/Qwen3.5-9B \
SFT_TRAIN_FILE=outputs/sft-collection/clean/train.jsonl \
SFT_VALIDATION_FILE=outputs/sft-collection/clean/validation.jsonl \
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
| 精度 | BF16 |
| 优化器 | PyTorch AdamW |
| Gradient checkpointing | 开启，full 使用 `use_reentrant=false` |
| Attention | SDPA |
| 并行方式 | 单机 8 卡数据并行 + DeepSpeed ZeRO-2，DP=8、TP=1 |
| DeepSpeed 配置 | `configs/sft_zero2.json`；不做 CPU/NVMe offload |
| 有效全局 batch | `每卡 1 × 8 卡 × 梯度累积 8 = 64` |
| 保存与验证 | 每个 epoch；最多保留 3 个 checkpoint |

一个样本是一条完整购物轨迹。通过模型自身的 chat template 渲染，只对
assistant 回合计算 Loss，包括工具调用与保留的 assistant 文字；
system、user、tool observation 的 labels 为 `-100`。
超长样本整条丢弃并报告数量，不截断工具调用或购买终局。

Qwen3.5 使用 AutoProcessor 的模板与对应多模态模型类。
所有参数设置为可训练，但当前数据只有文本：没有被前向计算使用的视觉分支
不会获得梯度，因此本方案不构成视觉能力训练。默认 ZeRO-2 配置启用
`ignore_unused_parameters=true`；直接调用 Python 入口且不启用 DeepSpeed 时，
DDP 才使用相应的未使用参数检测。全量模式搭配非重入 gradient checkpointing。

## 单机多卡与显存

默认启动单机 8 个训练进程，需要当前进程可见至少 8 张可用设备。
直接执行即可使用默认 Qwen3.5-9B、BF16 和 8 卡 ZeRO-2：

```bash
bash scripts/sft.sh
```

脚本使用同一个 Python 的
`-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node 8`。
Ascend 上先导入 torch_npu、按 `LOCAL_RANK` 绑定 NPU，随后由 Trainer 检查精度配置；
Transformers/Accelerate 接入 DeepSpeed 管理分布式训练。各卡处理不同的数据分片，
ZeRO-2 将梯度和优化器状态分摊到 8 张卡，每张卡仍保留完整模型参数。
默认没有张量并行，TP=1，也没有 FSDP；它不是 DP=4、TP=2 的配置。

默认加载 [`configs/sft_zero2.json`](../configs/sft_zero2.json)。配置不量化模型，
不开启 CPU/NVMe offload，使用 `overlap_comm=true` 尝试重叠计算和通信，
`reduce_bucket_size` 和 `allgather_bucket_size` 均为 50,000,000 个元素。
batch 和精度相关的 `auto` 字段由 Trainer 按实际命令行参数解析。

需要更改卡数时显式覆盖，单卡模式不启动 torchrun：

```bash
SFT_NPROC_PER_NODE=4 bash scripts/sft.sh
SFT_NPROC_PER_NODE=1 bash scripts/sft.sh
```

默认梯度累积为 8，有效全局 batch 为 64。更改设备数或累积步数会改变
全局 batch 和每个 epoch 的优化器更新步数。若需要有效全局 batch=8，
在默认 8 卡启动时传入 `--gradient-accumulation-steps 1`。
最后不足一个完整 batch 的更新可能包含更少样本。

ZeRO-2 降低的是梯度与优化器状态的占用，并不切分模型参数或自动消除长序列激活。
9B 模型、24,576 token 上限在 8 张 64 GB NPU 上能否运行，仍需依据实际样本长度、
算子实现和显存峰值验证；本次仅进行了离线测试，没有实际加载 NPU 模型。
如果仍不足，可先缩短最大长度或改用已验证的 ZeRO-3 配置；缩短长度会导致更多
完整轨迹被剔除，需要检查数据统计。可以覆盖默认 DeepSpeed 配置：

```bash
bash scripts/sft.sh --deepspeed /path/to/validated-zero-config.json
```

ZeRO-3 配置必须设置
`zero_optimization.stage3_gather_16bit_weights_on_model_save=true`，确保保存可直接加载的完整权重。
不要将 LoRA 的显存记录当作全量配置的容量保证。

## ZeRO-2 对速度与精度的影响

ZeRO-2 的目的是节省训练状态显存，不能直接断言它一定比普通 DDP 慢。
速度取决于通信带宽、序列长度、模型计算量和通信重叠情况；它可能增加通信开销，
也可能因减少内存压力而更快。实际应比较相同数据、全局 batch 和精度下的
训练 token/s 与每步耗时。

ZeRO-2 本身不降低数值精度，也不量化参数。保持相同数据、优化器、全局 batch、
精度和更新规则时，分片在理想算术下实现等价的训练更新。但浮点归约顺序、随机性
和优化器实现可能造成差异，不能保证权重逐位相同或最终任务分数完全一致。
默认 BF16 是单独的精度选择，不是 ZeRO-2 带来的精度降级。参见
[Transformers 的 DeepSpeed 说明](https://huggingface.co/docs/transformers/main/deepspeed)。

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
