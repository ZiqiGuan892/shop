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
| 交叉熵 | 128 token 分块，FP32 计算，每块反向时重算 |
| 并行方式 | 单机 8 卡数据并行 + DeepSpeed ZeRO-3，DP=8、TP=1 |
| DeepSpeed 配置 | `configs/sft_zero3.json`；不做 CPU/NVMe offload |
| 有效全局 batch | `每卡 1 × 8 卡 × 梯度累积 8 = 64` |
| 保存与验证 | 每个 epoch；最多保留 3 个 checkpoint |

一个样本是一条完整购物轨迹。通过模型自身的 chat template 渲染，只对
assistant 回合计算 Loss，包括工具调用与保留的 assistant 文字；
system、user、tool observation 的 labels 为 `-100`。
超长样本整条丢弃并报告数量，不截断工具调用或购买终局。

Qwen3.5 使用 AutoProcessor 的模板与对应多模态模型类。
所有参数设置为可训练，但当前数据只有文本：没有被前向计算使用的视觉分支
不会获得梯度，因此本方案不构成视觉能力训练。直接调用 Python 入口且不启用
DeepSpeed 时，DDP 使用相应的未使用参数检测。全量模式搭配非重入
gradient checkpointing，并关闭模型顶层与 `text_config` 中的 `use_cache`。
先前出现的缓存不兼容警告表示模型已自动关闭缓存；显式同步配置可以避免重复
警告，但不能据此认定反向传播 OOM 已解决。

## 单机多卡与显存

默认启动单机 8 个训练进程，需要当前进程可见至少 8 张可用设备。
直接执行即可使用默认 Qwen3.5-9B、BF16 和 8 卡 ZeRO-3：

```bash
bash scripts/sft.sh
```

脚本使用同一个 Python 的
`-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node 8`。
Ascend 上先导入 torch_npu、按 `LOCAL_RANK` 绑定 NPU，随后由 Trainer 检查精度配置；
Transformers/Accelerate 接入 DeepSpeed 管理分布式训练。各卡处理不同的数据分片，
ZeRO-3 将参数、梯度和优化器状态分摊到 8 张卡，需要计算某层时再聚合该层参数。
默认没有张量并行，TP=1，也没有 FSDP；它不是 DP=4、TP=2 的配置。

默认加载 [`configs/sft_zero3.json`](../configs/sft_zero3.json)。为降低峰值占用，
关闭通信重叠与参数预取，并限制聚合后参数的保留。配置不量化模型，不开启
CPU/NVMe offload，也不默认启用 Liger。batch 和精度相关的 `auto` 字段由
Trainer 按实际命令行参数解析。内存相关配置如下，数量单位为参数/元素，而非字节：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `overlap_comm` | `false` | 避免为通信重叠保留额外缓冲，可能增加等待 |
| `reduce_bucket_size` | 5,000,000 | 减小梯度归约桶 |
| `stage3_prefetch_bucket_size` | 0 | 不提前聚合下一批参数 |
| `stage3_param_persistence_threshold` | 10,000 | 限制单个常驻参数张量的大小 |
| `stage3_model_persistence_threshold` | 1,000,000 | 限制模型常驻参数总量 |
| `stage3_max_live_parameters` | 50,000,000 | 限制聚合参数的保留量 |
| `stage3_max_reuse_distance` | 0 | 减少为后续复用保留的参数 |
| `sub_group_size` | 50,000,000 | 减小优化器处理的参数组 |

这些设置限制参数缓存和缓冲，**不是设备显存的硬上限**；最大单层仍需聚合完整
参数，激活、logits、临时算子张量也会占用显存。参数更频繁地释放和重新聚合，
会增加通信或等待。训练参数在 `from_pretrained` 之前建立，使 ZeRO-3 能参与
模型加载初始化；保存时开启 `stage3_gather_16bit_weights_on_model_save=true`，
继续导出后续模型服务和 GRPO 可直接加载的完整权重。

需要更改卡数时显式覆盖，单卡模式不启动 torchrun：

```bash
SFT_NPROC_PER_NODE=4 bash scripts/sft.sh
SFT_NPROC_PER_NODE=1 bash scripts/sft.sh
```

默认梯度累积为 8，有效全局 batch 为 64。更改设备数或累积步数会改变
全局 batch 和每个 epoch 的优化器更新步数。若需要有效全局 batch=8，
在默认 8 卡启动时传入 `--gradient-accumulation-steps 1`。
最后不足一个完整 batch 的更新可能包含更少样本。

保持 BF16、全量训练、每卡 batch=1、梯度累积 8、gradient checkpointing 和
24,576 token 上限，不通过缩短轨迹或量化换取显存。ZeRO-3 不会自动消除长序列
激活或 logits 的峰值，交叉熵需要单独控制中间张量的大小。

主入口默认传入 `--loss-chunk-size 128`，将模型的标准 `ForCausalLMLoss` 替换为
仓库中的分块实现。原生 loss 会先把整段 `[batch, sequence, vocab]` logits 转成
FP32，再计算交叉熵；长序列与大词表会使这一步产生数 GiB 甚至更大的临时张量。
分块实现只在每块内部转换为 FP32，并通过非重入 checkpoint 在反向时重算该块，
避免所有块的 FP32 softmax 中间结果同时保留。训练与 loss-only 验证均使用此路径。

这不是切割训练轨迹：模型仍一次读取完整上下文，每个监督 token 仍计算完整词表的
交叉熵，`-100` 掩码、因果标签移位和原来的损失归一化保持一致。各块的 **sum loss
相加后统一除以分母**，不能取各块 mean loss 的平均值；Trainer 传入
`num_items_in_batch` 时继续使用该分母。交叉熵仍为 FP32，BF16 模型权重不变。
理论目标及梯度相同，浮点求和顺序可能造成微小差异，不能承诺最终权重逐位一致。

默认 128 token、248,320 词表时，一块 FP32 logits 约 0.118 GiB；这只是一个张量
的大小，**不是总显存峰值**。模型仍产生完整 BF16 logits，并需要相应梯度与模型
激活。分块和重算会增加计算/调度开销。此实现经过 CPU loss/梯度回归检查，尚未在
NPU 上验证；完整训练仍需按实际长度、算子和可用显存确认。

调整块大小或恢复原生 loss：

```bash
bash scripts/sft.sh --loss-chunk-size 64
bash scripts/sft.sh --loss-chunk-size 0
```

`scripts/train_sft.py` 直接调用时默认值为 0。若模型使用非标准自定义 loss，启用
分块时会明确报错，不自动覆盖其特殊计算。实现保留模型原生 forward、LM-head 和
DeepSpeed 参数管理；不引入 CUDA/Triton 融合内核依赖。参见
[Transformers loss 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py)
和 [PyTorch checkpoint 说明](https://docs.pytorch.org/docs/2.9/checkpoint.html)。

若服务器已有兼容模型与 NPU 的 Liger 实现，可显式传入
`--loss-chunk-size 0 --liger-kernel`；两种 loss 实现不能同时启用。这不是
默认依赖，也不能保证本地 Liger 包支持当前 Qwen3.5/NPU 组合。覆盖默认
DeepSpeed 配置时使用：

```bash
bash scripts/sft.sh --deepspeed /path/to/validated-zero-config.json
```

仍保留 ZeRO-2 供显存足够时显式选择；它只分片梯度和优化器状态，每卡保留完整参数：

```bash
bash scripts/sft.sh --deepspeed configs/sft_zero2.json
```

自定义 ZeRO-3 配置必须设置
`zero_optimization.stage3_gather_16bit_weights_on_model_save=true`，确保保存可直接加载的完整权重。
不要将 LoRA 的显存记录当作全量配置的容量保证。

## 分片对速度与精度的影响

ZeRO-3 比 ZeRO-2 进一步分片参数，通常需要更多参数聚合通信；当前默认还关闭
通信重叠、减少参数缓存，以优先降低显存峰值，可能降低吞吐。实际速度取决于
通信带宽、序列长度和算子实现，应比较相同数据、全局 batch 和精度下的训练
token/s 与每步耗时，不能仅按分片阶段给出固定加速或减速比例。

ZeRO-2/3 本身不降低数值精度，也不量化参数。保持相同数据、优化器、全局 batch、
精度和更新规则时，分片在理想算术下实现等价的训练更新。但浮点归约顺序、随机性
和优化器实现可能造成差异，不能保证权重逐位相同或最终任务分数完全一致。
默认 BF16 是单独的精度选择，不是 ZeRO 分片带来的精度降级。参见
[Transformers 的 DeepSpeed 说明](https://huggingface.co/docs/transformers/main/deepspeed)。

## 日志、恢复和产物

控制台日志默认只输出到终端；需要保存时：

```bash
set -o pipefail
mkdir -p outputs/sft
bash scripts/sft.sh 2>&1 | tee -a outputs/sft/train.log
```

日志包含实际 Python、微调类型、可训练参数数量、数据保留/剔除数量，
以及训练 loss、步耗时、设备峰值显存等。默认启动时应能看到：

```text
deepspeed_zero_stage=3
loss_mode=chunked_cross_entropy loss_chunk_size=128
model_config={...}
```

`model_config` 打印实际文本模型的 `hidden_size`、`intermediate_size`、层数和词表
大小等；参数数量按 ZeRO 的完整 `ds_numel` 统计。若参数数量明显不符合预期，应
核对目录中的 `config.json` 与权重，不能仅凭目录名确认模型规格。配置和 loss 模式
也会写入训练完成后的 `train_summary.json`。完整输出默认在：

```text
outputs/models/sft-full/
  config.json
  model*.safetensors          完整模型权重，可因大小分片保存
  ...                        tokenizer / processor / chat template 文件
  checkpoint-*/              Trainer checkpoint，可恢复训练
  train_summary.json         参数计数、模式、训练指标与参数
```

使用相同 ZeRO 阶段及兼容训练配置时，可从对应 checkpoint 恢复：

```bash
bash scripts/sft.sh --resume-from-checkpoint outputs/models/sft-full/checkpoint-100
```

**从 ZeRO-2 改为 ZeRO-3 时，不要直接用旧 ZeRO-2 checkpoint 的
`--resume-from-checkpoint`。** 优化器状态的分片格式不能假定可以直接复用。
如果已有完整导出的 Hugging Face 模型，可将其设为 `BASE_MODEL`，重新建立
优化器继续微调；这不会恢复原有优化器、调度器或训练步数，因此不等价于断点恢复。
只有分布式 checkpoint、尚无完整模型时，需先按对应 DeepSpeed/Transformers
版本的导出流程转换；启动脚本不自动转换。

从基模重新开始 ZeRO-3 训练时，可以显式使用新输出目录保留旧产物：

```bash
SFT_OUTPUT_DIR=outputs/models/sft-full-zero3 bash scripts/sft.sh
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
