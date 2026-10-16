# Data collection

## Goal

The SFT stage needs complete examples of a shopping agent using tools correctly:
searching, opening products, inspecting evidence, choosing options and ending
with a valid purchase. The repository contains the accepted action-only
trajectories, not historical failed collection attempts.

## How the dataset was produced

The current collection used ShopSimulator Environment v2.1, Reward v3 and
`deepseek-v4-flash` as the teacher. It produced 2,498 raw trajectories.
Every trajectory executed its actions in ShopSimulator during collection. The
saved result was accepted only when Environment v2.1 returned a valid Reward v3
gold purchase; no second model judged whether the trajectory succeeded.

Collection audit:

| Item | Value |
|---|---:|
| Raw trajectories | 2,498 |
| Accepted trajectories | 1,026 |
| Acceptance rate | 41.07% |
| Frozen rows used | 1,000 |
| Unused accepted rows | 26 |

The frozen subset was split into 800 training and 200 validation rows. The two
splits are task-disjoint and also have zero task-ID overlap with GRPO and the
Final-200 evaluation set.

## Frozen deliverables

| File | Rows | SHA-256 |
|---|---:|---|
| `data/sft/train.jsonl` | 800 | `8c3a6ff0033f6ea672af609891e747d60652ddc17e8d3c8eacb19e9d96dd9477` |
| `data/sft/validation.jsonl` | 200 | `9525cc2fb04a1d8d38ae2db959397da908dde3fea766f580fdcf77d1239533cc` |

Raw teacher responses are intentionally not committed; their collection path
is retained in `data/sft/metadata.json` as provenance.

## 重新采集：两个独立步骤

当前入口明确分为两个脚本：

| 步骤 | 脚本 | 网络和服务要求 | 主要产物 |
|---|---|---|---|
| 1. 教师采样 | `scripts/collect_sft_data.py` | ShopSimulator + 教师模型 API | 原始 `raw.jsonl`，包括成功与失败 |
| 2. 离线清洗 | `scripts/clean_sft_data.py` | 无网络要求，不需要模型服务 | 验收审计、训练集、验证集 |

`collect_sft_data.py` 不再自动清洗，也不再接受 `--build-only`、
`--target-accepted`、`--validation-ratio`；清洗参数放在第二个脚本。
第一步按任务数和每题尝试次数采样，不会为了凑成功样本而删除失败轨迹。

### 准备

在仓库根目录执行以下命令。使用已配置好的 Python >=3.10 即可；这两个
脚本会加载仓库的 `src/`，不要求 `.venv`，无需重装 veRL、vLLM 或 PyTorch。
采样、清洗本身使用 Python 标准库。

如果 ShopSimulator 已启动，复用现有服务；否则在另一个终端运行：

```bash
bash scripts/start_environment.sh
```

在采样终端设置教师模型接口。地址和模型名称必须与实际供应商匹配：

```bash
export SHOPSIM_BASE_URL="http://127.0.0.1:5700"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_MODEL="deepseek-v4-flash"
read -rsp "Teacher API key: " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

教师接口需要支持项目使用的工具调用协议。若使用本地模型服务，可设置它的
`/v1` 地址和 served-model-name，并将 `OPENAI_API_KEY` 设为 `EMPTY`。
使用外部教师时，不要求先启动学生 Qwen 的 vLLM 服务。
脚本不会自动读取 `.env`，上述变量需要导出到当前 shell。

### 第一步：只采样原始轨迹

以下示例从环境任务 ID `[0, 5000)` 中排除保留任务，再选前 3,000 个任务，
每题尝试一次。任务指令每次都由 `env.reset(task_id)` 返回，完全重新调用
教师模型，不读取 `data/sft/` 或 `data/sft_pure_v4/` 的既有示范。
任务 ID 必须属于所运行的固定 ShopSimulator 版本；范围右端不包含在内。

```bash
set -o pipefail
mkdir -p outputs/sft-collection/raw
python -u scripts/collect_sft_data.py \
  --task-range 0 5000 \
  --limit 3000 \
  --attempts-per-task 1 \
  --workers 4 \
  --progress-interval 15 \
  --output-dir outputs/sft-collection/raw \
  2>&1 | tee -a outputs/sft-collection/raw/console.log
```

先验证少量样本时，可用 `--limit 10 --workers 1`，同时改用新的输出目录，
例如 `outputs/sft-check/raw`。并发数不要超过环境的空闲 slots 或教师接口并发额度。
`--limit` 是排除保留任务后的唯一任务数上限；总尝试数是任务数乘
`--attempts-per-task`，不是清洗后保证得到的成功轨迹数。
默认 `temperature=0`，重复尝试可能得到相同动作；需要多样性时可以显式设置
`--temperature 0.7 --attempts-per-task 2`，这会改变采样配置。

也支持用 `--tasks path/to/tasks.jsonl` 替代 `--task-range`，每行至少含：

```json
{"task_id": 1535}
```

只读取 ID；输入文件中的旧 `messages`、`prompt` 或答案不会被用作教师上下文。
不要用 `data/grpo/train.jsonl` 作为新 SFT 的任务源：当前 GRPO 任务池会被排除。
两个脚本都强制排除以下文件中的 ID，缺少文件会直接报错：

- `data/evaluation/tasks.jsonl`
- `data/grpo/train.jsonl`
- `data/grpo/validation.jsonl`

需要保留其他任务时，可在两步命令中追加 `--exclude-tasks path/to/reserved.jsonl`，
此参数可以重复；它不会取代上述固定排除项。

采样默认每 15 秒显示进度、当前任务 ID 和粗略 ETA，并在每条任务写入后输出状态。
每完成一次任务尝试就追加一条原始记录，包括失败记录。
重复运行相同命令会跳过已经写入的 `(task_id, attempt_index)`；更换教师、
采样参数或任务集合时必须使用新输出目录，避免混合不同来源。
并发数和打印间隔可以在续跑时调整。

按 Ctrl+C 会停止派发新任务，等待当前任务结束并保存、释放环境后退出；
正在等待模型的任务可能需要等到请求超时。不要同时向同一个输出目录启动两个采集进程。
服务故障会保留错误轨迹并停止继续派发，已在运行的任务仍会保存。
已记录的失败也会被跳过；若需重试这些题，提供对应 ID 列表并使用新的输出目录。

### 第二步：离线验收、清洗和划分

第一步退出后执行；此时可以关闭教师模型与 ShopSimulator 服务：

```bash
set -o pipefail
mkdir -p outputs/sft-collection/clean
python -u scripts/clean_sft_data.py \
  --raw outputs/sft-collection/raw/raw.jsonl \
  --output-dir outputs/sft-collection/clean \
  --validation-ratio 0.2 \
  --seed 42 \
  2>&1 | tee -a outputs/sft-collection/clean/console.log
```

清洗重新执行确定性验收，不调用模型，也不回放环境。通过条件包括完整终局、
实际购买、Reward v3 `gold_purchase`、`reward_valid=true`、
`purchase_success=true`、无采集/释放错误，以及已执行动作通过合法性检查。
每个任务保留原始文件中第一条通过验收的轨迹，其余通过轨迹记为 `duplicate_task`。
再次排除评测、GRPO 和额外保留任务；审计中的 `held_out_excluded` 统计这些排除项的总和。

清洗移除独立的 `reasoning_content` 等审计字段、被守卫阻止的调用及其对应消息，
将终局奖励文本替换为“购买已完成。”，保留正常 assistant 文字和工具调用。
然后按 task_id 稳定划分，保证同题不同时进入训练和验证。
`--validation-ratio 0.2` 是约 80/20 划分，不保证固定得到 800/200 条；
样本数量取决于本次成功数，且一个任务多次成功只算一条。
样本太少时可能没有验证集；零条通过时生成拒绝审计并以非零状态退出。

同一份 raw 文件、相同参数可重复清洗，派生文件会重新生成；raw 文件内容保持不变。
清洗不需要 API key，也不读取已有的 SFT 训练文件。

```text
outputs/sft-collection/
  raw/
    raw.jsonl              所有原始任务轨迹及错误、终局奖励
    tasks.jsonl            本次实际选择的任务 ID
    collection_config.json 教师和采样配置、任务列表、排除清单哈希；不含 API key
    console.log            仅在使用上述 tee 命令时生成
  clean/
    accepted.jsonl         通过验收且去重后的原始轨迹，保留审计信息
    rejected.jsonl         任务 ID、轨迹 ID 和拒绝原因
    reject_stats.json      通过、质量拒绝、保留任务排除、重复排除等统计
    sft.jsonl              清洗后全部训练格式轨迹
    train.jsonl            训练集
    validation.jsonl       验证集
    metadata.json          数量、划分配置、来源、排除清单和文件 SHA-256
    console.log            仅在使用上述 tee 命令时生成
```

查看结果：

```bash
wc -l outputs/sft-collection/raw/raw.jsonl
python -m json.tool outputs/sft-collection/clean/reject_stats.json
wc -l outputs/sft-collection/clean/{train,validation}.jsonl
```

后续 SFT 应将 `--train`、`--validation` 指向上述新文件。
`scripts/sft.sh` 仍固定使用 `data/sft/`，课程训练仍使用它自己的数据和 manifest；
单独完成这两步不会自动切换训练入口，也不会覆盖仓库冻结数据。
原始响应和采集产物保存在被 Git 忽略的 `outputs/`，不要提交 API key 或原始响应。

## What a training row contains

Each JSONL row is a chat trajectory with:

- the shopping instruction;
- assistant tool calls;
- ShopSimulator tool observations;
- the final terminal action;
- metadata tying the row to Environment v2.1 and Reward v3.

During SFT, user and tool tokens are masked. Loss is computed on assistant
turns, including tool calls and any retained assistant text. See [SFT](sft.md)
for the exact training recipe.
