# Qwen3.5-9B 基线评测复现指南

本指南对应 `Baseline → SFT → GRPO → Evaluation` 的第一步：使用未经本项目 SFT 或 GRPO 的 **`Qwen/Qwen3.5-9B` 官方权重**，在仓库的 **Final-200 Clean** 上完成基线评测，保存逐任务轨迹、确定性指标和 HTML 报告。

这里的“基模”指后续购物后训练的起点，不是把模型名称替换成 `Qwen3.5-9B-Base`。模型名称、模型 revision、思考模式和数值精度都必须记录，后续 SFT/GRPO 使用同一起点和同一评测协议。当前仓库默认示例是 2B；本指南建立的是新的 9B 基线，不能把仓库既有 2B 成绩当作预期结果。

以下是供 Linux 服务器执行的操作说明。编写本文不代表已经下载模型、部署 NPU 服务或完成 200 题评测。

## 1 本步骤的输入与产出

| 项目 | 本次选择 |
|---|---|
| Actor 模型 | `Qwen/Qwen3.5-9B`，固定到一个 Hugging Face commit |
| 购物环境 | 仓库内嵌 ShopSimulator Environment v2.1 |
| 奖励与接口 | Reward v3、observation v2、tool schema v2 |
| 正式数据 | `data/evaluation/tasks.jsonl`，200 个任务 ID |
| 用户协议 | 用户开头一次性给出需求；Agent 与环境多步交互，不追问用户 |
| 基础评测 | 每题一次 rollout，固定 200 题为成功率分母 |
| 输出 | `trajectories.jsonl`、`summary.json`、`report.html` |

正式任务文件只包含任务 ID。评测程序通过 `reset(task_id)` 从环境恢复任务需求；无需自己填写问题或把目标商品喂给模型。

这一步不需要教师模型 API、SFT 数据采集、LoRA、veRL 或强化学习训练。`evaluate_shop_benchmark.py` 与 `build_eval_report.py` 可以直接完成环境奖励和轨迹统计。它们不调用外部 LLM Judge；[完整评测文档](evaluation.md) 中的 Rubric/Judge 四面板属于额外评测层，不能将本步骤的 HTML 报告称为已完成全部 Judge 评测。

## 2 环境与版本

### 2.1 三个运行组件

```text
评测程序                 ShopSimulator                 模型服务
Python 3.12              Python 3.10                  NPU 适配运行时
.venv                    .venv-shopsim                独立容器
        ├──────── HTTP ──→ 5700
        └──────── HTTP ─────────────────────────────→ 8000
```

评测程序和 ShopSimulator 在 CPU 上运行，只有模型服务需要加速卡。三个组件可以在同一台服务器；下文以同机部署为例。无需为了顺序执行 200 题而启用全部 8 张卡。

系统准备 Linux、Bash、Git、gzip、curl、uv；选择容器路线时另需 Docker。主机需要访问 Python 包源、模型仓库和容器仓库，或提前准备对应的离线文件。建议预留至少 80 GB 磁盘给权重、镜像、缓存与轨迹，这是容量规划建议，不是项目测得的最低需求。

### 2.2 评测程序版本

| 包或组件 | 版本 | 来源 |
|---|---|---|
| Python | 3.12 系列，实际补丁号随运行记录保存 | `scripts/setup.sh` 的主环境默认值 |
| shopping-grpo | 0.1.0，配合当前仓库 commit | `pyproject.toml` |
| tqdm | 4.70.0 | 当前 `uv.lock` |
| 模型下载辅助工具 huggingface-hub | 1.25.1，在单独工具环境安装 | 当前 `uv.lock` |

评测 HTTP 客户端使用 Python 标准库 `urllib`；基础评测和 HTML 报告不要求在 `.venv` 安装 `openai`、`torch`、`transformers`、`vllm` 或 `verl`。安装时使用仓库锁文件，不重新解析整套训练依赖。

### 2.3 ShopSimulator 版本

以下直接依赖已经在 [requirements.txt](../environments/ShopSimulator/shop_env/requirements.txt) 固定：

| 包 | 版本 |
|---|---|
| Python | 3.10 系列 |
| beautifulsoup4 | 4.11.1 |
| Flask | 2.1.2 |
| gym | 0.24.0 |
| numpy | 1.26.4 |
| rich | 12.4.4 |
| tqdm | 4.64.0 |
| Werkzeug | 2.1.2 |

这是与主环境分开的虚拟环境，因此两个 `tqdm` 版本并不冲突。该文件没有锁定全部间接依赖；安装完成后必须导出完整 `pip freeze`。之后在同系统和架构复现时使用该完整列表，而不是重新解析间接依赖。

### 2.4 NPU 模型服务的适用范围

仅凭“8 张 64 GB NPU”不能确定厂商、芯片型号和驱动兼容性。以下提供 **昇腾 Atlas A2/A3 的独立容器参考组合**，其他厂商 NPU 应使用对应模型服务栈，不能照搬 `torch_npu`。本指南不修改服务器驱动，也不把该参考组合表述成已在你的服务器验证通过。

vLLM Ascend 官方 `v0.23.0.post1` 的配套版本如下；需要整组匹配，不能把仓库 CUDA 训练环境中的 PyTorch/vLLM 版本混进这个容器：

| 组件 | 版本 |
|---|---|
| vllm-ascend | 0.23.0.post1 |
| vllm | 0.23.0 |
| torch | 2.10.0 |
| torch-npu | 2.10.0.post4 |
| transformers | 5.5.4 |
| CANN | 9.1.0 |
| NNAL | 9.1.0 |
| triton-ascend | 3.2.2，适用于所选 A2/A3 路线 |
| Python | A2 默认镜像基于 Python 3.12；实际补丁号以镜像内版本为准 |

版本依据：[官方配套矩阵](https://docs.vllm.ai/projects/ascend/en/v0.23.0/community/versioning_policy.html)、[安装文档](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)、[发布标签的依赖文件](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0.post1/requirements.txt)、[默认镜像定义](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0.post1/Dockerfile)。完整间接包由镜像 digest 和容器内 `pip freeze` 固定。

镜像按硬件选取：

- Atlas A2：`quay.io/ascend/vllm-ascend:v0.23.0.post1`。
- Atlas A3：`quay.io/ascend/vllm-ascend:v0.23.0.post1-a3`。

驱动和固件须满足该 CANN 版本的厂商要求。已有可用 NPU 模型服务时，可以跳过容器创建，直接对照第 7 节检查接口与推理配置。

不带参数的 `scripts/setup.sh` 会安装 `sft` 和 `grpo` extras，其中包含 `torch==2.11.0`、`vllm==0.25.1`、`verl==0.8.0` 等训练依赖。已有 NPU 训练环境时，使用 `bash scripts/setup.sh --shopsim-only` 仅准备 ShopSimulator，或采用下文的分组件安装命令；不要执行无参数的全量安装。

## 3 获取仓库并固定评测文件

在服务器上使用 Linux Git 检出，避免从 Windows 复制产生 CRLF 换行差异。已有 Linux 仓库时，直接进入根目录即可。

```bash
read -r -p "本仓库 Git URL: " REPO_URL
git -c core.autocrlf=false clone "$REPO_URL" shopping-grpo-longhorizon
cd shopping-grpo-longhorizon
export ROOT="$(pwd)"
mkdir -p outputs/reproduction/baseline-qwen3.5-9b
git rev-parse HEAD > outputs/reproduction/baseline-qwen3.5-9b/repository-commit.txt
git diff --binary > outputs/reproduction/baseline-qwen3.5-9b/working-tree.patch
git status --short > outputs/reproduction/baseline-qwen3.5-9b/working-tree-status.txt
```

以下命令默认从仓库根目录执行。每个新终端进入根目录后重新设置 `export ROOT="$(pwd)"`。如果工作区还有影响运行的未跟踪代码，应一并归档；单独的 commit 和 diff 不会包含未跟踪文件内容。

安装 uv 后，创建仅含项目基础依赖的评测环境：

```bash
# 已安装 uv 时跳过这一行；也可使用 https://docs.astral.sh/uv/getting-started/installation/ 中的方法。
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

uv --version
uv sync --frozen --python 3.12
uv pip freeze --python .venv/bin/python \
  > outputs/reproduction/baseline-qwen3.5-9b/evaluator-requirements.txt
.venv/bin/python --version \
  > outputs/reproduction/baseline-qwen3.5-9b/evaluator-python.txt
uv --version > outputs/reproduction/baseline-qwen3.5-9b/uv-version.txt
.venv/bin/python scripts/evaluate_shop_benchmark.py --help
```

`uv sync` 不加 `--extra sft`、`--extra grpo` 或 `--all-extras`。上述最后一条只展示帮助，不会评测模型。

校验正式任务数、唯一性和文件哈希：

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

path = Path("data/evaluation/tasks.jsonl")
meta = json.loads(Path("data/evaluation/metadata.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(rows) == len({int(row["task_id"]) for row in rows}) == 200
assert digest == meta["sha256"], (digest, meta["sha256"])
print("Final-200 Clean:", len(rows), digest)
PY
```

当前正确哈希为 `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5`。不匹配时先核对代码版本和文件换行，不要通过修改元数据来绕过校验。

## 4 安装并启动 ShopSimulator

### 4.1 独立安装环境依赖

可直接运行 `bash scripts/setup.sh --shopsim-only`，一次完成独立依赖安装、商品校验解压和索引构建，然后跳到 4.3 启动服务。这个模式不会安装训练依赖或修改现有 veRL。以下保留分步执行方式，二者任选其一。

```bash
export SHOP_ENV_ROOT="$ROOT/environments/ShopSimulator/shop_env"
export SHOP_PYTHON="$ROOT/environments/ShopSimulator/.venv-shopsim/bin/python"

uv venv --python 3.10 "$ROOT/environments/ShopSimulator/.venv-shopsim"
uv pip install --no-config --python "$SHOP_PYTHON" -r "$SHOP_ENV_ROOT/requirements.txt"
uv pip check --no-config --python "$SHOP_PYTHON"
uv pip freeze --python "$SHOP_PYTHON" \
  > outputs/reproduction/baseline-qwen3.5-9b/shopsimulator-requirements.txt
"$SHOP_PYTHON" --version \
  > outputs/reproduction/baseline-qwen3.5-9b/shopsimulator-python.txt
```

### 4.2 解压商品并构建搜索索引

商品压缩包已经包含在仓库中；本步骤不需要抓取电商平台数据。下列代码只在目标文件不存在时解压，并校验与仓库安装脚本相同的 SHA-256。

```bash
.venv/bin/python - <<'PY'
import gzip
import hashlib
import shutil
from pathlib import Path

folder = Path("environments/ShopSimulator/shop_env/data")
source = folder / "fine_items_eval_train_all.json.gz"
target = folder / "items_eval_train.json"
expected = "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
if not target.exists():
    temporary = target.with_suffix(".json.preparing")
    with gzip.open(source, "rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    with temporary.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == expected, actual
    temporary.replace(target)
with target.open("rb") as handle:
    actual = hashlib.file_digest(handle, "sha256").hexdigest()
assert actual == expected, actual
print("Product SHA-256:", actual)
PY

"$SHOP_PYTHON" "$SHOP_ENV_ROOT/scripts/build_index.py"
```

应生成 `environments/ShopSimulator/shop_env/search_engine/products.sqlite3`。该环境使用内置多字段 BM25/SQLite 索引；本步骤无需安装 Elasticsearch 或 Java 检索服务。

### 4.3 在终端 A 启动环境

```bash
export SHOPSIM_PORT=5700
export SHOPSIM_ENV_SLOTS=8
export SHOP_MAX_STEPS=35
bash scripts/start_environment.sh
```

保持这个终端运行。环境 API 是 `http://127.0.0.1:5700/api/shop_agent`；访问根路径或假设存在 `/health` 不能作为本项目的健康检查。

## 5 下载固定版本的 Qwen3.5-9B

在另一个终端回到仓库根目录。为下载工具使用单独环境，避免改变基础评测锁文件：

```bash
export ROOT="$(pwd)"
uv venv --python 3.12 .venv-model-download
uv pip install --python .venv-model-download/bin/python huggingface-hub==1.25.1

.venv-model-download/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

model_id = "Qwen/Qwen3.5-9B"
record = Path("outputs/reproduction/baseline-qwen3.5-9b/model.json")
if record.exists():
    metadata = json.loads(record.read_text(encoding="utf-8"))
    assert metadata["model_id"] == model_id
    revision = metadata["revision"]
else:
    revision = HfApi().model_info(model_id).sha
    metadata = {"model_id": model_id, "revision": revision,
                "local_dir": "models/Qwen3.5-9B"}
    record.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
snapshot_download(repo_id=model_id, revision=revision, local_dir=metadata["local_dir"])
print(json.dumps(metadata, indent=2))
PY

uv pip freeze --python .venv-model-download/bin/python \
  > outputs/reproduction/baseline-qwen3.5-9b/download-requirements.txt
```

首次下载会解析并保存一个精确 revision；重跑沿用该 revision。另一台服务器复现时先复制 `model.json`，不要重新解析当时的 `main`。确认目录中包含模型配置、tokenizer、chat template 和全部权重分片。

本次使用原始非量化权重，服务端采用 BF16。不要在建立基线时悄悄换成 AWQ、GPTQ、GGUF、FP8 或其他模型版本。

## 6 在 NPU 上启动模型服务

### 6.1 昇腾 A2 或 A3 容器

以下示例使用逻辑设备 0、TP=1。9B 的 BF16 参数量级约 18 GB，但运行时还需要 KV cache、工作区和其他状态；64 GB 卡能否按该配置启动应通过实际加载检查。此处不承诺特定 NPU 的内存占用或吞吐。

先确认硬件属于适用范围，并且主机驱动支持第 2 节配套运行时。选择正确镜像，再保存其 digest；容器启动时使用 digest，避免标签后续更新改变环境。

```bash
export ROOT="$(pwd)"
npu-smi info > outputs/reproduction/baseline-qwen3.5-9b/npu-info.txt

# Atlas A2；Atlas A3 改用下一行所示的 -a3 镜像。
export SHOPSIM_MODEL_IMAGE="quay.io/ascend/vllm-ascend:v0.23.0.post1"
# export SHOPSIM_MODEL_IMAGE="quay.io/ascend/vllm-ascend:v0.23.0.post1-a3"
docker pull "$SHOPSIM_MODEL_IMAGE"
export SHOPSIM_MODEL_IMAGE_DIGEST="$(docker image inspect \
  --format '{{index .RepoDigests 0}}' "$SHOPSIM_MODEL_IMAGE")"
printf '%s\n' "$SHOPSIM_MODEL_IMAGE_DIGEST" \
  > outputs/reproduction/baseline-qwen3.5-9b/model-image-digest.txt

docker run --rm -it \
  --name shopping-qwen35-9b \
  --network host \
  --shm-size 8g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$ROOT/models:/models:ro" \
  -v "$ROOT/outputs:/outputs" \
  "$SHOPSIM_MODEL_IMAGE_DIGEST" bash
```

上面的 `/usr/local/Ascend` 等是厂商标准运行时挂载路径；如果主机采用不同安装布局，应按实际驱动部署调整。其他设备系列不要仅改镜像后缀照抄 A2/A3 设备挂载。

### 6.2 容器内固定版本并启动

在容器终端 B 执行：

```bash
python --version > /outputs/reproduction/baseline-qwen3.5-9b/server-python.txt
python -m pip freeze > /outputs/reproduction/baseline-qwen3.5-9b/server-requirements.txt
python - <<'PY'
import json
from importlib.metadata import version
from pathlib import Path
import torch
import torch_npu

expected = {"vllm": "0.23.0", "vllm-ascend": "0.23.0.post1",
            "torch": "2.10.0", "torch-npu": "2.10.0.post4",
            "transformers": "5.5.4"}
for name, target in expected.items():
    actual = version(name)
    print(name, actual)
    assert actual.split("+")[0] == target, (name, actual, target)
assert torch.npu.is_available(), "NPU runtime unavailable"
print("NPU devices:", torch.npu.device_count())
settings = {
    "served_model_name": "qwen3.5-9b-baseline", "dtype": "bfloat16",
    "tensor_parallel_size": 1, "max_model_len": 24576, "max_num_seqs": 1,
    "gpu_memory_utilization": 0.90, "language_model_only": True,
    "enforce_eager": True, "generation_config": "vllm",
    "enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder",
    "default_chat_template_kwargs": {"enable_thinking": False},
    "versions": {name: version(name) for name in expected},
}
Path("/outputs/reproduction/baseline-qwen3.5-9b/server-settings.json").write_text(
    json.dumps(settings, indent=2) + "\n", encoding="utf-8"
)
PY

export ASCEND_RT_VISIBLE_DEVICES=0
set -o pipefail
vllm serve /models/Qwen3.5-9B \
  --served-model-name qwen3.5-9b-baseline \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 24576 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --enforce-eager \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  2>&1 | tee /outputs/reproduction/baseline-qwen3.5-9b/model-server.log
```

版本核验失败时，先核对镜像与官方配套表，不要在容器中随意升级某一个核心包。服务启动成功后再继续。

参数作用：

- `qwen3_coder` 把模型生成的工具格式解析为 `message.tool_calls`；只有普通聊天能回复还不足以通过检查。
- `enable_thinking=false` 将本次基线定义为非思考模式。当前评测客户端没有为 Qwen 发送这个参数；其 Python 参数 `thinking=False` 不等于已经修改了 Qwen 的模板默认值，故在服务端显式设置。
- `--generation-config vllm` 避免模型目录的 generation config 隐式改变默认采样行为；评测请求仍显式设置温度、top-p 和生成长度。
- `--max-model-len 24576` 与评测上下文保持一致；不直接使用模型卡的最大窗口。
- `--language-model-only` 对应本项目纯文本 observation；`--enforce-eager` 与单并发用于先建立简单的服务配置，不代表性能最优。

工具解析与非思考模式依据 [Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen3.5-9B)，服务端参数见 [vLLM 0.23.0 CLI](https://docs.vllm.ai/en/v0.23.0/cli/serve/)。这是针对本仓库协议组合的启动配置，需要通过下面的接口和开发任务检查确认在实际设备可用。

`scripts/serve_model.sh` 支持复用当前环境中的 vLLM，也可通过 `PYTHON_BIN` 指定解释器，并透传模型名称之后的服务参数；找不到当前环境的启动程序时才回退到仓库 `.venv/bin/vllm`。传入已有模型目录时直接使用；短名称或 `Qwen/名称` 默认解析到仓库同级的 `models/名称`，可通过 `MODEL_ROOT` 更换根目录，缺少本地模型时不会自动下载。脚本默认模型仍为 2B，不会自动配置 NPU 容器、并行度或非思考模式，因此本步骤保留上面的完整 9B 服务命令以明确评测配置。

## 7 检查接口并运行开发任务

### 7.1 服务连通性和工具格式

在宿主机终端 C，进入仓库根目录执行：

```bash
export ROOT="$(pwd)"
export SHOPSIM_BASE_URL=http://127.0.0.1:5700
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=EMPTY
export SERVED_MODEL_NAME=qwen3.5-9b-baseline

curl --fail --silent --show-error "$LLM_BASE_URL/models"

.venv/bin/python - <<'PY'
import json
import os
from shopping_grpo.environment.context import VllmTextTokenCounter
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.evaluation.rollout import OpenAIChatClient

settings = dict(model=os.environ["SERVED_MODEL_NAME"],
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"])
client = OpenAIChatClient(**settings, timeout=180, max_tokens=512,
                          context_window=24576, observation_token_budget=1536)
messages = [
    {"role": "system", "content": "这是工具接口检查。搜索功能可用。请调用一次 search_products，不要输出普通回答。"},
    {"role": "user", "content": "搜索乳胶枕。"},
]
response = client.complete(messages, SHOP_TOOL_SCHEMAS)
print(json.dumps(response, ensure_ascii=False, indent=2))
calls = response.get("tool_calls") or []
assert len(calls) == 1, "Expected one parsed tool call"
assert calls[0]["function"]["name"] == "search_products"
assert isinstance(json.loads(calls[0]["function"]["arguments"]), dict)
count = VllmTextTokenCounter(**settings)("购物环境接口检查")
assert count > 0
print("chat tokenize:", client.last_context_tokens, "text tokenize:", count)
PY
```

这不会执行购物动作，也不使用正式评测题。它同时检查聊天、工具解析、chat `/tokenize` 和文本 `/tokenize`。若你的服务只兼容 `/v1/chat/completions` 而不支持根路径 `/tokenize`，还不满足当前评测协议。

### 7.2 从开发集选三题

联调使用 GRPO validation 的任务 ID，检查与正式评测集零重叠；不使用 Final-200 Clean 选择 prompt、模型版本或 checkpoint。

```bash
mkdir -p outputs/smoke/qwen3.5-9b
.venv/bin/python - <<'PY'
import json
from pathlib import Path

def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

test_ids = {int(x["task_id"]) for x in read("data/evaluation/tasks.jsonl")}
dev = read("data/grpo/validation.jsonl")[:3]
assert len(dev) == 3
assert not ({int(x["task_id"]) for x in dev} & test_ids)
target = Path("outputs/smoke/qwen3.5-9b/tasks.jsonl")
target.write_text("".join(json.dumps({"task_id": int(x["task_id"])}) + "\n" for x in dev), encoding="utf-8")
print(target)
PY
```

先独立检查环境 reset/search/release：

```bash
DEV_TASK_ID="$(.venv/bin/python -c \
  'import json; print(json.loads(open("outputs/smoke/qwen3.5-9b/tasks.jsonl").readline())["task_id"])')"
.venv/bin/python scripts/smoke_shop_env.py \
  --base-url "$SHOPSIM_BASE_URL" \
  --task-id "$DEV_TASK_ID" \
  --actions 'search[乳胶枕]' \
  --output-dir outputs/smoke/qwen3.5-9b/environment
```

这里的搜索词只用于 API 连通性检查，不要求与该开发任务匹配。随后运行真实 AgentLoop：

```bash
.venv/bin/python scripts/evaluate_shop_benchmark.py \
  --benchmark outputs/smoke/qwen3.5-9b/tasks.jsonl \
  --output outputs/smoke/qwen3.5-9b/trajectories.jsonl \
  --summary outputs/smoke/qwen3.5-9b/summary.json \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --max-steps 35 --temperature 0 --top-p 1 --max-tokens 512 \
  --context-window 24576 --context-safety-margin 512 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768 \
  --observation-search-top-k 20 --timeout 180
```

检查轨迹确实出现 `assistant → tool → assistant → tool`，没有连接、tokenizer 或环境协议错误。开发题不必全部成功；买错、循环或过早结束可以是模型能力问题。若修改了服务配置后需要重新联调，使用新的 smoke 输出目录，避免采集器按任务 ID 跳过旧记录。

## 8 执行正式 200 题基线评测

进入本节前固定模型 revision、服务版本与以下参数。正式测试集不用于反复挑选最优配置。

| 参数 | 值 |
|---|---|
| 每题 rollout 数 | 1 |
| temperature / top-p | 0.0 / 1.0 |
| 每次生成上限 | 512 tokens |
| 最大环境步骤 | 35 |
| 上下文窗口 / 安全余量 | 24576 / 512 tokens |
| 上下文压缩 | 关闭 |
| 搜索 observation 预算 | 1536 tokens |
| 商品详情预算 | 4096 tokens |
| 其他 observation 预算 | 768 tokens |
| 可见搜索候选上限 | 20 |
| 思考模式 / 精度 | 关闭 / BF16 |

开始前保存协议相关文件哈希。运行记录放在忽略的 `outputs/`，不把本地结果写回冻结数据：

```bash
export EVAL_OUTPUT_DIR="$ROOT/outputs/evaluation/baseline-qwen3.5-9b"
mkdir -p "$EVAL_OUTPUT_DIR"
sha256sum \
  data/evaluation/tasks.jsonl \
  data/evaluation/metadata.json \
  environments/ShopSimulator/EMBEDDED_SOURCE.json \
  environments/ShopSimulator/shop_env/configs/environment.json \
  src/shopping_grpo/environment/tools.py \
  src/shopping_grpo/evaluation/rollout.py \
  uv.lock \
  > outputs/reproduction/baseline-qwen3.5-9b/input-sha256.txt

set -o pipefail
.venv/bin/python scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$EVAL_OUTPUT_DIR/trajectories.jsonl" \
  --summary "$EVAL_OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --max-steps 35 --temperature 0 --top-p 1 --max-tokens 512 \
  --context-window 24576 --context-safety-margin 512 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768 \
  --observation-search-top-k 20 --timeout 180 \
  2>&1 | tee "$EVAL_OUTPUT_DIR/evaluation.log"

.venv/bin/python scripts/build_eval_report.py --run-dir "$EVAL_OUTPUT_DIR"
```

在相同环境变量和默认评测参数下，仓库的 `bash scripts/baseline.sh` 也是入口，它会调用 `evaluate.sh baseline` 并生成报告。该入口优先使用显式 `PYTHON_BIN`，其次查找当前 `PATH` 中的 `python` / `python3`，最后回退到仓库 `.venv/bin/python`；它会将仓库 `src` 加入 `PYTHONPATH`，支持仅安装 ShopSimulator 的场景。若要固定使用本节创建的环境，可执行 `PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/baseline.sh`。上面展开命令便于核对实际协议，完成后无需再运行一次封装入口。

当前采集器按任务顺序执行。`trajectories.jsonl` 每题追加一行；`summary.json` 在采集正常结束后生成。耗时取决于模型吞吐、任务步骤和请求耗时，不应根据卡数直接推定。

### 8.1 中断与续跑

同一配置下重跑同一命令会跳过已经存在的 `(task_id, attempt_index)`。这里的“存在”包括失败和错误记录，不代表只跳过成功任务。

- 普通任务失败保留在正式分母中，不为提高分数重复生成。
- 进程中断后，先检查最后一行是否完整、已有任务是否重复，再继续同一配置的剩余任务。
- 服务错误已经写入轨迹时，简单重跑不会自动修复该题。保留原始文件，按明确的基础设施重试规则形成独立修复记录；不要挑选高分重试结果混入原文件。
- 若改变模型、精度、思考模式或评测参数，应使用独立输出目录，并记录协议差异；这些结果不构成严格同配置续跑。

## 9 验收评测产物

先检查每个正式任务恰好有一条记录，并输出核心指标：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from collections import Counter

folder = Path("outputs/evaluation/baseline-qwen3.5-9b")
tasks = [json.loads(s) for s in Path("data/evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
rows = [json.loads(s) for s in (folder / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
expected = {int(t["task_id"]) for t in tasks}
counts = Counter(int(r["task_id"]) for r in rows)
assert len(rows) == 200 and set(counts) == expected
assert all(n == 1 for n in counts.values())
assert summary["expected_tasks"] == summary["completed_tasks"] == 200
assert summary["missing_tasks"] == []
for key in ("strict_successes", "strict_success_rate", "purchase_success_rate",
            "done_rate", "reward_valid_rate", "mean_final_reward", "average_steps",
            "status_counts", "reward_type_counts"):
    print(key, summary[key])
errors = [(r["task_id"], r.get("error"), r.get("release_error"))
          for r in rows if r.get("error") or r.get("release_error")]
print("error records:", len(errors))
for item in errors:
    print(item)
PY
```

上述记录完整性检查通过不等于模型完成了 200 次成功购买。需要分别理解：

| 指标 | 含义 |
|---|---|
| completed_tasks | 已记录的任务数，包含失败与错误 |
| strict_success_rate | 严格成功任务数除以固定 200 |
| purchase_success_rate | 环境标记购买成功的比例，可包含有效替代商品 |
| done_rate | 达到终局的比例，终局不一定是成功 |
| reward_valid_rate | 获得可核验 Reward v3 的比例 |
| mean_final_reward | 已记录轨迹的平均最终奖励 |
| average_steps | 已记录轨迹的平均环境步骤数 |

严格成功必须是完整终局：`status=done`、轨迹和环境 `done=true`、环境 `over=true`、Reward v3、`reward_type=gold_purchase`、`reward_valid=true`、`purchase_success=true`、`termination_reason=gold_purchase`。不要只检查 reward 是否大于零或是否调用过 `buy_now`。实现见 [summary.py](../src/shopping_grpo/evaluation/summary.py)。

最后保留以下文件，作为后续 SFT、GRPO 对照的基线证据：

```text
outputs/evaluation/baseline-qwen3.5-9b/
  trajectories.jsonl
  summary.json
  report.html
  evaluation.log

outputs/reproduction/baseline-qwen3.5-9b/
  repository-commit.txt
  working-tree.patch
  working-tree-status.txt
  model.json
  model-image-digest.txt
  evaluator-requirements.txt
  shopsimulator-requirements.txt
  server-requirements.txt
  server-settings.json
  download-requirements.txt
  evaluator-python.txt
  shopsimulator-python.txt
  server-python.txt
  uv-version.txt
  npu-info.txt
  model-server.log
  input-sha256.txt
```

报告中写明这是 Qwen3.5-9B、非思考、BF16、指定 revision 在 Final-200 Clean 上的成绩。温度为零仍不保证跨硬件、跨推理版本逐 token 完全一致，因此服务器版本和协议记录属于结果的一部分。

## 10 常见问题

| 现象 | 排查方向 |
|---|---|
| 普通聊天正常，轨迹只有文本回复 | 检查工具 parser、`tool_choice=auto` 和实际 `tool_calls`；确认关闭思考。配置正确后仍只回复文本，可能是模型行为失败 |
| 很快耗尽 512 tokens，没有工具调用 | 确认非思考模板确实生效；不要在正式测试中临时提高预算再混合结果 |
| `/tokenize` 返回 404 | 当前服务缺少评测依赖的接口；检查 `/v1` 与根路径区别，不以关闭 token 预算作为同协议替代 |
| 提示上下文过长 | 保留失败记录并分析；开发阶段如需另一个上下文配置，应建立独立协议 |
| `model not found` | 请求中的 `SERVED_MODEL_NAME` 必须与服务的 `--served-model-name` 一致 |
| Flask/Werkzeug 导入错误 | 确认使用 `.venv-shopsim` 及固定依赖，没有混入主环境的包 |
| 找不到商品或搜索索引 | 核对解压输出位置、商品哈希和 `build_index.py` 产物 |
| NPU 算子或驱动报错 | 核对硬件系列、镜像、CANN/NNAL/驱动配套；不使用 CUDA 训练环境覆盖服务容器 |
| 正式输出已经有记录，重跑没重新生成 | 采集器具有按任务跳过逻辑；参见中断与续跑规则 |
| 成功率没有达到仓库展示数字 | 先核对模型规模、数据集版本、协议及基础设施；仓库旧结果不是本次 9B 的目标分数 |

基线完成后，保留这套正式任务与评测设置，第二步再准备 SFT。训练、调参和模型选择始终使用训练集及开发集，不使用 `data/evaluation/tasks.jsonl`。
