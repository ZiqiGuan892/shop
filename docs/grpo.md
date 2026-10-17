# GRPO with veRL

## Purpose and inputs

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

The default initial policy is the complete **Qwen3.5-9B SFT model** saved by
`scripts/sft.sh` to `outputs/models/sft-full`. An adapter-only directory or an
unexported DeepSpeed checkpoint is not a complete model. The launcher does not
silently substitute a base model if the SFT output is missing. Set
`GRPO_MODEL_PATH` or pass `--model` if SFT used another output directory:

```bash
bash scripts/grpo.sh --model outputs/models/sft-full-zero3 --check-only
```

The other inputs remain:

- Train set: `data/grpo/train.parquet` (1,000 tasks).
- Validation set: `data/grpo/validation.parquet` (50 tasks).
- Environment: ShopSimulator Environment v2.1, observation v2, tool schema v2.
- Reward: Reward v3. Strict success requires a complete `gold_purchase`
  terminal result with `reward_valid=true`.

Data hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).
Preflight checks train/validation task isolation and excludes evaluation tasks.

## Reuse the server environment

Activate the Python environment in which the server's veRL and vLLM already
work. `scripts/grpo.sh` uses the shared interpreter selection rule:
`PYTHON_BIN`, then `python` / `python3` on `PATH`, then the repository `.venv`
as a fallback. It does not install or replace torch, torch_npu, vLLM, Transformers
or veRL. The launcher adds this checkout's `src` to `PYTHONPATH`.

```bash
export PYTHON_BIN="$(command -v python)"
```

When the correct veRL checkout is already installed in that environment, no
extra path is needed. Otherwise, set `VERL_ROOT` to the **checkout containing
`verl/__init__.py`**, not its inner `verl` package directory:

```bash
export VERL_ROOT=/path/to/verl-checkout
```

The directory must contain `$VERL_ROOT/verl/__init__.py`. A source checkout can
work with `python -m verl.trainer.main_ppo` without ever having been registered
with pip: a launcher running from the checkout root makes that source visible
to Python. Switching to the Shopping directory removes that implicit search
path. `VERL_ROOT` makes the selection explicit for preflight and training.

`pip show verl` reports distribution metadata, not source importability. Both
preflight and the trainer installer read the selected package's
`__version__` (or its own `version/version` file), print its source path, and
still enforce the supported version and APIs. They do not require veRL pip
metadata, and do not borrow version metadata from another installed checkout.
Other required dependencies are still checked in the selected environment.

Verify the path and imports before training:

```bash
test -f "$VERL_ROOT/verl/__init__.py"
PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -c 'import verl; print(verl.__file__); print(verl.__version__)'
bash scripts/grpo.sh --check-only
```

A missing dependency during `import verl` is reported with its original error;
it is distinct from a missing source path. No reinstall of the working NPU
stack is needed merely because `pip show verl` has no entry.

The selected runtime must report **`0.9.0.dev`** or its equivalent spelling
`0.9.0.dev0`; local build suffixes such as `+ascend` are accepted. Release 0.8,
stable 0.9, other development revisions and 0.10 are not selected implicitly.
This is a source-version requirement, not a request to install or replace the
server's veRL. The `grpo` extra no longer installs veRL from PyPI; supply the
prepared server checkout through its existing import path or `VERL_ROOT`.

Preflight requires Python 3.10 or newer, the AgentLoop/Tool APIs used by the adapter,
compatible installed dependencies and working NPU/HCCL support. It prints the
selected versions and source paths instead of requiring the CUDA package pins
or Python 3.12 from the clean-install recipe. A working unrelated veRL training
job does not establish that this Shopping AgentLoop, Qwen3.5 FSDP training,
LoRA weight synchronization and dynamic sampler are supported in a custom fork.
Passing preflight is necessary but is not an end-to-end NPU training validation.

The NPU recipe sets `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="1"`
in Ray's worker runtime environment. Ray then leaves the runtime device mask
alone, and veRL's native worker setup explicitly selects the NPU assigned by
Ray. The launcher preserves the platform's `ASCEND_VISIBLE_DEVICES` and any
existing `ASCEND_RT_VISIBLE_DEVICES`; do not copy physical card IDs from the
platform mask into `LOCAL_RANK`. It removes inherited torchrun rank and
rendezvous variables from the child environment so Ray/veRL can establish
their own ranks. See the [Ray NPU manager](https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/npu.py)
and [veRL worker setup](https://github.com/volcengine/verl/blob/main/verl/single_controller/base/worker.py).

Before each NPU worker loads its model, `GRPO NPU worker binding:` logs the
PID, global rank, Ray-assigned NPU ID, local rank, current logical device and
both visibility masks. Ray log deduplication defaults to off so these lines
remain visible; an explicit shell `RAY_DEDUP_LOGS` setting is preserved.
In this NOSET mode the assigned ID, local rank and
current device must agree; a mismatch stops the worker before model loading.
Global rank need not equal the device ID. The hook observes veRL's selection
and does not choose another device. An HCCL error reporting the **same physical
device ID for two ranks** indicates a binding conflict; `LOCAL_RANK=0` alone
does not establish that conflict in other Ray visibility modes. `--check-only`
checks the launch policy and worker API, while actual worker binding is checked
only when training starts.

`configs/grpo.yaml` explicitly sets `trainer.use_v1: false`, overriding the
server's inherited V1 default. In the
[upstream 0.9 launcher](https://github.com/volcengine/verl/blob/v0.9.0/verl/trainer/main_ppo.py),
this selects `main_ppo_v0.TaskRunner`, which constructs `RayPPOTrainer`; the
Shopping sampling and TRACE hooks target that class. This selects a trainer
within the existing veRL installation; it does not downgrade veRL. Preflight
still rejects an explicit `trainer.use_v1=true` override.

The development fork may differ from that release. Selecting this backend
resolves the V1 configuration conflict; the rest of preflight checks the
available APIs and the directly installed trainer described below. Passing
these checks does not establish an end-to-end NPU training result.

`scripts/setup.sh` without arguments prepares the other pinned training dependencies
and ShopSimulator. It does not install veRL or replace its trainer automatically,
and does not certify that a GRPO environment is ready. It is not needed for an
already working training environment. If only the simulator is missing, use
`bash scripts/setup.sh --shopsim-only`.

### Direct trainer integration

[`integrations/verl/ray_trainer.py`](../integrations/verl/ray_trainer.py) is the
server's supplied **0.9.0.dev** trainer with Shopping dynamic sampling integrated
directly into its training loop. Sampling helpers remain in
`src/shopping_grpo/training/grpo/`. The former 0.8 patch file, patch application
script and patch-marker requirement have been removed. Dynamic sampling itself
stays enabled with the original training recipe.

Stop any GRPO job using this veRL checkout before replacing its source. From
the Shopping repository root, select the same Python and `VERL_ROOT` as training:

```bash
export PYTHON_BIN="$(command -v python)"
PYTHONPATH="${VERL_ROOT:+$VERL_ROOT:}${PYTHONPATH:-}" \
  "$PYTHON_BIN" scripts/install_verl_trainer.py
```

The installer locates `verl/trainer/ppo/ray_trainer.py` through the actual
imported `verl` package, verifies the supplied server original and repository
replacement hashes, and checks Python syntax without importing the training
loop. It saves the original beside the file as
`ray_trainer.py.shopping-grpo-trainer.orig`, then atomically replaces only
`ray_trainer.py`. It preserves file permissions, never reinstalls packages, and
does not start training. Repeating installation with identical content is safe.
Hashes normalize CRLF to LF and trailing empty lines to one final newline to
support Windows checkouts and pasted source files; other content changes are
not ignored. The backup preserves the original bytes. Unknown source files or
corrupted backups are rejected.

Verify deployment and the complete launch configuration:

```bash
PYTHONPATH="${VERL_ROOT:+$VERL_ROOT:}${PYTHONPATH:-}" \
  "$PYTHON_BIN" scripts/install_verl_trainer.py --check
bash scripts/grpo.sh --check-only
```

Preflight checks the **deployed trainer content hash**, along with helper APIs,
dynamic-sampling settings, environment, dataset isolation and device topology.
There is no old-patch marker check to bypass. A mismatched hash means the
selected file differs from the reviewed implementation; inspect the printed
source path rather than disabling dynamic sampling.

To undo this source replacement, stop jobs using the checkout and restore the
verified backup with the same interpreter/source selection:

```bash
PYTHONPATH="${VERL_ROOT:+$VERL_ROOT:}${PYTHONPATH:-}" \
  "$PYTHON_BIN" scripts/install_verl_trainer.py --restore
```

Restore refuses to overwrite unknown subsequent edits. Restoring the original
removes this Shopping training-loop integration, so the default Shopping
preflight will require installation again. The rest of the server's veRL
installation is unchanged. Full NPU training still requires server validation.

## Run

Keep ShopSimulator running in a separate terminal:

```bash
bash scripts/start_environment.sh
```

Its default URL is `http://127.0.0.1:5700`; use `SHOPSIM_BASE_URL` or `--env-url`
for another address. Stop your baseline/SFT `vllm serve` process before training
to release the NPUs. GRPO creates its own vLLM rollout workers and does not use
the evaluation model service at `LLM_BASE_URL`.

Inspect the selected model, interpreter, source path and final command:

```bash
bash scripts/grpo.sh --dry-run
```

This checks input paths and prints the launch plan without importing the
training runtime or starting Ray. Next run the actual preflight:

```bash
bash scripts/grpo.sh --check-only
```

`--check-only` checks the selected runtime, device availability, data and
integration contract without starting training, starting Ray or loading model
weights. It leaves the output directory untouched. It does not replace a live
rollout test; keep the simulator available for the subsequent training run.

Train after preflight succeeds:

```bash
bash scripts/grpo.sh
```

Ray remains required by veRL even on one machine. The launcher selects a local
Ray runtime with `nnodes=1`; it does not wait for a second node and does not run
`ray start` or `ray stop` against other jobs.

The server's `main_ppo.py` converts the Ray initialization configuration with
`OmegaConf.to_container` without resolving interpolation. The eight-NPU recipe
therefore sets `ray_kwargs.ray_init.resources.NPU` to the number `8`, and the
launcher passes the actual `PYTHONPATH` as a quoted Hydra argument to both
preflight and training. Set additional import directories with shell
`PYTHONPATH` or `VERL_ROOT`; do not override the worker path separately.
Preflight checks the unresolved container that Ray will receive, rejecting
`${...}` placeholders, string resource quantities and a different worker path.
This does not change sampling budgets or model training parameters.

The output directory must be new or empty. To retain an earlier run or a failed
launch's log, select a new directory using `--output`, for example:

```bash
bash scripts/grpo.sh --output outputs/models/grpo-9b
```

Starting directly from `../models/Qwen3.5-9B` is possible only when explicitly
requested with `--model ../models/Qwen3.5-9B`. This skips SFT and changes the
initial policy; it is not the default Baseline → SFT → GRPO comparison.

## Parallelism and training recipe

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
The launch defaults target a single machine with **8 Ascend NPUs** and the 9B
SFT policy; they have not been validated by a training run on your server.

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Actor backend | FSDP across 8 devices; parameters are sharded |
| Rollout tensor parallelism | TP=2, giving 4 rollout replica groups on 8 devices |
| Precision | BF16 |
| Train / validation prompt batch | 2 / 2 |
| Rollouts per training prompt | 4 (8 trajectories for a retained global update) |
| PPO mini batch / per-device micro batch | 2 prompts / 1 trajectory |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Policy learning rate / warmup ratio | `1e-6` / `0.03` |
| GRPO LoRA rank / alpha / target modules | 16 / 32 / `all-linear`, merge enabled |
| Prompt / response / total length limit | 4,096 / 20,480 / 24,576 |
| PPO clipping, low / high | 0.2 / 0.2 |
| Maximum training steps / epochs | 500 / 50 |
| Save / validation frequency | 50 / 50; validation before training enabled |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement | disabled; entropy coefficient remains 0 |
| Actor activation memory | native veRL activation offload + gradient checkpointing |
| Dynamic sampling | enabled; at most 3 generation batches and 10 consecutive skipped updates |

The prompt batch is global: 8 devices do not multiply it by another factor of
8. Dynamic sampling may generate additional trajectories to retain groups with
useful reward differences. Validation uses one deterministic rollout per prompt.
The original learning rate, loss settings, batch sizes and sampling budgets
above remain unchanged. Model selection, device topology and execution settings
are adjusted for the 9B NPU run.

Dynamic sampling needs fresh environment rollouts for each generation batch.
Cached rollout replay must remain disabled: both `skip.rollout.enable` and
`actor_rollout_ref.rollout.skip.enable` must be false when present in the selected
veRL configuration. Replaying a cached batch under the same optimizer step
would prevent additional sampling from supplying fresh task groups. This
requirement does not change the rollout count, temperature, learning rate or
other training parameters above.

GRPO retains the project's **LoRA** recipe even though the preceding SFT updates
all model parameters. Its FSDP actor is separate from the SFT Trainer's
DeepSpeed ZeRO-3 configuration; `configs/sft_zero3.json` does not configure GRPO.
Actor and rollout parallelism serve different phases: FSDP shards the actor
across 8 devices, while TP=2 distributes a rollout model across each pair.

For memory and NPU runtime behavior, actor parameter/optimizer offload and
gradient checkpointing remain enabled, and **activation offload is enabled**.
veRL moves saved layer activations to host memory and restores them for backward;
when combined with checkpointing, its offload implementation supplies the
checkpoint wrapper. This adds host-memory and transfer costs and can reduce
training throughput. The native implementation skips offload with a warning
if it finds fewer than three wrapped layers; inspect that warning if changing
the model's FSDP wrapping policy. FSDP explicitly reshards parameters after forward and
leaves forward prefetch disabled. Parameter/optimizer offload alone does not
eliminate saved activations or the loss's full-vocabulary intermediate tensors.
See [veRL activation offload](https://github.com/volcengine/verl/blob/main/verl/utils/activation_offload.py).

`actor.calculate_entropy=false` removes the entropy statistic and its graph.
The entropy coefficient was already zero, so this preserves the mathematical
GRPO objective. With entropy enabled, the native loss can still execute
`policy_loss -= 0 * entropy_loss` and backpropagate through the large vocabulary
softmax; a zero coefficient does not prune that graph. The `actor/entropy_loss`
metric is consequently absent. Preflight requires nonzero entropy-coefficient
overrides to explicitly enable entropy calculation as well, making their
additional memory cost visible in the launch configuration. See
[veRL policy loss](https://github.com/volcengine/verl/blob/main/verl/workers/utils/losses.py).

These settings target actor forward/backward memory while preserving prompt,
response and total sequence limits, rollout count, learning rate, clipping,
LoRA rank and effective batch. Per-device micro batch is already one; lowering
token limits used only for dynamic batching would not split that single long
trajectory in this static-batch recipe. Preflight prints the selected entropy,
activation-offload, checkpointing and FSDP settings and verifies the native
offload API without starting a model. It cannot certify the peak memory of a
particular server build or trajectory; model logits and their gradients still
require device memory.

The rollout memory utilization is 0.45,
with at most 8 sequences and eager execution enabled. Eager execution avoids
the NPU graph-capture stream/SQ-CQ failure seen previously; it may reduce
throughput and is not a guarantee against memory pressure. Actor
`use_torch_compile=false` also avoids relying on CUDA-oriented compilation.
The actor's model loading dtype is explicitly BF16; this differs from veRL's
FP32 loading default and reduces storage for the large frozen base. It does not
enable quantization. Sequence length and sampling count are preserved, but
these hardware/dtype changes do not promise bitwise-identical results.

Advanced Hydra overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

Use `--model`, `--train-data`, `--val-data` and `--output` for paths. The
equivalent Hydra path overrides are rejected so preflight, logging and training
cannot refer to different files.

## Logs and checkpoints

For the default `--output outputs/models/grpo`, the launcher writes:

- `train.log`: launch details and merged training stdout/stderr,
  streamed to both the console and file with unbuffered Python output.
- `launch.json`: selected paths, interpreter and command.
- `training_diagnostics.jsonl`: generated trajectories and optimizer metrics.
- veRL checkpoints and Hydra artifacts beneath the same run directory.
- Each `global_step_N/` checkpoint also contains `data.pt` and
  `shopping_sampler_state.json` for dynamic-sampling resume, alongside the
  actor checkpoint.

Preflight prints to the console before any output files are created. A failed
preflight leaves the output directory available for retry; a failed training
run retains its log and requires a new output directory for a fresh run.

Follow the console log with:

```bash
tail -f outputs/models/grpo/train.log
```

`generation_batch` diagnostic records contain every generated rollout, its
public tool sequence, terminal result, reward breakdown, Guard rejection
reasons and group keep/drop decision. `optimizer_step` records preserve scalar
veRL metrics, including entropy, PPO KL, clip fractions, response lengths and
effective-group rates. `skipped_update` records expose zero-signal attempts
that do not advance the optimizer step. Data is written as those events occur;
a preflight failure will not produce rollout records.

Dynamic sampling consumes a variable number of input batches per optimizer
update. Therefore, `global_step_N` cannot determine the actual data position.
The directly integrated trainer saves `shopping_sampler_state.json` with its
schema version, completed optimizer step, actual epoch, number of batches
consumed within that epoch, dataloader length and sampler state. The companion
`data.pt` stores the stateful dataloader state. Resume loads these files
together to continue from the recorded cursor; it does not estimate the epoch
or cursor from the optimizer step.

To continue a checkpoint, retain its complete directory and select a new or
empty output directory for the resumed run:

```bash
bash scripts/grpo.sh --output outputs/models/grpo-resumed -- \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=outputs/models/grpo/global_step_100
```

Replace `global_step_100` with the checkpoint you actually saved. Keep the
training data and sampling configuration consistent with that run. The trainer
rejects a missing `shopping_sampler_state.json`, missing `data.pt`, mismatched
step/dataloader metadata or invalid cursor. Older checkpoints without the
Shopping sampler state cannot be resumed under this integration by guessing a
cursor or creating an empty state file. This is a format and state consistency
check, not a promise of bitwise-identical NPU results.

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

Select checkpoints using validation metrics rather than assuming that the
final training step is best. The existing reported step-100 result uses the
original model and hardware; it does not measure the new 9B NPU configuration.
