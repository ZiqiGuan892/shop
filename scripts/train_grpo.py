#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-full"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/validation.parquet"


def _quote_hydra_string(value: str) -> str:
    """Escape quotes and adjacent backslashes without changing ordinary paths."""
    # Hydra quoted strings do not use JSON's general backslash escaping.
    escaped = re.sub(
        r'(\\*)("|$)',
        lambda match: match[1] * 2 + ('\\"' if match[2] == '"' else ""),
        value,
    )
    return f'"{escaped}"'


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path(os.environ.get("GRPO_MODEL_PATH", DEFAULT_MODEL)),
        help="complete Qwen3.5-9B SFT model (default: outputs/models/sft-full)",
    )
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "console", "swanlab"),
        default="tensorboard",
        help="metric backend alongside console (default: local TensorBoard)",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print command; no runtime checks or training")
    mode.add_argument("--check-only", action="store_true", help="check installed runtime and data without training")
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args(argv)


def hydra_overrides(args: argparse.Namespace, environment=None) -> list[str]:
    logger = "[console]" if args.logger == "console" else f"[console,{args.logger}]"
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    paths = {
        "actor_rollout_ref.model.path": "--model",
        "data.train_files": "--train-data",
        "data.val_files": "--val-data",
        "trainer.default_local_dir": "--output",
    }
    worker_pythonpath = "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH"
    worker_tensorboard = "ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR"
    for item in extra:
        if item.startswith("--"):
            raise SystemExit("only Hydra key=value overrides are accepted after --")
        key = item.split("=", 1)[0].lstrip("+~")
        if key == worker_pythonpath or worker_pythonpath.startswith(key + "."):
            raise SystemExit(
                "set worker PYTHONPATH using shell PYTHONPATH or VERL_ROOT; "
                "the launcher forwards the same concrete import path to preflight and Ray workers"
            )
        if key == worker_tensorboard:
            raise SystemExit("TENSORBOARD_DIR is managed by --output; do not override its Ray environment value")
        for field, option in paths.items():
            if key == field or field.startswith(key + "."):
                raise SystemExit(f"set {field} using {option}, so preflight and training use the same path")
    overrides = [f"trainer.logger={logger}", f"trainer.experiment_name={args.experiment_name}"]
    if environment is not None:
        # The server's main_ppo converts ray_init without resolving interpolation.
        # Quote the actual path as one Hydra string, passed via argv (not a shell).
        value = _quote_hydra_string(environment["PYTHONPATH"])
        overrides.append(f"++{worker_pythonpath}={value}")
        tensorboard_dir = _quote_hydra_string(environment["TENSORBOARD_DIR"])
        overrides.append(f"++{worker_tensorboard}={tensorboard_dir}")
    return [*overrides, *extra]


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if any(output.iterdir()):
            raise SystemExit(f"output directory must be new or empty: {output}")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    # Ray/veRL creates its own worker ranks and rendezvous. Do not inherit a
    # surrounding torchrun job's identity; retain the platform's device masks.
    for name in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "NODE_RANK", "MASTER_ADDR", "MASTER_PORT",
    ):
        environment.pop(name, None)
    # Show each worker's device binding even when only numeric IDs differ.
    environment.setdefault("RAY_DEDUP_LOGS", "0")
    import_paths = [str(ROOT / "src")]
    if environment.get("VERL_ROOT"):
        verl_root = _validated_path(Path(environment["VERL_ROOT"]), "veRL checkout")
        if not (verl_root / "verl/__init__.py").is_file():
            raise SystemExit("VERL_ROOT must point to the checkout containing verl/__init__.py")
        import_paths.append(str(verl_root))
        environment["VERL_ROOT"] = str(verl_root)
    if environment.get("PYTHONPATH"):
        import_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(import_paths),
            "PYTHONUNBUFFERED": "1",
            "RAY_ADDRESS": "local",
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            # Isolate event files by run, including a resumed run. Forward the
            # literal value to Ray because vendor ray_init may not resolve OmegaConf.
            "TENSORBOARD_DIR": str(output / "tensorboard"),
            "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_NAME": config.stem,
            "GRPO_CONFIG_DIR": str(config.parent),
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *hydra_overrides(args, environment),
    ]
    return command, environment


def run_logged(command, environment, log) -> int:
    """Stream merged stdout/stderr to the terminal and the run's train.log."""
    with subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "tensorboard_log_dir": environment["TENSORBOARD_DIR"],
        "config": str(args.config.resolve()),
        "python": sys.executable,
        "verl_root": environment.get("VERL_ROOT", "not set; resolving verl from current Python import path"),
        "log": str(Path(environment["GRPO_OUTPUT_DIR"]) / "train.log"),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    preflight = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *hydra_overrides(args, environment),
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status or args.check_only:
        raise SystemExit(preflight_status)
    output = Path(environment["GRPO_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        log.flush()
        raise SystemExit(run_logged(command, environment, log))


if __name__ == "__main__":
    main()
