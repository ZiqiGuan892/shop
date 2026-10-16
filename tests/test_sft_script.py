"""Exercise the full-SFT launcher without importing a model or running training."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH_BIN") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "requires Bash (set BASH_BIN on Windows)")
class SFTScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopping-sft-script-")
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.record = self.root / "calls.txt"
        for name in ("sft.sh", "lib/python_runtime.sh"):
            self.write(
                self.root / "scripts" / name,
                (ROOT / "scripts" / name).read_text(encoding="utf-8"),
            )
        self.train = self.root / "outputs/sft-collection/clean/train.jsonl"
        self.validation = self.root / "outputs/sft-collection/clean/validation.jsonl"
        self.write(self.train, "{}\n")
        self.write(self.validation, "{}\n")

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def python(self, relative, label):
        path = self.root / relative
        self.write(path, f'''#!/usr/bin/env bash
set -euo pipefail
printf 'runtime={label}\\npath=%s\\ncwd=%s\\n' "$PYTHONPATH" "$PWD" >> "$FIXTURE_RECORD"
printf 'arg=<%s>\\n' "$@" >> "$FIXTURE_RECORD"
exit "${{FIXTURE_TRAIN_EXIT:-0}}"
''')
        return path

    def run_script(self, *arguments, **extra_env):
        environment = dict(os.environ)
        for key in (
            "PYTHON_BIN", "PYTHONPATH", "BASE_MODEL", "MODEL_ROOT", "SFT_TRAIN_FILE",
            "SFT_VALIDATION_FILE", "SFT_OUTPUT_DIR", "SFT_NPROC_PER_NODE",
            "SFT_ADAPTER_DIR", "SFT_MERGED_DIR",
        ):
            environment.pop(key, None)
        environment.update(
            FIXTURE_BIN=self.bin.as_posix(), FIXTURE_RECORD=self.record.as_posix(),
            BASE_MODEL="fixture-base-model",
        )
        environment.update(extra_env)
        shell = '''
for tool in bash dirname mkdir cat; do
  executable="$(command -v "$tool")" || exit 91
  printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$executable" > "$FIXTURE_BIN/$tool"
  chmod +x "$FIXTURE_BIN/$tool"
done
export PATH="$(cd "$FIXTURE_BIN" && pwd)"
exec bash "$@"
'''
        return subprocess.run(
            [BASH, "-c", shell, "fixture", (self.root / "scripts/sft.sh").as_posix(), *arguments],
            cwd=self.bin, env=environment, capture_output=True, text=True, timeout=30,
        )

    def recorded(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.record.read_text(encoding="utf-8")

    def test_default_uses_clean_data_full_training_and_current_environment(self):
        self.python("fake-bin/python", "current")
        self.python(".venv/bin/python", "repo")
        result = self.run_script(PYTHONPATH="/existing/imports")
        text = self.recorded(result)
        self.assertEqual(text.count("runtime=current\n"), 1)
        self.assertNotIn("runtime=repo", text)
        self.assertIn("Using SFT Python:", result.stderr)
        self.assertIn("scripts/train_sft.py>", text)
        self.assertIn("arg=<--finetuning-type>\narg=<full>", text)
        self.assertIn("arg=<--model>\narg=<fixture-base-model>", text)
        self.assertIn("/outputs/sft-collection/clean/train.jsonl>", text)
        self.assertIn("/outputs/sft-collection/clean/validation.jsonl>", text)
        self.assertIn("/outputs/models/sft-full>", text)
        self.assertIn("/src:/existing/imports\n", text)
        self.assertNotIn("merge_lora", text)
        self.assertIn("arg=<-m>\narg=<torch.distributed.run>", text)
        self.assertIn("arg=<--standalone>", text)
        self.assertIn("arg=<--nnodes=1>", text)
        self.assertIn("arg=<--nproc-per-node>\narg=<8>", text)
        self.assertIn("nodes: 1; processes: 8;", result.stderr)

    def test_explicit_interpreter_with_spaces_overrides_both_environments(self):
        python = self.python("prepared env/bin/python", "explicit")
        self.python("fake-bin/python", "current")
        self.python(".venv/bin/python", "repo")
        text = self.recorded(self.run_script(PYTHON_BIN=python.as_posix()))
        self.assertEqual(text.count("runtime=explicit\n"), 1)
        self.assertNotIn("runtime=current", text)
        self.assertNotIn("runtime=repo", text)

    def test_python3_and_repository_interpreter_fallbacks(self):
        for relative, label in (("fake-bin/python3", "python3"), (".venv/bin/python", "repo")):
            with self.subTest(runtime=label):
                python = self.python(relative, label)
                text = self.recorded(self.run_script())
                self.assertEqual(text.count(f"runtime={label}\n"), 1)
                python.unlink()
                self.record.unlink()

    def test_invalid_explicit_python_does_not_use_current_environment(self):
        self.python("fake-bin/python", "current")
        result = self.run_script(PYTHON_BIN="fixture-missing-python")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PYTHON_BIN is not executable", result.stderr)
        self.assertFalse(self.record.exists())

    def test_distributed_launch_uses_selected_interpreter_and_single_node(self):
        python = self.python("prepared env/bin/python", "explicit")
        self.python("fake-bin/python", "current")
        text = self.recorded(self.run_script(
            "--learning-rate", "1e-5", "--epochs", "2",
            PYTHON_BIN=python.as_posix(), SFT_NPROC_PER_NODE="4",
        ))
        self.assertEqual(text.count("runtime=explicit\n"), 1)
        self.assertIn("arg=<-m>\narg=<torch.distributed.run>", text)
        self.assertIn("arg=<--standalone>", text)
        self.assertIn("arg=<--nnodes=1>", text)
        self.assertIn("arg=<--nproc-per-node>\narg=<4>", text)
        self.assertIn("scripts/train_sft.py>", text)
        self.assertIn("arg=<--learning-rate>\narg=<1e-5>", text)
        self.assertIn("arg=<--epochs>\narg=<2>", text)

    def test_single_device_override_runs_without_distributed_launcher(self):
        self.python("fake-bin/python", "current")
        result = self.run_script(SFT_NPROC_PER_NODE="1")
        text = self.recorded(result)
        self.assertNotIn("torch.distributed.run", text)
        self.assertIn("scripts/train_sft.py>", text)
        self.assertIn("nodes: 1; processes: 1;", result.stderr)

    def test_custom_data_output_and_model_paths_preserve_spaces(self):
        self.python("fake-bin/python", "current")
        train = self.root / "new data/train.jsonl"
        validation = self.root / "new data/validation.jsonl"
        output = self.root / "new model/output"
        self.write(train, "{}\n")
        self.write(validation, "{}\n")
        text = self.recorded(self.run_script(
            BASE_MODEL="/models/prepared model", SFT_TRAIN_FILE=train.as_posix(),
            SFT_VALIDATION_FILE=validation.as_posix(), SFT_OUTPUT_DIR=output.as_posix(),
        ))
        self.assertIn("arg=<--model>\narg=</models/prepared model>", text)
        self.assertIn("new data/train.jsonl>", text)
        self.assertIn("new data/validation.jsonl>", text)
        self.assertIn("new model/output>", text)
        self.assertNotIn("/outputs/sft-collection/clean/", text)

    def test_missing_or_empty_split_fails_before_launching_training(self):
        self.python("fake-bin/python", "current")
        for path in (self.train, self.validation):
            for mode in ("missing", "empty"):
                with self.subTest(split=path.name, mode=mode):
                    if mode == "missing":
                        path.unlink()
                    else:
                        self.write(path, "")
                    result = self.run_script()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.record.exists(), result.stdout + result.stderr)
                    self.write(path, "{}\n")

    def test_lora_or_quantized_training_cannot_override_full_mode(self):
        self.python("fake-bin/python", "current")
        for arguments in (("--finetuning-type", "lora"), ("--finetuning-type=lora",), ("--qlora",)):
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.record.exists(), result.stdout + result.stderr)

    def test_invalid_worker_count_fails_before_launching_training(self):
        self.python("fake-bin/python", "current")
        for workers in ("0", "-1", "1.5", "many"):
            with self.subTest(workers=workers):
                result = self.run_script(SFT_NPROC_PER_NODE=workers)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.record.exists(), result.stdout + result.stderr)

    def test_training_failure_preserves_status_and_does_not_start_merge(self):
        self.python("fake-bin/python", "current")
        result = self.run_script(FIXTURE_TRAIN_EXIT="42")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        text = self.record.read_text(encoding="utf-8")
        self.assertEqual(text.count("runtime=current\n"), 1)
        self.assertNotIn("merge_lora", text)

    def test_help_does_not_require_data_or_model(self):
        self.python("fake-bin/python", "current")
        self.train.unlink()
        self.validation.unlink()
        result = self.run_script("--help", BASE_MODEL="/missing/model")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Default: 8 (single-node torchrun/DDP)", result.stdout)
        if self.record.exists():
            self.assertIn("arg=<--help>", self.record.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
