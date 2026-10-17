"""Offline launch checks: never start Ray, load model weights, or train."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_grpo


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH_BIN") or (shutil.which("bash") if os.name != "nt" else None)


class GRPOLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopping-grpo-launch-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "9B SFT model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        self.output = self.root / "run"
        self.options = ["--model", str(self.model), "--output", str(self.output)]
        self.env = patch.dict(os.environ, {"PYTHONPATH": "/existing/vendor/imports", "RAY_ADDRESS": "auto"})
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ("VERL_ROOT", "GRPO_MODEL_PATH"):
            if name in os.environ:
                saved = os.environ.pop(name)
                self.addCleanup(os.environ.__setitem__, name, saved)

    def args(self, *extra):
        return train_grpo.parse_args([*self.options, *extra])

    def main(self, args):
        with patch.object(train_grpo, "parse_args", return_value=args), patch("builtins.print"):
            train_grpo.main()

    def test_current_python_paths_and_local_ray_are_preserved(self):
        command, env = train_grpo.build_command(self.args())
        self.assertEqual(command[:3], [sys.executable, "-m", "verl.trainer.main_ppo"])
        self.assertEqual(env["PYTHONPATH"], str(ROOT / "src") + os.pathsep + "/existing/vendor/imports")
        self.assertEqual(env["RAY_ADDRESS"], "local")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        self.assertEqual(env["GRPO_MODEL_PATH"], str(self.model.resolve()))
        self.assertFalse(self.output.exists())

    def test_external_verl_checkout_is_selected_without_installation(self):
        checkout = self.root / "vendor verl"
        (checkout / "verl").mkdir(parents=True)
        (checkout / "verl/__init__.py").write_text("", encoding="utf-8")
        with patch.dict(os.environ, {"VERL_ROOT": str(checkout)}):
            _, env = train_grpo.build_command(self.args())
        self.assertEqual(env["VERL_ROOT"], str(checkout.resolve()))
        self.assertIn(str(checkout.resolve()), env["PYTHONPATH"].split(os.pathsep))
        with patch.dict(os.environ, {"VERL_ROOT": str(checkout / "verl")}):
            with self.assertRaisesRegex(SystemExit, "checkout containing"):
                train_grpo.build_command(self.args())

    def test_default_model_is_sft_result_and_explicit_override_wins(self):
        self.assertEqual(train_grpo.parse_args([]).model, ROOT / "outputs/models/sft-full")
        with patch.dict(os.environ, {"GRPO_MODEL_PATH": str(self.model)}):
            self.assertEqual(train_grpo.parse_args([]).model, self.model)
            self.assertEqual(train_grpo.parse_args(["--model", "another"]).model, Path("another"))

    def test_missing_sft_never_falls_back_to_base_model(self):
        args = self.args()
        args.model = self.root / "missing"
        with self.assertRaisesRegex(SystemExit, "model directory does not exist"):
            train_grpo.build_command(args)

    def test_hydra_paths_must_use_dedicated_options(self):
        for override in (
            "actor_rollout_ref.model.path=/wrong", "data.train_files=/wrong",
            "data.val_files=/wrong", "++trainer.default_local_dir=/wrong", "data={}",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(SystemExit, "using --"):
                train_grpo.build_command(self.args("--", override))

    def test_custom_config_directory_reaches_preflight(self):
        config = self.root / "custom.yaml"
        config.write_text("value: 1\n", encoding="utf-8")
        command, env = train_grpo.build_command(self.args("--config", str(config)))
        self.assertIn(f"--config-path={config.parent}", command)
        self.assertEqual(env["GRPO_CONFIG_DIR"], str(config.parent))
        self.assertEqual(env["GRPO_CONFIG_NAME"], "custom")
        from scripts.check_grpo_runtime import compose_runtime_config
        with patch.dict(os.environ, env):
            composed = compose_runtime_config(["value=2"])
        self.assertEqual(composed.value, 2)

    def test_dry_run_never_calls_preflight_or_training(self):
        with patch.object(train_grpo.subprocess, "call") as check, patch.object(train_grpo, "run_logged") as train:
            self.main(self.args("--dry-run"))
        check.assert_not_called()
        train.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_check_only_preserves_exit_status_and_does_not_reserve_output(self):
        with patch.object(train_grpo.subprocess, "call", return_value=0) as check, patch.object(train_grpo, "run_logged") as train:
            with self.assertRaises(SystemExit) as exc:
                self.main(self.args("--check-only", "--", "trainer.total_training_steps=20"))
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("trainer.total_training_steps=20", check.call_args.args[0])
        self.assertIn("check_grpo_runtime.py", check.call_args.args[0][1])
        train.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_preflight_failure_can_be_retried_without_deleting_output(self):
        with patch.object(train_grpo.subprocess, "call", return_value=5), patch.object(train_grpo, "run_logged") as train:
            with self.assertRaises(SystemExit) as exc:
                self.main(self.args())
        self.assertEqual(exc.exception.code, 5)
        self.assertFalse(self.output.exists())
        train.assert_not_called()
        train_grpo.build_command(self.args())

    def test_actual_launch_uses_same_overrides_and_saves_log_and_audit(self):
        args = self.args("--", "trainer.total_training_steps=20")
        with patch.object(train_grpo.subprocess, "call", return_value=0) as check, patch.object(train_grpo, "run_logged", return_value=7) as train:
            with self.assertRaises(SystemExit) as exc:
                self.main(args)
        self.assertEqual(exc.exception.code, 7)
        self.assertEqual(check.call_args.args[0][2:], train.call_args.args[0][5:])
        audit = json.loads((self.output / "launch.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["python"], sys.executable)
        self.assertEqual(audit["model"], str(self.model.resolve()))
        self.assertIn("trainer.total_training_steps=20", (self.output / "train.log").read_text(encoding="utf-8"))

    def test_existing_training_output_is_not_overwritten(self):
        self.output.mkdir()
        (self.output / "checkpoint").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "new or empty"):
            train_grpo.build_command(self.args())

    def test_log_stream_contains_stdout_stderr_and_exit_code(self):
        log = io.StringIO()
        with patch("builtins.print") as console:
            status = train_grpo.run_logged(
                [sys.executable, "-u", "-c", "import sys; print('progress'); print('failure', file=sys.stderr); sys.exit(3)"],
                dict(os.environ), log,
            )
        self.assertEqual(status, 3)
        self.assertIn("progress", log.getvalue())
        self.assertIn("failure", log.getvalue())
        self.assertEqual(console.call_count, 2)


@unittest.skipUnless(BASH, "requires Bash (set BASH_BIN on Windows)")
class GRPOBashTest(unittest.TestCase):
    def test_launcher_uses_active_python_even_when_repository_venv_exists(self):
        with tempfile.TemporaryDirectory(prefix="shopping-grpo-bash-") as directory:
            root = Path(directory)
            for relative in ("scripts/grpo.sh", "scripts/lib/python_runtime.sh"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            for relative, label in (("bin/python", "active"), (".venv/bin/python", "venv")):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f'#!/bin/bash\nprintf "{label}\\n"\nprintf "<%s>\\n" "$@"\n', encoding="utf-8", newline="\n")
                target.chmod(0o755)
            env = dict(os.environ)
            env.pop("PYTHON_BIN", None)
            env["FIXTURE_BIN"] = (root / "bin").as_posix()
            result = subprocess.run(
                [BASH, "-c", 'export PATH="$(cd "$FIXTURE_BIN" && pwd):$PATH"; exec bash "$@"', "fixture",
                 (root / "scripts/grpo.sh").as_posix(), "--check-only"],
                capture_output=True, text=True, env=env, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Using GRPO Python:", result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["active", "<scripts/train_grpo.py>", "<--check-only>"])


if __name__ == "__main__":
    unittest.main()
