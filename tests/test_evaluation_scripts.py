"""Exercise evaluation wrappers with fake Python, without running benchmark tasks."""

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
class EvaluationScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopping-eval-test-")
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.record = self.root / "calls.txt"
        for name in (
            "baseline.sh", "evaluate.sh", "report.sh", "report_all.sh", "lib/python_runtime.sh",
        ):
            self.write(
                self.root / "scripts" / name,
                (ROOT / "scripts" / name).read_text(encoding="utf-8"),
            )

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
if [[ "${{FIXTURE_EVAL_FAIL:-0}}" == 1 && "$1" == *evaluate_shop_benchmark.py ]]; then
  echo 'fixture evaluation failed' >&2
  exit 42
fi
''')
        return path

    def run_script(self, name, *arguments, **extra_env):
        environment = dict(os.environ)
        for key in (
            "PYTHON_BIN", "EVAL_OUTPUT_DIR", "SHOPSIM_BASE_URL", "LLM_BASE_URL",
            "LLM_API_KEY", "SERVED_MODEL_NAME", "PYTHONPATH",
            "EVAL_PROGRESS_INTERVAL", "EVAL_ROLLOUT_PRINT_EVERY",
        ):
            environment.pop(key, None)
        environment.update(FIXTURE_BIN=self.bin.as_posix(), FIXTURE_RECORD=self.record.as_posix())
        environment.update(extra_env)
        shell = '''
for tool in bash dirname mkdir; do
  executable="$(command -v "$tool")" || exit 91
  printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$executable" > "$FIXTURE_BIN/$tool"
  chmod +x "$FIXTURE_BIN/$tool"
done
export PATH="$(cd "$FIXTURE_BIN" && pwd)"
exec bash "$@"
'''
        return subprocess.run(
            [BASH, "-c", shell, "fixture", (self.root / "scripts" / name).as_posix(), *arguments],
            cwd=self.bin, env=environment, capture_output=True, text=True, timeout=30,
        )

    def recorded(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.record.read_text(encoding="utf-8")

    def test_baseline_without_repo_venv_uses_current_python_for_eval_and_report(self):
        self.python("fake-bin/python", "current")
        text = self.recorded(self.run_script("baseline.sh", PYTHONPATH="/existing/imports"))
        self.assertEqual(text.count("runtime=current\n"), 2)
        self.assertIn("arg=<scripts/evaluate_shop_benchmark.py>", text)
        self.assertIn("arg=<data/evaluation/tasks.jsonl>", text)
        self.assertIn("arg=<scripts/build_eval_report.py>", text)
        self.assertIn("arg=<--progress-interval>\narg=<15>", text)
        self.assertIn("arg=<--rollout-print-every>\narg=<5>", text)
        self.assertIn("/outputs/evaluation/baseline/trajectories.jsonl>", text)
        paths = [line for line in text.splitlines() if line.startswith("path=")]
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(line.endswith("/src:/existing/imports") for line in paths), text)

    def test_explicit_python_overrides_path_and_repo_environment(self):
        python = self.python("prepared env/bin/python", "explicit")
        self.python("fake-bin/python", "current")
        self.python(".venv/bin/python", "repo")
        text = self.recorded(self.run_script("evaluate.sh", "sft", PYTHON_BIN=python.as_posix()))
        self.assertEqual(text.count("runtime=explicit\n"), 2)
        self.assertNotIn("runtime=current", text)
        self.assertNotIn("runtime=repo", text)

    def test_progress_and_rollout_intervals_can_be_configured(self):
        self.python("fake-bin/python", "current")
        text = self.recorded(self.run_script(
            "baseline.sh", EVAL_PROGRESS_INTERVAL="10", EVAL_ROLLOUT_PRINT_EVERY="2",
        ))
        self.assertIn("arg=<--progress-interval>\narg=<10>", text)
        self.assertIn("arg=<--rollout-print-every>\narg=<2>", text)

    def test_python3_fallback(self):
        self.python("fake-bin/python3", "python3")
        text = self.recorded(self.run_script("baseline.sh"))
        self.assertEqual(text.count("runtime=python3\n"), 2)

    def test_repository_python_remains_a_fallback(self):
        self.python(".venv/bin/python", "repo")
        text = self.recorded(self.run_script("baseline.sh"))
        self.assertEqual(text.count("runtime=repo\n"), 2)

    def test_missing_python_fails_before_creating_outputs(self):
        result = self.run_script("baseline.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot find Python", result.stderr)
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse(self.record.exists())

    def test_missing_explicit_python_does_not_fall_back(self):
        self.python("fake-bin/python", "current")
        result = self.run_script("baseline.sh", PYTHON_BIN="fixture-missing-python")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PYTHON_BIN is not executable", result.stderr)
        self.assertFalse(self.record.exists())

    def test_failed_evaluation_does_not_generate_a_report(self):
        self.python("fake-bin/python", "current")
        result = self.run_script("baseline.sh", FIXTURE_EVAL_FAIL="1")
        self.assertEqual(result.returncode, 42)
        self.assertIn("fixture evaluation failed", result.stderr)
        self.assertNotIn("build_eval_report.py", self.record.read_text(encoding="utf-8"))

    def test_output_and_service_configuration_is_preserved(self):
        self.python("fake-bin/python", "current")
        output = self.root / "custom output"
        text = self.recorded(self.run_script(
            "baseline.sh", EVAL_OUTPUT_DIR=output.as_posix(), SERVED_MODEL_NAME="my-agent",
            SHOPSIM_BASE_URL="http://localhost:5701", LLM_BASE_URL="http://localhost:8001/v1",
            LLM_API_KEY="test-key",
        ))
        for value in ("my-agent", "http://localhost:5701", "http://localhost:8001/v1", "test-key"):
            self.assertIn(f"arg=<{value}>", text)
        self.assertIn("custom output/trajectories.jsonl>", text)
        self.assertIn("custom output/summary.json>", text)

    def test_standalone_report_reuses_current_python(self):
        self.python("fake-bin/python", "current")
        text = self.recorded(self.run_script("report.sh", "baseline"))
        self.assertEqual(text.count("runtime=current\n"), 1)
        self.assertIn("arg=<scripts/build_eval_report.py>", text)
        self.assertNotIn("evaluate_shop_benchmark", text)

    def test_all_reports_reuse_current_python_and_skip_incomplete_runs(self):
        self.python("fake-bin/python", "current")
        for relative in (
            "outputs/evaluation/baseline/summary.json",
            "outputs/evaluation/baseline/trajectories.jsonl",
            "outputs/evaluation/incomplete/summary.json",
        ):
            self.write(self.root / relative, "{}\n")
        text = self.recorded(self.run_script("report_all.sh"))
        self.assertEqual(text.count("runtime=current\n"), 2)
        self.assertIn("/build_eval_report.py>", text)
        self.assertIn("/build_comparison_report.py>", text)
        self.assertNotIn("/incomplete>", text)


if __name__ == "__main__":
    unittest.main()
