"""Check serving environment selection without importing vLLM or loading models."""

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
class ServeModelScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopping-serve-test-")
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.script = self.root / "scripts/serve_model.sh"
        self.script.parent.mkdir()
        self.script.write_text(
            (ROOT / "scripts/serve_model.sh").read_text(encoding="utf-8"),
            encoding="utf-8", newline="\n",
        )
        self.record = self.root / "arguments.bin"

    def launcher(self, relative, label):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'if [[ "${1:-}" == "-c" ]]; then\n'
            '  exit "${FIXTURE_DISCOVERY_EXIT:-0}"\n'
            "fi\n"
            f"printf '%s\\0' '{label}' \"$@\" > \"$FIXTURE_RECORD\"\n"
            'if [[ "${FIXTURE_STARTUP_EXIT:-0}" != 0 ]]; then\n'
            "  echo 'fixture: device initialization failed' >&2\n"
            "fi\n"
            'exit "${FIXTURE_STARTUP_EXIT:-0}"\n',
            encoding="utf-8", newline="\n",
        )
        path.chmod(0o755)
        return path

    def run_script(self, *arguments, **extra_env):
        environment = dict(os.environ)
        for name in ("PYTHON_BIN", "SERVED_MODEL_NAME", "LLM_PORT"):
            environment.pop(name, None)
        environment.update(
            FIXTURE_BIN=self.bin.as_posix(),
            FIXTURE_RECORD=self.record.as_posix(),
            FIXTURE_DISCOVERY_EXIT="0", FIXTURE_STARTUP_EXIT="0",
        )
        environment.update(extra_env)
        # Isolate PATH so no installed Python or vLLM can start a real server.
        shell = '''
for tool in bash dirname cat; do
  executable="$(command -v "$tool")" || exit 91
  printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$executable" > "$FIXTURE_BIN/$tool"
  chmod +x "$FIXTURE_BIN/$tool"
done
export PATH="$(cd "$FIXTURE_BIN" && pwd)"
exec bash "$@"
'''
        return subprocess.run(
            [BASH, "-c", shell, "fixture", self.script.as_posix(), *arguments],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=30,
        )

    def recorded(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.record.read_bytes().decode("utf-8").split("\0")[:-1]

    def test_current_cli_precedes_repository_environment(self):
        self.launcher("fake-bin/vllm", "current-cli")
        self.launcher(".venv/bin/vllm", "repo-cli")
        result = self.run_script("Qwen/Qwen3.5-4B")
        arguments = self.recorded(result)
        self.assertEqual(arguments[:3], ["current-cli", "serve", "Qwen/Qwen3.5-4B"])
        self.assertIn("Using vLLM launcher:", result.stderr)

    def test_python_package_without_cli_precedes_repository_environment(self):
        self.launcher("fake-bin/python", "current-python")
        self.launcher(".venv/bin/vllm", "repo-cli")
        arguments = self.recorded(self.run_script())
        self.assertEqual(arguments[:6], [
            "current-python", "-m", "vllm.entrypoints.cli.main",
            "serve", "Qwen/Qwen3.5-2B", "--served-model-name",
        ])

    def test_explicit_python_and_extra_arguments_preserve_boundaries(self):
        python = self.launcher("prepared env/bin/python", "explicit-python")
        self.launcher("fake-bin/vllm", "current-cli")
        extra = [
            "--enforce-eager", "--tensor-parallel-size", "4",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
        ]
        arguments = self.recorded(self.run_script(
            "models/my model", *extra, PYTHON_BIN=python.as_posix(),
            SERVED_MODEL_NAME="my-agent", LLM_PORT="9000",
        ))
        self.assertEqual(arguments[:5], [
            "explicit-python", "-m", "vllm.entrypoints.cli.main",
            "serve", "models/my model",
        ])
        self.assertEqual(arguments[-len(extra):], extra)
        self.assertEqual(arguments[5:9], ["--served-model-name", "my-agent", "--port", "9000"])

    def test_python3_can_supply_the_package(self):
        self.launcher("fake-bin/python3", "python3")
        self.assertEqual(self.recorded(self.run_script())[0], "python3")

    def test_repository_launcher_remains_a_fallback(self):
        self.launcher("fake-bin/python", "python-without-vllm")
        self.launcher(".venv/bin/vllm", "repo-cli")
        result = self.run_script(FIXTURE_DISCOVERY_EXIT="1")
        self.assertEqual(self.recorded(result)[0], "repo-cli")

    def test_missing_launcher_reports_environment_mismatch(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PYTHON_BIN", result.stderr)
        self.assertIn("-m pip show vllm", result.stderr)
        self.assertNotIn("Run: bash scripts/setup.sh", result.stderr)
        self.assertFalse(self.record.exists())

    def test_missing_explicit_interpreter_does_not_fall_back(self):
        self.launcher("fake-bin/vllm", "current-cli")
        result = self.run_script(PYTHON_BIN="missing-python-for-fixture")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PYTHON_BIN is not executable", result.stderr)
        self.assertFalse(self.record.exists())

    def test_startup_failure_is_preserved_without_fallback(self):
        self.launcher("fake-bin/vllm", "broken-cli")
        self.launcher(".venv/bin/vllm", "repo-cli")
        result = self.run_script(FIXTURE_STARTUP_EXIT="42")
        self.assertEqual(result.returncode, 42)
        self.assertIn("device initialization failed", result.stderr)
        self.assertNotIn("not installed", result.stderr)
        self.assertEqual(self.record.read_bytes().split(b"\0")[0], b"broken-cli")

    def test_explicit_python_preserves_import_or_startup_error(self):
        python = self.launcher("prepared-env/bin/python", "broken-python")
        self.launcher("fake-bin/vllm", "current-cli")
        result = self.run_script(PYTHON_BIN=python.as_posix(), FIXTURE_STARTUP_EXIT="43")
        self.assertEqual(result.returncode, 43)
        self.assertIn("device initialization failed", result.stderr)
        self.assertEqual(self.record.read_bytes().split(b"\0")[0], b"broken-python")

    def test_help_does_not_require_or_start_vllm(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PYTHON_BIN", result.stdout)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
