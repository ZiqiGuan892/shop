"""Exercise launcher side effects with fake installers, without downloading packages."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH_BIN") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "requires Bash (set BASH_BIN on Windows)")
class SetupScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopsim-setup-test-")
        self.root = Path(self.temporary.name).resolve()
        # Cleanup is limited to this newly created fixture directory.
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.shop = self.root / "environments/ShopSimulator/shop_env"
        (self.root / "scripts").mkdir()
        (self.shop / "data").mkdir(parents=True)
        (self.shop / "shop_env").mkdir()
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.record = self.root / "calls.txt"
        self.training = self.root / "existing-training-env"
        self.training.mkdir()
        (self.training / "marker").write_text("keep existing NPU environment", encoding="utf-8")

        # Keep real gzip/hash checks, using a tiny corpus and its checksum only
        # in the copied test fixture. The repository's corpus is not modified.
        products = b'[{"asin":"fixture-product","title":"fixture"}]\n'
        archive = self.shop / "data/fine_items_eval_train_all.json.gz"
        archive.write_bytes(gzip.compress(products))
        self.products = products
        for relative in (
            "scripts/setup.sh", "scripts/start_environment.sh",
            "environments/ShopSimulator/shop_env/start.sh",
        ):
            content = (ROOT / relative).read_text(encoding="utf-8")
            if relative == "scripts/setup.sh":
                content = content.replace(
                    "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f",
                    hashlib.sha256(products).hexdigest(),
                )
            self.write(self.root / relative, content)

        self.write(self.bin / "fake-python", """#!/usr/bin/env bash
set -euo pipefail
printf 'python %s' "$0" >> "$FIXTURE_RECORD"
printf ' <%s>' "$@" >> "$FIXTURE_RECORD"
printf '\\n' >> "$FIXTURE_RECORD"
case "${1:-}" in
  -) cat >/dev/null ;;
  scripts/build_index.py) mkdir -p search_engine; touch search_engine/products.sqlite3 ;;
  pack_api.py)
    printf 'slots=%s port=%s version=%s\\n' "$SHOPSIM_ENV_SLOTS" "$SHOPSIM_PORT" "$SHOP_ENVIRONMENT_VERSION" >> "$FIXTURE_RECORD"
    echo 'Fake service stdout'
    echo 'Fake service stderr' >&2 ;;
esac
""")
        self.write(self.bin / "uv", """#!/usr/bin/env bash
set -euo pipefail
printf 'uv' >> "$FIXTURE_RECORD"
printf ' <%s>' "$@" >> "$FIXTURE_RECORD"
printf '\\n' >> "$FIXTURE_RECORD"
case "$1" in
  --version) echo 'uv fixture'; exit 0 ;;
  sync) destination="$PWD/.venv" ;;
  venv) destination="${@: -1}" ;;
  pip) exit 0 ;;
  *) exit 90 ;;
esac
mkdir -p "$destination/bin"
cp "$FIXTURE_PYTHON" "$destination/bin/python"
chmod +x "$destination/bin/python"
""")

    @staticmethod
    def write(path, content):
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_script(self, relative, *arguments, **extra_env):
        environment = dict(os.environ)
        for name in ("SHOP_SEARCH_INDEX", "SHOP_ENV_CONFIG", "SHOPSIM_LOG_DIR"):
            environment.pop(name, None)
        environment.update(
            FIXTURE_ROOT=self.root.as_posix(),
            FIXTURE_RECORD=self.record.as_posix(),
            FIXTURE_PYTHON=(self.bin / "fake-python").as_posix(),
            FIXTURE_BIN=self.bin.as_posix(),
            FIXTURE_UV_PAYLOAD=(self.root / "uv-payload").as_posix(),
            FIXTURE_ISOLATE_TOOLS="0",
            VIRTUAL_ENV=self.training.as_posix(),
            SHOPSIM_PORT="5700", SHOPSIM_ENV_SLOTS="8", SHOP_MAX_STEPS="35",
        )
        environment.update(extra_env)
        shell = '''
export UV_INSTALL_DIR="$(cd "$FIXTURE_ROOT" && pwd)/uv-bin"
if [[ "$FIXTURE_ISOLATE_TOOLS" == 1 ]]; then
  # Give bootstrap tests only known shell utilities and fake downloaders.
  # A real host uv/curl/wget must never be used by a missing-uv test.
  for tool in bash sh dirname mkdir cp chmod touch cat gzip sha256sum awk mv rm mktemp tee; do
    executable="$(command -v "$tool")" || exit 91
    printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$executable" > "$FIXTURE_BIN/$tool"
    chmod +x "$FIXTURE_BIN/$tool"
  done
  export PATH="$(cd "$FIXTURE_BIN" && pwd)"
else
  export PATH="$(cd "$FIXTURE_BIN" && pwd):$PATH"
fi
exec bash "$@"
'''
        return subprocess.run(
            [BASH, "-c", shell,
             "fixture", (self.root / relative).as_posix(), *arguments],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=30,
        )

    def install_simulator(self):
        result = self.run_script("scripts/setup.sh", "--shopsim-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.record.read_text(encoding="utf-8")

    def prepare_bootstrap(self, downloader="curl"):
        (self.bin / "uv").rename(self.root / "uv-payload")
        self.write(self.bin / downloader, """#!/usr/bin/env bash
set -euo pipefail
printf 'download %s\\n' "$0" >> "$FIXTURE_RECORD"
destination="${@: -1}"
if [[ "${FIXTURE_DOWNLOAD_FAIL:-0}" == 1 ]]; then
  echo 'echo PARTIAL_INSTALLER_EXECUTED >> "$FIXTURE_RECORD"' > "$destination"
  exit 22
fi
cat > "$destination" <<'INSTALLER'
#!/bin/sh
set -eu
echo "installer no-modify-path=$UV_NO_MODIFY_PATH" >> "$FIXTURE_RECORD"
if [ "${FIXTURE_INSTALL_FAIL:-0}" = 1 ]; then exit 9; fi
mkdir -p "$UV_INSTALL_DIR"
cp "$FIXTURE_UV_PAYLOAD" "$UV_INSTALL_DIR/uv"
chmod +x "$UV_INSTALL_DIR/uv"
INSTALLER
""")

    def test_missing_uv_is_installed_once_and_available_to_setup(self):
        self.prepare_bootstrap()
        for _ in range(2):
            result = self.run_script(
                "scripts/setup.sh", "--shopsim-only", FIXTURE_ISOLATE_TOOLS="1",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.record.read_text()
        self.assertEqual(calls.count("download "), 1)
        self.assertEqual(calls.count("installer no-modify-path=1"), 1)
        self.assertIn("<pip> <install>", calls)
        self.assertNotIn("<sync>", calls)
        self.assertTrue((self.root / "uv-bin/uv").is_file())
        self.assertFalse((self.root / ".venv").exists())

    def test_missing_uv_can_use_wget_without_curl(self):
        self.prepare_bootstrap("wget")
        result = self.run_script(
            "scripts/setup.sh", "--shopsim-only", FIXTURE_ISOLATE_TOOLS="1",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("fake-bin/wget", self.record.read_text())

    def test_failed_uv_download_does_not_execute_partial_installer(self):
        self.prepare_bootstrap()
        result = self.run_script(
            "scripts/setup.sh", "--shopsim-only",
            FIXTURE_ISOLATE_TOOLS="1", FIXTURE_DOWNLOAD_FAIL="1",
        )
        self.assertNotEqual(result.returncode, 0)
        calls = self.record.read_text()
        self.assertNotIn("PARTIAL_INSTALLER_EXECUTED", calls)
        self.assertNotIn("uv <", calls)
        self.assertFalse((self.root / "environments/ShopSimulator/.venv-shopsim").exists())

    def test_failed_uv_installation_stops_setup(self):
        self.prepare_bootstrap()
        result = self.run_script(
            "scripts/setup.sh", "--shopsim-only",
            FIXTURE_ISOLATE_TOOLS="1", FIXTURE_INSTALL_FAIL="1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uv installation failed", result.stderr)
        self.assertNotIn("uv <", self.record.read_text())

    def test_standalone_setup_keeps_training_environment_and_prepares_products(self):
        calls = self.install_simulator()
        self.assertNotIn("<sync>", calls)
        self.assertNotIn("apply_verl_dynamic_sampling_patch.py", calls)
        self.assertFalse((self.root / ".venv").exists())
        self.assertIn("<pip> <install> <--no-config>", calls)
        self.assertIn(".venv-shopsim/bin/python", calls)
        self.assertIn("shop_env/requirements.txt", calls)
        self.assertIn("<scripts/build_index.py>", calls)
        self.assertEqual((self.shop / "data/items_eval_train.json").read_bytes(), self.products)
        self.assertEqual((self.training / "marker").read_text(), "keep existing NPU environment")

    def test_full_setup_still_syncs_training_and_applies_patch(self):
        result = self.run_script("scripts/setup.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.record.read_text()
        self.assertIn("<sync>", calls)
        self.assertIn("<scripts/apply_verl_dynamic_sampling_patch.py>", calls)

    def test_service_uses_simulator_python_and_captures_console(self):
        self.install_simulator()
        result = self.run_script("scripts/start_environment.sh", SHOPSIM_PORT="5800", SHOPSIM_ENV_SLOTS="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.record.read_text()
        self.assertIn(".venv-shopsim/bin/python <pack_api.py>", calls)
        self.assertIn("slots=3 port=5800 version=shopsimulator-environment-v2.1", calls)
        console = (self.root / "outputs/shopsimulator/server.log").read_text()
        self.assertIn("Fake service stdout", console)
        self.assertIn("Fake service stderr", console)

    def test_corrupt_products_fail_before_index_build(self):
        (self.shop / "data/items_eval_train.json").write_bytes(b"corrupted")
        result = self.run_script("scripts/setup.sh", "--shopsim-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertNotIn("build_index.py", self.record.read_text())

    def test_missing_installation_reports_standalone_command(self):
        result = self.run_script("scripts/start_environment.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("setup.sh --shopsim-only", result.stdout + result.stderr)

    def test_help_and_invalid_options_have_no_installation_side_effects(self):
        for relative in ("scripts/setup.sh", "scripts/start_environment.sh"):
            for argument, expected in (("--help", 0), ("--invalid", 2)):
                with self.subTest(relative=relative, argument=argument):
                    result = self.run_script(relative, argument)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertFalse(self.record.exists())
        self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
