"""Exercise source-only veRL selection in fresh Python processes, without veRL."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from shopping_grpo.training.grpo.verl_source import resolve_verl_source

try:
    metadata_version = version("verl")
except PackageNotFoundError:
    metadata_version = None
try:
    selected_version, source = resolve_verl_source()
except RuntimeError as exc:
    print(json.dumps({"error": str(exc), "metadata_version": metadata_version}))
    sys.exit(3)
print(json.dumps({
    "version": selected_version,
    "source": str(source),
    "metadata_version": metadata_version,
}))
"""


class VerLSourceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="shopping-verl-source-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        # The child must resolve the package through PYTHONPATH, not its cwd.
        self.work = self.root / "unrelated-working-directory"
        self.work.mkdir()

    def package(self, source="__version__ = '0.9.0.dev'\n", version_file=None):
        package = self.checkout / "verl"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text(source, encoding="utf-8")
        if version_file is not None:
            (package / "version").mkdir(exist_ok=True)
            (package / "version/version").write_text(version_file, encoding="utf-8")
        return (package / "__init__.py").resolve()

    def metadata(self, value):
        metadata = self.checkout / f"verl-{value}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: verl\nVersion: {value}\n",
            encoding="utf-8",
        )

    def probe(self, *, verl_root=None, succeeds=True):
        environment = dict(os.environ)
        environment.pop("VERL_ROOT", None)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(self.checkout), str(ROOT / "src"))
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONUTF8"] = "1"
        if verl_root is not None:
            environment["VERL_ROOT"] = str(verl_root)
        completed = subprocess.run(
            [sys.executable, "-S", "-B", "-c", PROBE],
            cwd=self.work,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        self.assertEqual(
            completed.returncode, 0 if succeeds else 3,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        return json.loads(completed.stdout)

    def test_source_package_imports_without_distribution_metadata(self):
        source = self.package("__version__ = '0.9.0.dev+ascend.local'\n")
        result = self.probe(verl_root=self.checkout)
        self.assertEqual(result["version"], "0.9.0.dev+ascend.local")
        self.assertEqual(Path(result["source"]), source)
        self.assertIsNone(result["metadata_version"])

    def test_source_version_file_fallback_without_distribution_metadata(self):
        source = self.package("", version_file="  0.9.0.dev+source\n")
        result = self.probe()
        self.assertEqual(result["version"], "0.9.0.dev+source")
        self.assertEqual(Path(result["source"]), source)
        self.assertIsNone(result["metadata_version"])

    def test_dev_and_dev0_spellings_and_local_builds_are_equivalent(self):
        for value in ("0.9.0.dev", "0.9.0.dev0", "0.9.0.dev+ascend", "0.9.0.dev0+ascend"):
            with self.subTest(version=value):
                self.package(f"__version__ = {value!r}\n")
                self.assertEqual(self.probe()["version"], value)

    def test_stable_future_and_other_dev_revisions_are_not_silently_accepted(self):
        for value in ("0.8.0", "0.9.0", "0.9.0rc1", "0.9.0.dev1", "0.10.0.dev"):
            with self.subTest(version=value):
                self.package(f"__version__ = {value!r}\n")
                self.assertIn("requires veRL 0.9.0.dev source", self.probe(succeeds=False)["error"])

    def test_blank_or_non_string_attribute_falls_back_to_source_file(self):
        for value in ("'  '", "None", "8"):
            with self.subTest(value=value):
                self.package(f"__version__ = {value}\n", version_file="0.9.0.dev\n")
                self.assertEqual(self.probe()["version"], "0.9.0.dev")

    def test_missing_package_explains_source_path_selection(self):
        error = self.probe(succeeds=False)["error"]
        self.assertIn("cannot import verl", error)
        self.assertIn("set VERL_ROOT", error)
        self.assertIn("PYTHONPATH", error)
        self.assertNotIn("dependency is missing", error)

    def test_missing_internal_dependency_does_not_claim_verl_is_absent(self):
        self.package("import shopping_test_missing_vendor_dependency\n")
        error = self.probe(succeeds=False)["error"]
        self.assertIn("dependency is missing", error)
        self.assertIn("shopping_test_missing_vendor_dependency", error)
        self.assertNotIn("set VERL_ROOT", error)

    def test_other_import_failure_preserves_original_diagnostic(self):
        self.package("raise ImportError('vendor API mismatch')\n")
        error = self.probe(succeeds=False)["error"]
        self.assertIn("cannot import the selected verl source", error)
        self.assertIn("vendor API mismatch", error)

    def test_verl_root_mismatch_is_rejected(self):
        source = self.package()
        requested = self.root / "another-checkout"
        error = self.probe(verl_root=requested, succeeds=False)["error"]
        self.assertIn("VERL_ROOT selection mismatch", error)
        self.assertIn(str(source), error)
        self.assertIn(str(requested / "verl/__init__.py"), error)

    def test_other_source_version_is_rejected(self):
        self.package("__version__ = '0.7.0+ascend'\n")
        error = self.probe(succeeds=False)["error"]
        self.assertIn("requires veRL 0.9.0.dev source", error)
        self.assertIn("0.7.0+ascend", error)

    def test_old_distribution_metadata_does_not_override_selected_source(self):
        source = self.package()
        self.metadata("0.7.0")
        result = self.probe()
        self.assertEqual(result["metadata_version"], "0.7.0")
        self.assertEqual(result["version"], "0.9.0.dev")
        self.assertEqual(Path(result["source"]), source)

    def test_matching_distribution_metadata_cannot_mask_wrong_source_version(self):
        self.package("__version__ = '0.7.0'\n")
        self.metadata("0.9.0.dev")
        result = self.probe(succeeds=False)
        self.assertEqual(result["metadata_version"], "0.9.0.dev")
        self.assertIn("found '0.7.0'", result["error"])

    def test_distribution_metadata_cannot_replace_missing_source_version(self):
        self.package("")
        self.metadata("0.9.0.dev")
        result = self.probe(succeeds=False)
        self.assertEqual(result["metadata_version"], "0.9.0.dev")
        self.assertIn("cannot determine veRL source version", result["error"])

    def test_namespace_package_is_not_a_complete_verl_checkout(self):
        (self.checkout / "verl").mkdir()
        error = self.probe(succeeds=False)["error"]
        self.assertIn("has no __file__", error)


if __name__ == "__main__":
    unittest.main()
