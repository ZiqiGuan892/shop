"""Check active-environment resolution and fail-closed patching without veRL."""

from __future__ import annotations

import sys
import os
import tempfile
import types
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

from scripts import apply_verl_dynamic_sampling_patch as patcher


class GRPOPatchEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopping-verl-env-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.package = self.root / "external-checkout" / "verl"
        self.package.mkdir(parents=True)
        self.init_file = self.package / "__init__.py"
        self.init_file.write_text("", encoding="utf-8")
        self.target = self.package / "trainer" / "ppo" / "ray_trainer.py"
        self.target.parent.mkdir(parents=True)
        self.target.write_text("# customized trainer\n", encoding="utf-8")
        self.verl = types.ModuleType("verl")
        self.verl.__file__ = str(self.init_file)
        if "VERL_ROOT" in os.environ:
            saved_verl_root = os.environ.pop("VERL_ROOT")
            self.addCleanup(os.environ.__setitem__, "VERL_ROOT", saved_verl_root)

    def resolve(self, version):
        self.verl.__version__ = version
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError("verl")), \
                mock.patch.dict(sys.modules, {"verl": self.verl}):
            return patcher.resolve_installed_ray_trainer()

    def test_resolves_external_editable_checkout_from_active_python(self):
        self.assertFalse(self.target.is_relative_to(patcher.PROJECT_ROOT / ".venv"))
        self.assertEqual(self.resolve("0.8.0"), self.target.resolve())

    def test_accepts_local_build_version_without_relaxing_public_version(self):
        self.assertEqual(self.resolve("0.8.0+ascend.local"), self.target.resolve())

    def test_rejects_other_public_versions(self):
        for version in ("0.7.0", "0.8.1+ascend", "0.8.0rc1", "0.8.0.post1"):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "requires veRL 0.8.0 APIs"):
                self.resolve(version)

    def test_missing_trainer_has_actionable_error(self):
        self.target.unlink()
        with self.assertRaisesRegex(RuntimeError, "installed ray_trainer.py does not exist"):
            self.resolve("0.8.0+ascend")

    def test_unknown_fork_is_unchanged_without_invoking_patch_program(self):
        before = self.target.read_bytes()
        with mock.patch.object(patcher.shutil, "which") as find_patch, \
                mock.patch.object(patcher.subprocess, "run") as run_patch, \
                self.assertRaisesRegex(RuntimeError, "needs a reviewed port"):
            patcher.apply_patch(self.target)
        find_patch.assert_not_called()
        run_patch.assert_not_called()
        self.assertEqual(self.target.read_bytes(), before)
        self.assertFalse(Path(str(self.target) + patcher.BACKUP_SUFFIX).exists())


if __name__ == "__main__":
    unittest.main()
