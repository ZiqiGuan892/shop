"""Active-environment source replacement, with no veRL installation or training."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import types
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

from scripts import install_verl_trainer as installer


class GRPOTrainerInstallTest(unittest.TestCase):
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
        self.original = b"# Original server trainer\nclass RayPPOTrainer:\n    pass\n"
        self.replacement = b"# Reviewed direct integration\nclass RayPPOTrainer:\n    integration = True\n"
        self.target.write_bytes(self.original)
        self.source = self.root / "repository_trainer.py"
        self.source.write_bytes(self.replacement)
        self.backup = Path(str(self.target) + installer.BACKUP_SUFFIX)
        self.verl = types.ModuleType("verl")
        self.verl.__file__ = str(self.init_file)
        for name, value in {
            "TRAINER_FILE": self.source,
            "EXPECTED_ORIGINAL_SHA256": hashlib.sha256(self.original).hexdigest(),
            "EXPECTED_TRAINER_SHA256": hashlib.sha256(self.replacement).hexdigest(),
        }.items():
            patch = mock.patch.object(installer, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        if "VERL_ROOT" in os.environ:
            saved_verl_root = os.environ.pop("VERL_ROOT")
            self.addCleanup(os.environ.__setitem__, "VERL_ROOT", saved_verl_root)

    def resolve(self, version):
        self.verl.__version__ = version
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError("verl")), \
                mock.patch.dict(sys.modules, {"verl": self.verl}):
            return installer.resolve_installed_ray_trainer()

    def test_resolves_external_editable_checkout_from_active_python(self):
        self.assertFalse(self.target.is_relative_to(installer.PROJECT_ROOT / ".venv"))
        self.assertEqual(self.resolve("0.9.0.dev"), self.target.resolve())

    def test_accepts_local_build_version_without_relaxing_public_version(self):
        for version in ("0.9.0.dev", "0.9.0.dev0", "0.9.0.dev+ascend.local", "0.9.0.dev0+ascend"):
            with self.subTest(version=version):
                self.assertEqual(self.resolve(version), self.target.resolve())

    def test_rejects_other_public_versions(self):
        for version in ("0.8.0", "0.9.0", "0.9.0rc1", "0.9.0.dev1", "0.10.0.dev"):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "requires veRL 0.9.0.dev source"):
                self.resolve(version)

    def test_missing_trainer_has_actionable_error(self):
        self.target.unlink()
        with self.assertRaisesRegex(RuntimeError, "installed ray_trainer.py does not exist"):
            self.resolve("0.9.0.dev+ascend")

    def test_install_is_idempotent_and_restore_recovers_exact_original(self):
        self.target.write_bytes(self.original.replace(b"\n", b"\r\n"))
        original_bytes = self.target.read_bytes()
        installer.install_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), self.replacement)
        self.assertEqual(self.backup.read_bytes(), original_bytes)
        installer.verify_installed(self.target)
        installer.install_trainer(self.target)
        self.assertEqual(self.backup.read_bytes(), original_bytes)
        installer.restore_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), original_bytes)
        installer.restore_trainer(self.target)

    def test_check_is_read_only_and_accepts_crlf_equivalent(self):
        self.target.write_bytes(self.replacement.replace(b"\n", b"\r\n"))
        before = self.target.read_bytes()
        installer.verify_installed(self.target)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertFalse(self.backup.exists())

    def test_unknown_fork_is_unchanged_without_creating_backup(self):
        self.target.write_bytes(self.original + b"# custom change\n")
        before = self.target.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite unknown"):
            installer.install_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertFalse(self.backup.exists())

    def test_corrupt_repository_source_cannot_touch_server_source(self):
        self.source.write_bytes(self.replacement + b"# changed\n")
        with self.assertRaisesRegex(RuntimeError, "repository trainer hash mismatch"):
            installer.install_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertFalse(self.backup.exists())

    def test_corrupt_existing_backup_is_preserved_and_blocks_install(self):
        self.backup.write_bytes(b"do not replace this backup\n")
        with self.assertRaisesRegex(RuntimeError, "original backup hash mismatch"):
            installer.install_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertEqual(self.backup.read_bytes(), b"do not replace this backup\n")

    def test_restore_refuses_unknown_changes_after_install(self):
        installer.install_trainer(self.target)
        changed = self.replacement + b"# independent change\n"
        self.target.write_bytes(changed)
        with self.assertRaisesRegex(RuntimeError, "refusing to restore over unknown"):
            installer.restore_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), changed)

    def test_failed_post_install_verification_rolls_back_original(self):
        with mock.patch.object(installer, "verify_installed", side_effect=RuntimeError("verification failed")):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                installer.install_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertEqual(self.backup.read_bytes(), self.original)

    def test_missing_original_backup_blocks_restore(self):
        self.target.write_bytes(self.replacement)
        with self.assertRaisesRegex(RuntimeError, "cannot restore without original backup"):
            installer.restore_trainer(self.target)
        self.assertEqual(self.target.read_bytes(), self.replacement)


if __name__ == "__main__":
    unittest.main()
