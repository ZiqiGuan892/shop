import unittest
import json
from pathlib import Path
import tempfile

from shopping_grpo.environment.manifest import (
    MANIFEST_VERSION,
    shopsimulator_source_commit,
    validate_manifest,
)
from scripts.check_grpo_runtime import CURRENT_RUNTIME_FILES, validate_reward_runtime_files


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentManifestTest(unittest.TestCase):
    def test_repository_runtime_sources_match_frozen_manifest(self):
        manifest = json.loads((ROOT / "data/environment.json").read_text(encoding="utf-8"))
        validate_manifest(manifest)
        validate_reward_runtime_files(manifest, ROOT)

    def test_runtime_source_drift_is_still_rejected(self):
        manifest = json.loads((ROOT / "data/environment.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path in CURRENT_RUNTIME_FILES.values():
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative_path).read_bytes())
            validate_reward_runtime_files(manifest, root)
            source = root / CURRENT_RUNTIME_FILES["web_agent_text_env.py"]
            source.write_bytes(source.read_bytes() + b"\n# unexpected source change\n")
            with self.assertRaisesRegex(SystemExit, "runtime file hash mismatch.*web_agent_text_env.py"):
                validate_reward_runtime_files(manifest, root)

    def test_current_environment_contract_is_validated(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_page_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest({})

    def test_current_environment_requires_reward_v3(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)
        manifest["reward"] = {"version": "unsupported-reward"}
        with self.assertRaisesRegex(ValueError, "requires shopsimulator-reward-v3"):
            validate_manifest(manifest)

    def test_wrong_tool_contract_is_rejected(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "unsupported-tools",
            "max_steps": 35,
            "seed": 20260726,
        }
        with self.assertRaisesRegex(ValueError, "Tool v2"):
            validate_manifest(manifest)

    def test_embedded_shopsimulator_commit_is_read_without_nested_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "EMBEDDED_SOURCE.json").write_text(
                json.dumps({"source_commit": "e" * 40}),
                encoding="utf-8",
            )
            self.assertEqual(shopsimulator_source_commit(root), "e" * 40)


if __name__ == "__main__":
    unittest.main()
