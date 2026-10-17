"""Offline checks for the server environment and eight-NPU GRPO recipe."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from scripts import check_grpo_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


class GRPORuntimeEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(
            os.environ,
            {
                "GRPO_TRAIN_FILE": str(ROOT / "data/grpo/train.parquet"),
                "GRPO_VAL_FILE": str(ROOT / "data/grpo/validation.parquet"),
                "GRPO_MODEL_PATH": str(ROOT / "outputs/models/sft-full"),
                "GRPO_OUTPUT_DIR": str(ROOT / "outputs/grpo"),
                "SHOPPING_GRPO_ROOT": str(ROOT),
                "PYTHONPATH": str(ROOT / "src"),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        if "VERL_ROOT" in os.environ:
            saved_verl_root = os.environ.pop("VERL_ROOT")
            self.addCleanup(os.environ.__setitem__, "VERL_ROOT", saved_verl_root)
        # Load project overrides directly: these checks must not import veRL or
        # require its Hydra package defaults merely to validate the launch recipe.
        self.config = OmegaConf.load(ROOT / "configs/grpo.yaml")
        self.stdout = io.StringIO()
        self.output = contextlib.redirect_stdout(self.stdout)
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.versions = {
            "verl": "0.9.0.dev+ascend.local",
            "vllm": "0.18.0+empty",
            "torch": "2.9.0",
            "transformers": "5.0.0.dev0",
            "ray": "2.48.0",
            "tensordict": "0.10.0",
            "numpy": "1.26.4",
            "peft": "0.17.0",
            "pyarrow": "21.0.0",
            "torch-npu": "2.9.0",
            "vllm-ascend": "0.18.0+local",
        }
        self.verl = types.SimpleNamespace(
            __file__=str(ROOT / "fake-vendor/verl/__init__.py"),
            __version__="0.9.0.dev+ascend.local",
        )

    def installed_version(self, package):
        try:
            return self.versions[package]
        except KeyError as exc:
            raise PackageNotFoundError(package) from exc

    def dependencies(self):
        with mock.patch.object(runtime.sys, "version_info", (3, 11, 0)), \
                mock.patch.dict(sys.modules, {"verl": self.verl}), \
                mock.patch.object(runtime, "version", side_effect=self.installed_version):
            return runtime.validate_installed_dependencies(self.config)

    @staticmethod
    def mock_npu_torch(count=8, hccl=True):
        return types.SimpleNamespace(
            npu=mock.Mock(is_available=mock.Mock(return_value=True), device_count=mock.Mock(return_value=count)),
            cuda=mock.Mock(is_available=mock.Mock(return_value=False)),
            distributed=mock.Mock(is_backend_available=mock.Mock(return_value=hccl)),
        )

    def test_server_versions_and_console_logging_do_not_require_reinstallation(self):
        installed = self.dependencies()
        self.assertEqual(installed["torch"], "2.9.0")
        self.assertEqual(installed["vllm"], "0.18.0+empty")
        self.assertEqual(installed["verl"], "0.9.0.dev+ascend.local")
        self.assertNotIn("swanlab", installed)
        self.assertIn("torch-npu", installed)
        self.assertIn("vllm-ascend", installed)

    def test_missing_required_packages_fail_before_loading_models(self):
        for package in ("peft", "torch-npu", "vllm-ascend"):
            with self.subTest(package=package), mock.patch.dict(self.versions):
                del self.versions[package]
                with self.assertRaisesRegex(SystemExit, f"missing GRPO dependency.*{package}"):
                    self.dependencies()

    def test_swanlab_is_required_only_when_selected(self):
        self.config.trainer.logger = ["console", "swanlab"]
        with self.assertRaisesRegex(SystemExit, "missing GRPO dependency.*swanlab"):
            self.dependencies()
        self.versions["swanlab"] = "0.7.0"
        self.assertEqual(self.dependencies()["swanlab"], "0.7.0")

    def test_wrong_verl_public_version_is_rejected(self):
        self.verl.__version__ = "0.7.0+ascend"
        with self.assertRaisesRegex(SystemExit, "requires veRL 0.9.0.dev source"):
            self.dependencies()

    def test_verl_source_without_pip_metadata_is_accepted(self):
        del self.versions["verl"]
        self.assertEqual(self.dependencies()["verl"], self.verl.__version__)
        self.assertIn(self.verl.__file__, self.stdout.getvalue())

    def test_verl_source_version_takes_precedence_over_unrelated_metadata(self):
        self.versions["verl"] = "0.7.0"
        self.assertEqual(self.dependencies()["verl"], "0.9.0.dev+ascend.local")

    def test_npu_launch_uses_hccl_without_requiring_cuda(self):
        torch = self.mock_npu_torch()
        runtime.validate_device(self.config, torch)
        torch.npu.is_available.assert_called_once_with()
        torch.npu.device_count.assert_called_once_with()
        torch.distributed.is_backend_available.assert_called_once_with("hccl")
        torch.cuda.is_available.assert_not_called()

    def test_v1_trainer_cannot_bypass_reviewed_shopping_hooks(self):
        # A vendor ppo_trainer may default to V1. Our overrides must select the
        # hooked trainer explicitly, not rely on the guard's missing-key default.
        self.assertIs(self.config.trainer.use_v1, False)
        self.config = OmegaConf.merge({"trainer": {"use_v1": True}}, self.config)
        runtime.validate_trainer_backend(self.config)
        self.config.trainer.use_v1 = True
        with self.assertRaisesRegex(SystemExit, "trainer.use_v1=true"):
            runtime.validate_trainer_backend(self.config)

    def test_too_few_visible_npus_fail_preflight(self):
        with self.assertRaisesRegex(SystemExit, "requested 8 npu devices, but only 4 are visible"):
            runtime.validate_device(self.config, self.mock_npu_torch(count=4))

    def test_missing_hccl_fails_preflight(self):
        with self.assertRaisesRegex(SystemExit, "HCCL is unavailable"):
            runtime.validate_device(self.config, self.mock_npu_torch(hccl=False))

    def test_single_node_requires_local_ray_and_matching_resource_count(self):
        self.config.ray_kwargs.ray_init.address = "auto"
        with self.assertRaisesRegex(SystemExit, "address=local"):
            runtime.validate_device(self.config, self.mock_npu_torch())
        self.config.ray_kwargs.ray_init.address = "local"
        self.config.ray_kwargs.ray_init.resources.NPU = 4
        with self.assertRaisesRegex(SystemExit, "Ray NPU resources must equal"):
            runtime.validate_device(self.config, self.mock_npu_torch())
        self.config.ray_kwargs.ray_init.resources.NPU = 8.5
        with self.assertRaisesRegex(SystemExit, "Ray NPU resources must equal"):
            runtime.validate_device(self.config, self.mock_npu_torch())

    def ray_config(self):
        config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=False))
        config.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH = os.environ["PYTHONPATH"]
        return config

    def test_literal_ray_init_is_valid_without_loading_ray_or_models(self):
        config = self.ray_config()
        runtime.validate_ray_init(config)
        raw = OmegaConf.to_container(config.ray_kwargs.ray_init, resolve=False)
        self.assertEqual(raw["resources"]["NPU"], 8)
        self.assertIs(type(raw["resources"]["NPU"]), int)
        self.assertIn("GRPO Ray init preflight passed", self.stdout.getvalue())

    def test_resource_interpolation_is_rejected_even_when_attribute_access_resolves_it(self):
        config = self.ray_config()
        config.ray_kwargs.ray_init.resources.NPU = "${trainer.n_gpus_per_node}"
        self.assertEqual(config.ray_kwargs.ray_init.resources.NPU, 8)
        self.assertEqual(
            OmegaConf.to_container(config.ray_kwargs.ray_init, resolve=False)["resources"]["NPU"],
            "${trainer.n_gpus_per_node}",
        )
        with self.assertRaisesRegex(SystemExit, "unresolved Ray interpolation.*resources.NPU"):
            runtime.validate_ray_init(config)

    def test_worker_path_interpolation_is_rejected_even_when_environment_resolves_it(self):
        config = self.ray_config()
        config.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH = "${oc.env:PYTHONPATH}"
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH, os.environ["PYTHONPATH"])
        with self.assertRaisesRegex(SystemExit, "unresolved Ray interpolation.*PYTHONPATH"):
            runtime.validate_ray_init(config)

    def test_nested_ray_interpolations_are_rejected_before_runtime_initialization(self):
        config = self.ray_config()
        config.ray_kwargs.ray_init.runtime_env.py_modules = ["${oc.env:PYTHONPATH}"]
        with self.assertRaisesRegex(SystemExit, "unresolved Ray interpolation.*py_modules"):
            runtime.validate_ray_init(config)

    def test_ray_resources_require_finite_nonnegative_numbers(self):
        for value in ("8", True, False, -1, float("inf"), float("-inf"), float("nan"), None):
            with self.subTest(value=value):
                config = self.ray_config()
                config.ray_kwargs.ray_init.resources.NPU = value
                with self.assertRaisesRegex(SystemExit, "Ray resource NPU must be a finite non-negative number"):
                    runtime.validate_ray_init(config)
        config = self.ray_config()
        config.ray_kwargs.ray_init.resources.extra_capacity = 0.5
        config.ray_kwargs.ray_init.resources.unused_capacity = 0
        runtime.validate_ray_init(config)

    def test_worker_environment_requires_strings_and_matching_concrete_import_path(self):
        for value in (1, True, None):
            with self.subTest(value=value):
                config = self.ray_config()
                config.ray_kwargs.ray_init.runtime_env.env_vars.EXTRA = value
                with self.assertRaisesRegex(SystemExit, "env_vars must contain string values"):
                    runtime.validate_ray_init(config)
        for value in ("", "/another/import/path"):
            with self.subTest(pythonpath=value):
                config = self.ray_config()
                config.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH = value
                with self.assertRaisesRegex(SystemExit, "Ray worker PYTHONPATH must equal"):
                    runtime.validate_ray_init(config)
        config = self.ray_config()
        with mock.patch.dict(os.environ, {"PYTHONPATH": ""}):
            with self.assertRaisesRegex(SystemExit, "Ray worker PYTHONPATH must equal"):
                runtime.validate_ray_init(config)

    def test_invalid_ray_init_fails_before_dependency_and_device_checks(self):
        config = self.ray_config()
        config.ray_kwargs.ray_init.resources.NPU = "${trainer.n_gpus_per_node}"
        fake_torch = self.mock_npu_torch()
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(runtime.importlib.util, "find_spec", return_value=None), \
                mock.patch.object(runtime, "compose_runtime_config", return_value=config), \
                mock.patch.object(runtime, "validate_installed_dependencies") as dependencies, \
                mock.patch.object(runtime, "validate_device") as device:
            with self.assertRaisesRegex(SystemExit, "unresolved Ray interpolation"):
                runtime.main()
        dependencies.assert_not_called()
        device.assert_not_called()
        fake_torch.npu.is_available.assert_not_called()

    def test_eight_trajectories_produce_one_local_accumulation_step_on_eight_cards(self):
        self.assertEqual(runtime.validate_parallel_batches(self.config), 1)
        self.assertIn("global trajectories=8", self.stdout.getvalue())
        runtime.validate_training_memory_budget(self.config)

    def test_rollout_tensor_parallelism_must_divide_available_cards(self):
        self.config.actor_rollout_ref.rollout.tensor_model_parallel_size = 3
        with self.assertRaisesRegex(SystemExit, "tensor parallel size must divide"):
            runtime.validate_parallel_batches(self.config)

    def test_prompt_batch_cannot_leave_actor_ranks_without_micro_batches(self):
        self.config.data.train_batch_size = 1
        self.config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
        with self.assertRaisesRegex(SystemExit, "expanded prompt/mini batches"):
            runtime.validate_parallel_batches(self.config)

    def test_ppo_mini_batch_must_divide_prompt_batch(self):
        self.config.data.train_batch_size = 3
        with self.assertRaisesRegex(SystemExit, "train batch must be divisible"):
            runtime.validate_parallel_batches(self.config)

    def test_qwen_sequence_parallelism_is_not_silently_enabled(self):
        for path in (
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size",
        ):
            with self.subTest(path=path):
                config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=False))
                OmegaConf.update(config, path, 2)
                with self.assertRaisesRegex(SystemExit, "sequence parallel size 1"):
                    runtime.validate_parallel_batches(config)

    def test_repository_parquets_are_disjoint_from_each_other_and_final_evaluation(self):
        runtime.validate_data_isolation(self.config)
        self.assertIn("GRPO data isolation passed", self.stdout.getvalue())

    def test_custom_config_cannot_redirect_validated_output_or_inputs(self):
        runtime.validate_launch_paths(self.config)
        for path in ("trainer.default_local_dir", "data.train_files", "actor_rollout_ref.model.path"):
            with self.subTest(path=path):
                config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=False))
                OmegaConf.update(config, path, "/unexpected/path")
                with self.assertRaisesRegex(SystemExit, "config path disagrees"):
                    runtime.validate_launch_paths(config)

    def test_train_and_validation_each_reject_final_evaluation_contamination(self):
        import pyarrow as pa
        import pyarrow.parquet as parquet

        evaluation = [
            int(json.loads(line)["task_id"])
            for line in (ROOT / "data/evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        clean_id = max(evaluation) + 1
        with tempfile.TemporaryDirectory(prefix="shopping-grpo-isolation-") as temporary:
            train = Path(temporary) / "train.parquet"
            validation = Path(temporary) / "validation.parquet"
            self.config.data.train_files = str(train)
            self.config.data.val_files = str(validation)
            for contaminated_split in ("train", "validation"):
                with self.subTest(split=contaminated_split):
                    train_id = evaluation[0] if contaminated_split == "train" else clean_id
                    validation_id = evaluation[0] if contaminated_split == "validation" else clean_id + 1
                    parquet.write_table(pa.Table.from_pylist([{"extra_info": {"task_id": train_id}}]), train)
                    parquet.write_table(pa.Table.from_pylist([{"extra_info": {"task_id": validation_id}}]), validation)
                    with self.assertRaisesRegex(SystemExit, "overlaps frozen data/evaluation/tasks.jsonl"):
                        runtime.validate_data_isolation(self.config)

    def test_larger_model_keeps_original_learning_and_trajectory_budgets(self):
        model = self.config.actor_rollout_ref.model
        actor = self.config.actor_rollout_ref.actor
        rollout = self.config.actor_rollout_ref.rollout
        self.assertEqual((model.lora_rank, model.lora_alpha), (16, 32))
        self.assertAlmostEqual(actor.optim.lr, 1e-6)
        self.assertEqual((self.config.data.train_batch_size, rollout.n), (2, 4))
        self.assertEqual(self.config.data.max_prompt_length + self.config.data.max_response_length, 24576)
        self.assertEqual(rollout.max_model_len, 24576)
        self.assertEqual(self.config.algorithm.adv_estimator, "grpo")
        self.assertTrue(self.config.shopping_dynamic_sampling.enable)


if __name__ == "__main__":
    unittest.main()
