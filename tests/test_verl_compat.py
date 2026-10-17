"""veRL 不应为了纯 padding 操作强制依赖 FlashAttention。"""

from contextlib import contextmanager
import json
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class VerlCompatTest(unittest.TestCase):
    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
                "verl.trainer": trainer,
                "verl.trainer.ppo": ppo,
                "verl.trainer.ppo.ray_trainer": ray_trainer,
            },
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            with patch("shopping_grpo.training.grpo.compat._install_npu_device_binding_diagnostics") as binding:
                install_torch_padding_fallback()
            binding.assert_called_once_with()

        self.assertEqual(attention._get_attention_functions(), expected)
        self.assertTrue(RayPPOTrainer._update_actor._shopping_trace)


class NPUDeviceBindingTest(unittest.TestCase):
    @contextmanager
    def worker_fixture(self, ids=("7",), local_rank="7", current_device=7, native_error=None):
        """Only fake native binding and Ray metadata; never initialize hardware."""
        calls = []
        result = object()

        class Worker:
            def _setup_env_cuda_visible_devices(self):
                calls.append("native")
                if native_error:
                    raise native_error
                if local_rank is None:
                    os.environ.pop("LOCAL_RANK", None)
                else:
                    os.environ["LOCAL_RANK"] = local_rank
                return result

        worker_module = ModuleType("verl.single_controller.base.worker")
        worker_module.Worker = Worker
        ray = ModuleType("ray")
        accelerator_ids = Mock(side_effect=lambda: calls.append("ray_ids") or {"NPU": list(ids)})
        ray.get_runtime_context = Mock(return_value=SimpleNamespace(get_accelerator_ids=accelerator_ids))
        torch = ModuleType("torch")
        torch.npu = SimpleNamespace(
            current_device=Mock(side_effect=lambda: calls.append("current_device") or current_device),
            set_device=Mock(),
        )
        masks = {
            "ASCEND_VISIBLE_DEVICES": "10,11,12,13,14,15,8,9",
            "ASCEND_RT_VISIBLE_DEVICES": "7,6,5,4,3,2,1,0",
        }
        modules = {name: ModuleType(name) for name in (
            "verl", "verl.single_controller", "verl.single_controller.base",
        )}
        modules.update({"verl.single_controller.base.worker": worker_module, "ray": ray, "torch": torch})
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
            **masks, "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "RANK": "3", "LOCAL_RANK": "0",
        }, clear=True), patch("builtins.print") as console:
            yield SimpleNamespace(
                Worker=Worker, calls=calls, result=result, torch=torch,
                ray=ray, masks=masks, console=console,
            )

    def test_native_binding_precedes_diagnostics_and_preserves_reordered_masks(self):
        from shopping_grpo.training.grpo.compat import _install_npu_device_binding_diagnostics

        for device_id in (7, 3, 0):
            with self.subTest(device_id=device_id), self.worker_fixture(
                ids=(str(device_id),), local_rank=str(device_id), current_device=device_id,
            ) as fixture:
                _install_npu_device_binding_diagnostics()
                fixture.ray.get_runtime_context.assert_not_called()
                fixture.torch.npu.current_device.assert_not_called()
                actual = fixture.Worker()._setup_env_cuda_visible_devices()
                self.assertIs(actual, fixture.result)
                self.assertEqual(fixture.calls, ["native", "ray_ids", "current_device"])
                fixture.torch.npu.set_device.assert_not_called()
                for name, value in fixture.masks.items():
                    self.assertEqual(os.environ[name], value)
                fixture.console.assert_called_once()
                line = fixture.console.call_args.args[0]
                self.assertTrue(line.startswith("GRPO NPU worker binding: "))
                diagnostic = json.loads(line.removeprefix("GRPO NPU worker binding: "))
                self.assertEqual(diagnostic["pid"], os.getpid())
                self.assertEqual(diagnostic["rank"], "3")
                self.assertEqual(diagnostic["ray_npu_ids"], [str(device_id)])
                self.assertEqual(diagnostic["local_rank"], str(device_id))
                self.assertEqual(diagnostic["current_device"], device_id)
                for name, value in fixture.masks.items():
                    self.assertEqual(diagnostic[name], value)

    def test_duplicate_installation_does_not_repeat_native_binding(self):
        from shopping_grpo.training.grpo.compat import _install_npu_device_binding_diagnostics

        with self.worker_fixture() as fixture:
            _install_npu_device_binding_diagnostics()
            first_wrapper = fixture.Worker._setup_env_cuda_visible_devices
            self.assertTrue(first_wrapper._shopping_npu_binding)
            _install_npu_device_binding_diagnostics()
            self.assertIs(fixture.Worker._setup_env_cuda_visible_devices, first_wrapper)
            fixture.Worker()._setup_env_cuda_visible_devices()
            self.assertEqual(fixture.calls.count("native"), 1)
            fixture.console.assert_called_once()

    def test_flag_disabled_does_not_import_or_modify_native_worker(self):
        from shopping_grpo.training.grpo.compat import _install_npu_device_binding_diagnostics

        for flag in (None, "0"):
            environment = {} if flag is None else {"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": flag}
            with self.subTest(flag=flag), patch.dict(os.environ, environment, clear=True), patch.dict(
                sys.modules, {"verl.single_controller.base.worker": None},
            ):
                _install_npu_device_binding_diagnostics()

    def test_worker_without_npu_allocation_never_queries_or_sets_npu(self):
        from shopping_grpo.training.grpo.compat import _install_npu_device_binding_diagnostics

        with self.worker_fixture(ids=()) as fixture:
            _install_npu_device_binding_diagnostics()
            self.assertIs(fixture.Worker()._setup_env_cuda_visible_devices(), fixture.result)
            self.assertEqual(fixture.calls, ["native", "ray_ids"])
            fixture.torch.npu.current_device.assert_not_called()
            fixture.torch.npu.set_device.assert_not_called()
            fixture.console.assert_not_called()

    def test_invalid_binding_fails_before_worker_initialization_can_continue(self):
        from shopping_grpo.training.grpo.compat import _install_npu_device_binding_diagnostics

        mismatches = (
            (("7",), "0", 7),
            (("7",), "7", 0),
            (("7", "6"), "7", 7),
            (("7",), None, 7),
            (("bad",), "bad", 7),
        )
        for ids, local_rank, current_device in mismatches:
            with self.subTest(ids=ids, local_rank=local_rank, current_device=current_device), self.worker_fixture(
                ids=ids, local_rank=local_rank, current_device=current_device,
            ) as fixture:
                _install_npu_device_binding_diagnostics()
                initialize_model = Mock()
                with self.assertRaisesRegex(RuntimeError, "NPU worker binding mismatch"):
                    fixture.Worker()._setup_env_cuda_visible_devices()
                    initialize_model()
                initialize_model.assert_not_called()
                self.assertEqual(fixture.calls.count("native"), 1)
                fixture.console.assert_called_once()
                fixture.torch.npu.set_device.assert_not_called()

    def test_original_binding_failure_is_not_hidden_by_diagnostics(self):
        from shopping_grpo.training.grpo.compat import _install_npu_device_binding_diagnostics

        error = ValueError("native device selection failed")
        with self.worker_fixture(native_error=error) as fixture:
            _install_npu_device_binding_diagnostics()
            with self.assertRaises(ValueError) as caught:
                fixture.Worker()._setup_env_cuda_visible_devices()
            self.assertIs(caught.exception, error)
            self.assertEqual(fixture.calls, ["native"])
            fixture.ray.get_runtime_context.assert_not_called()
            fixture.torch.npu.current_device.assert_not_called()
            fixture.console.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
