"""服务器 veRL 0.9.0.dev 的项目 runtime hooks；入口仍须检查实际接口。"""

import functools
import json
import os


def _install_npu_device_binding_diagnostics():
    """Verify veRL's native device choice before model/communicator creation."""
    if os.environ.get("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES") != "1":
        return

    from verl.single_controller.base.worker import Worker

    original = Worker._setup_env_cuda_visible_devices
    if getattr(original, "_shopping_npu_binding", False):
        return

    @functools.wraps(original)
    def setup_with_binding_diagnostics(self, *args, **kwargs):
        # NOSET makes veRL use Ray's accelerator ID with its native set_device.
        # The hook observes the result; it never sets a device or rewrites masks.
        result = original(self, *args, **kwargs)
        import ray

        ids = ray.get_runtime_context().get_accelerator_ids().get("NPU", [])
        if not ids:
            return result

        import torch

        current_device = torch.npu.current_device()
        binding = {
            "pid": os.getpid(),
            "rank": os.environ.get("RANK"),
            "ray_npu_ids": ids,
            "local_rank": os.environ.get("LOCAL_RANK"),
            "current_device": current_device,
            "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        }
        print("GRPO NPU worker binding: " + json.dumps(binding, sort_keys=True), flush=True)
        try:
            matches = (
                len(ids) == 1
                and int(ids[0]) == int(binding["local_rank"])
                and int(ids[0]) == current_device
            )
        except (ValueError, TypeError):
            matches = False
        if not matches:
            raise RuntimeError(
                "NPU worker binding mismatch: veRL must select its Ray-assigned NPU "
                "in NOSET mode before loading weights; inspect the binding log above. "
                "Do not replace platform device masks with physical card IDs."
            )
        return result

    setup_with_binding_diagnostics._shopping_npu_binding = True
    Worker._setup_env_cuda_visible_devices = setup_with_binding_diagnostics


def _install_trace_actor_update():
    """Score private gold targets before the existing veRL actor update."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._update_actor, "_shopping_trace", False):
        return

    from shopping_grpo.training.grpo.trace import (
        apply_trace_advantages,
        build_trace_score_batch,
    )

    original_update_actor = RayPPOTrainer._update_actor

    def update_actor_with_trace(self, batch):
        config = self.config.get("shopping_trace", {})
        enabled = bool(config.get("enable", False))
        if not enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            return original_update_actor(self, batch)
        if not self.ref_in_actor:
            raise RuntimeError("shopping TRACE requires LoRA so the frozen base model is available")

        score_batch, state_counts = build_trace_score_batch(
            batch,
            self.tokenizer,
            max_sequence_length=int(config["max_sequence_length"]),
        )
        reference = self._compute_ref_log_prob(score_batch)
        target_mask = score_batch.batch["response_mask"]
        mean_log_probs = (
            reference.batch["ref_log_prob"] * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1)
        batch.non_tensor_batch.pop("trace_target", None)
        trace_metrics = apply_trace_advantages(
            batch,
            mean_log_probs,
            state_counts,
            epsilon=float(config["epsilon"]),
            horizon=int(config["horizon"]),
            discount=float(config["discount"]),
            terminal_weight=float(config["terminal_weight"]),
            outcome_weight=float(config["outcome_weight"]),
            turn_weight=float(config["turn_weight"]),
        )
        trace_metrics["trace/target_log_prob_mean"] = float(mean_log_probs.mean())
        output = original_update_actor(self, batch)
        output.meta_info["metrics"].update(trace_metrics)
        return output

    update_actor_with_trace._shopping_trace = True
    RayPPOTrainer._update_actor = update_actor_with_trace


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。"""
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # 保留服务器分支需要的 torch padding hook；由预检查验证这些接口存在。
    attention_utils._get_attention_functions = lambda: functions
    _install_trace_actor_update()
    _install_npu_device_binding_diagnostics()
