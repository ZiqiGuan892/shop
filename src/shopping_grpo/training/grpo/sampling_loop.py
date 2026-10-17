"""Bounded Shopping group collection for the server's RayPPOTrainer loop."""

from __future__ import annotations

import json
import math

from .dynamic_sampling import (
    append_training_diagnostic,
    build_rollout_diagnostics,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


def validate_sampling_modes(config):
    """Sampling retries need fresh rollouts and an actor update every ready batch."""
    rollout = config.actor_rollout_ref.rollout
    skip = config.get("skip", {}) or {}
    if (skip.get("rollout", {}) or {}).get("enable", False) or (rollout.get("skip", {}) or {}).get("enable", False):
        raise ValueError("Shopping dynamic sampling requires rollout cache replay disabled (skip.rollout.enable=false)")
    if not rollout.calculate_log_probs or int(config.trainer.get("critic_warmup", 0)) != 0:
        raise ValueError("Shopping dynamic sampling requires rollout log probs and critic_warmup=0")


def align_collected_batches(batches, pad_token_id=0):
    """Align left-padded prompts and right-padded responses before concatenation.

    Standard AgentLoop output already has fixed lengths. This also handles
    generation batches with different lengths without moving response labels
    into the prompt region or misaligning Qwen's multi-axis position IDs.
    """
    import torch
    import torch.nn.functional as F

    response_keys = {"responses", "response_mask", "rollout_log_probs", "rm_scores", "token_level_scores"}
    sequence_keys = {"input_ids", "attention_mask", "position_ids"}
    prompt_sizes = [int(b.batch["prompts"].shape[-1]) if "prompts" in b.batch else 0 for b in batches]
    response_sizes = [int(b.batch["responses"].shape[-1]) for b in batches]
    max_prompt, max_response = max(prompt_sizes), max(response_sizes)
    for batch, prompt, response in zip(batches, prompt_sizes, response_sizes, strict=True):
        for key, value in list(batch.batch.items()):
            fill = pad_token_id if key in {"prompts", "responses", "input_ids"} else 0
            if key == "prompts":
                value = F.pad(value, (max_prompt - prompt, 0), value=fill)
            elif key in response_keys:
                if value.shape[-1] != response:
                    raise ValueError(f"{key} is not aligned with responses")
                value = F.pad(value, (0, max_response - response), value=fill)
            elif key in sequence_keys:
                if value.shape[-1] != prompt + response:
                    raise ValueError(f"{key} is not aligned with prompt + response")
                value = torch.cat((
                    F.pad(value[..., :prompt], (max_prompt - prompt, 0), value=fill),
                    F.pad(value[..., prompt:], (0, max_response - response), value=fill),
                ), dim=-1)
            batch.batch[key] = value
    # Unknown fields (e.g. router replay) must never be silently padded incorrectly.
    keys = set(batches[0].batch.keys())
    for batch in batches[1:]:
        if set(batch.batch.keys()) != keys:
            raise ValueError("generation batches have different tensor fields")
        for key in keys:
            if batch.batch[key].shape[1:] != batches[0].batch[key].shape[1:]:
                raise ValueError(f"unsupported variable shape for collected field {key}")


class ShoppingDynamicSampler:
    """Collect full valid UID groups; never commit partial optimizer batches."""

    def __init__(self, config, *, diagnostics_path=None, pad_token_id=0):
        settings = config.shopping_dynamic_sampling
        self.target = int(config.data.train_batch_size)
        self.rollout_n = int(config.actor_rollout_ref.rollout.n)
        self.max_batches = int(settings.max_num_gen_batches)
        self.max_skips = int(settings.max_consecutive_skipped_updates)
        self.tolerance = float(settings.reward_tolerance)
        if settings.metric != "seq_reward" or min(self.target, self.rollout_n, self.max_batches, self.max_skips) <= 0:
            raise ValueError("invalid Shopping dynamic-sampling configuration")
        if self.tolerance < 0 or not math.isfinite(self.tolerance):
            raise ValueError("reward_tolerance must be finite and nonnegative")
        if str(config.algorithm.adv_estimator) != "grpo" or config.algorithm.use_kl_in_reward:
            raise ValueError("Shopping dynamic sampling requires GRPO without in-reward KL")
        if not config.algorithm.rollout_correction.get("bypass_mode", False):
            raise ValueError("Shopping dynamic sampling requires rollout-log-prob bypass mode")
        validate_sampling_modes(config)
        self.diagnostics_path = diagnostics_path
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.skipped_updates = 0
        self.consecutive_skips = 0
        self.last_metrics = {}
        self._reset()
        print("SHOPPING_GRPO_DYNAMIC_SAMPLING_CONFIG " + json.dumps({
            "train_batch_size": self.target, "rollout_n": self.rollout_n,
            "max_num_gen_batches": self.max_batches, "max_consecutive_skipped_updates": self.max_skips,
            "reward_tolerance": self.tolerance,
        }, sort_keys=True))

    def _reset(self):
        self.batches = []
        self.accepted = 0
        self.generation_batches = 0
        self.counts = dict.fromkeys((
            "num_groups", "num_trajectories", "dropped_group_count", "all_equal_group_count",
            "all_zero_utility_group_count", "all_purchase_success_group_count",
            "no_purchase_success_group_count", "sampling_invalid_group_count",
        ), 0)

    def _metrics(self, trained):
        count = max(self.counts["num_groups"], 1)
        metrics = {
            "group/generated": self.counts["num_groups"], "group/trained": trained,
            "group/accepted": self.accepted, "group/effective_ratio": trained / count,
            "group/sampling_invalid": self.counts["sampling_invalid_group_count"],
            "group/resample_batches": max(self.generation_batches - 1, 0),
            "rollout/generated_total": self.counts["num_trajectories"],
            "shopping_dynamic_sampling/skipped_updates_total": self.skipped_updates,
            "shopping_dynamic_sampling/consecutive_skips": self.consecutive_skips,
        }
        for name in ("all_equal", "all_zero_utility", "all_purchase_success", "no_purchase_success"):
            metrics[f"group/{name}_ratio"] = self.counts[f"{name}_group_count"] / count
        return metrics

    def add(self, batch, reward_tensor, *, global_step, logger, epoch):
        """Return a full training batch, or None while resampling/skipping."""
        if "shopping" not in batch.non_tensor_batch:
            raise RuntimeError("Shopping GRPO requires AgentLoop shopping diagnostics")
        infos = batch.non_tensor_batch["shopping"].tolist()
        uids = batch.non_tensor_batch["uid"].tolist()
        utility, success, invalid, reasons = extract_shopping_group_signals(infos)
        _, stats = select_reward_varying_groups(
            uids, reward_tensor.sum(dim=-1).detach().cpu().tolist(),
            terminal_utilities=utility, purchase_success=success,
            sampling_invalid=invalid, sampling_invalid_reasons=reasons, tolerance=self.tolerance,
        )
        if any(len(group["indices"]) != self.rollout_n for group in stats["groups"]):
            raise RuntimeError("Shopping UID group size differs from rollout.n")
        self.generation_batches += 1
        for key in self.counts:
            self.counts[key] += stats[key]
        accepted = set(stats["kept_uids"][: self.target - self.accepted])
        indices = [i for i, uid in enumerate(uids) if uid in accepted]
        diagnostic = {
            "generation_batch": self.generation_batches, "epoch": epoch,
            "groups": [{**g, "accepted": g["uid"] in accepted} for g in stats["groups"]],
            "accepted_prompts_after_batch": self.accepted + len(accepted), "target_prompts": self.target,
        }
        print("SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH " + json.dumps(diagnostic, sort_keys=True))
        append_training_diagnostic(
            self.diagnostics_path, "generation_batch", global_step + 1,
            **diagnostic, rollouts=build_rollout_diagnostics(uids, infos),
        )
        if indices:
            selected = batch.select_idxs(indices)
            selected.meta_info = dict(selected.meta_info)
            for key in ("global_token_num", "images_seqlens", "timing", "metrics"):
                selected.meta_info.pop(key, None)
            self.batches.append(selected)
            self.accepted += len(accepted)
        if self.accepted < self.target:
            if self.generation_batches >= self.max_batches:
                self.discard_pending(global_step=global_step, logger=logger, reason="max_num_gen_batches")
            return None

        align_collected_batches(self.batches, self.pad_token_id)
        ready = type(batch).concat(self.batches)
        self.consecutive_skips = 0
        self.last_metrics = self._metrics(self.accepted)
        self.last_metrics["shopping_dynamic_sampling/skipped_update"] = 0
        print("SHOPPING_GRPO_DYNAMIC_SAMPLING_READY " + json.dumps({
            "generation_batches": self.generation_batches, "trained_groups": self.accepted,
            "generated_groups": self.counts["num_groups"],
        }, sort_keys=True))
        self._reset()
        return ready

    def discard_pending(self, *, global_step, logger, reason):
        if not self.generation_batches:
            return
        self.skipped_updates += 1
        self.consecutive_skips += 1
        self.last_metrics = self._metrics(trained=0)
        self.last_metrics.update({
            "training/global_step": global_step, "training/optimizer_updated": 0,
            "shopping_dynamic_sampling/skipped_update": 1,
        })
        print("SHOPPING_GRPO_DYNAMIC_SAMPLING_SKIPPED " + json.dumps({
            "reason": reason, **self.last_metrics,
        }, sort_keys=True))
        logger.log(data=self.last_metrics, step=global_step)
        append_training_diagnostic(
            self.diagnostics_path, "skipped_update", global_step,
            reason=reason, metrics=self.last_metrics,
        )
        self._reset()
        if self.consecutive_skips >= self.max_skips:
            raise RuntimeError(f"Shopping reached max_consecutive_skipped_updates={self.max_skips} without an update")

    def state_dict(self):
        if self.generation_batches:
            raise RuntimeError("cannot checkpoint a partially collected Shopping batch")
        return {"skipped_updates": self.skipped_updates, "consecutive_skips": self.consecutive_skips}

    def load_state_dict(self, state):
        self.skipped_updates = int(state["skipped_updates"])
        self.consecutive_skips = int(state["consecutive_skips"])
        if min(self.skipped_updates, self.consecutive_skips) < 0:
            raise ValueError("invalid Shopping sampler checkpoint counters")
