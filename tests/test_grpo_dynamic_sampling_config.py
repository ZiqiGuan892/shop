"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf
from scripts import check_grpo_runtime as runtime

from scripts.check_grpo_runtime import (
    compose_runtime_config,
    validate_dynamic_sampling,
    validate_training_memory_budget,
)


class DynamicSamplingConfigTest(unittest.TestCase):
    @staticmethod
    def compose_project_config(overrides):
        """Exercise real Hydra override parsing without the vendor training stack."""
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/grpo.yaml")
        del config["defaults"]
        del config["hydra"]
        with tempfile.TemporaryDirectory(prefix="shopping-grpo-config-") as temporary:
            config_path = Path(temporary) / "project-grpo.yaml"
            OmegaConf.save(config, config_path)
            with mock.patch.dict(os.environ, {
                "GRPO_CONFIG_DIR": temporary,
                "GRPO_CONFIG_NAME": config_path.stem,
            }):
                return compose_runtime_config(overrides)

    def test_training_memory_budget_enforces_real_micro_batch_one(self):
        config = self.compose_project_config([])
        validate_training_memory_budget(config)
        self.assertEqual(config.data.max_response_length, 20480)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 24576)
        self.assertFalse(config.actor_rollout_ref.actor.use_dynamic_bsz)
        self.assertFalse(config.actor_rollout_ref.actor.calculate_entropy)
        self.assertEqual(config.actor_rollout_ref.actor.entropy_coeff, 0.0)
        self.assertTrue(config.actor_rollout_ref.model.enable_activation_offload)
        self.assertTrue(config.actor_rollout_ref.model.enable_gradient_checkpointing)
        self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.reshard_after_forward)
        self.assertFalse(config.actor_rollout_ref.actor.fsdp_config.forward_prefetch)
        self.assertEqual(
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(
            config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
        )
        self.assertEqual(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz)
        self.assertEqual(
            config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu, 1
        )

    def test_training_memory_budget_rejects_unsafe_overrides(self):
        unsafe_response = self.compose_project_config(["data.max_response_length=24576"])
        with self.assertRaisesRegex(SystemExit, "unsafe GRPO response budget"):
            validate_training_memory_budget(unsafe_response)

        dynamic_actor = self.compose_project_config(
            ["actor_rollout_ref.actor.use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(SystemExit, "actor.use_dynamic_bsz must be false"):
            validate_training_memory_budget(dynamic_actor)

        dynamic_rollout_log_prob = self.compose_project_config(
            ["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(
            SystemExit, "rollout.log_prob_use_dynamic_bsz must be false"
        ):
            validate_training_memory_budget(dynamic_rollout_log_prob)

    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = self.compose_project_config(
            [
                "shopping_dynamic_sampling.enable=true",
                "shopping_dynamic_sampling.metric=seq_reward",
                "shopping_dynamic_sampling.max_num_gen_batches=3",
                "shopping_dynamic_sampling.max_consecutive_skipped_updates=10",
                "shopping_dynamic_sampling.reward_tolerance=1e-8",
            ]
        )
        self.assertTrue(config.shopping_dynamic_sampling.enable)
        self.assertEqual(config.shopping_dynamic_sampling.metric, "seq_reward")
        self.assertEqual(config.shopping_dynamic_sampling.max_num_gen_batches, 3)
        self.assertEqual(
            config.shopping_dynamic_sampling.max_consecutive_skipped_updates, 10
        )
        self.assertEqual(config.shopping_dynamic_sampling.reward_tolerance, 1.0e-8)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertTrue(config.actor_rollout_ref.rollout.calculate_log_probs)

    def test_enabled_config_requires_reviewed_direct_trainer(self):
        # Source verification needs no installed trainer or model dependencies.
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/grpo.yaml")
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            trainer_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "requires the reviewed server trainer"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.9.0.dev"})

            trainer_source.write_text("# SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "requires the reviewed server trainer"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.9.0.dev"})
            fixture_hash = runtime.normalized_sha256(trainer_source)
            with mock.patch.object(runtime, "EXPECTED_TRAINER_SHA256", fixture_hash):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.9.0.dev"})
                validate_dynamic_sampling(config, verl_source, {"verl": "0.9.0.dev0+ascend"})
                with self.assertRaisesRegex(SystemExit, "requires veRL 0.9.0.dev source"):
                    validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

    def test_preflight_rejects_rollout_replay_and_critic_warmup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            trainer_source.write_text("# reviewed fixture\n", encoding="utf-8")
            fixture_hash = runtime.normalized_sha256(trainer_source)
            cases = (
                ("skip.rollout.enable", True, "rollout cache replay disabled"),
                ("actor_rollout_ref.rollout.skip.enable", True, "rollout cache replay disabled"),
                ("trainer.critic_warmup", 1, "critic_warmup=0"),
            )
            with mock.patch.object(runtime, "EXPECTED_TRAINER_SHA256", fixture_hash):
                for key, value, message in cases:
                    with self.subTest(override=key):
                        config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/grpo.yaml")
                        OmegaConf.update(config, key, value, force_add=True)
                        with self.assertRaisesRegex(SystemExit, message):
                            validate_dynamic_sampling(config, verl_source, {"verl": "0.9.0.dev"})

    def test_zero_entropy_weight_can_omit_entropy_backward_without_changing_ppo_gradient(self):
        import torch

        logits = torch.tensor(
            [[0.5, -0.2, 0.8, -0.4], [0.1, 0.4, -0.3, 0.6], [-0.7, 0.2, 0.5, 0.9]],
            dtype=torch.float64,
        )
        actions = torch.tensor([0, 1, 3])
        old_log_probs = torch.tensor([-1.2, -1.1, -1.4], dtype=torch.float64)
        advantages = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64)
        entropy_backward_calls = []

        def ppo_loss(parameters, calculate_entropy):
            log_probs = parameters.log_softmax(dim=-1)
            selected = log_probs.gather(-1, actions[:, None]).squeeze(-1)
            ratio = (selected - old_log_probs).exp()
            loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
            if calculate_entropy:
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                entropy.register_hook(lambda gradient: entropy_backward_calls.append(gradient.detach().clone()))
                loss = loss - 0.0 * entropy
            return loss

        with_entropy = logits.clone().requires_grad_()
        original_loss = ppo_loss(with_entropy, True)
        original_loss.backward()
        self.assertEqual(len(entropy_backward_calls), 1)
        self.assertEqual(entropy_backward_calls[0].item(), 0.0)

        entropy_backward_calls.clear()
        without_entropy = logits.clone().requires_grad_()
        reduced_memory_loss = ppo_loss(without_entropy, False)
        reduced_memory_loss.backward()
        self.assertEqual(entropy_backward_calls, [])
        torch.testing.assert_close(original_loss, reduced_memory_loss, rtol=0, atol=0)
        torch.testing.assert_close(with_entropy.grad, without_entropy.grad, rtol=0, atol=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
