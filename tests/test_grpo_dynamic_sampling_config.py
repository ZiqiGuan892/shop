"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

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
    def test_training_memory_budget_enforces_real_micro_batch_one(self):
        config = compose_runtime_config([])
        validate_training_memory_budget(config)
        self.assertEqual(config.data.max_response_length, 20480)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 24576)
        self.assertFalse(config.actor_rollout_ref.actor.use_dynamic_bsz)
        self.assertTrue(config.actor_rollout_ref.actor.calculate_entropy)
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
        unsafe_response = compose_runtime_config(["data.max_response_length=24576"])
        with self.assertRaisesRegex(SystemExit, "unsafe GRPO response budget"):
            validate_training_memory_budget(unsafe_response)

        dynamic_actor = compose_runtime_config(
            ["actor_rollout_ref.actor.use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(SystemExit, "actor.use_dynamic_bsz must be false"):
            validate_training_memory_budget(dynamic_actor)

        dynamic_rollout_log_prob = compose_runtime_config(
            ["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(
            SystemExit, "rollout.log_prob_use_dynamic_bsz must be false"
        ):
            validate_training_memory_budget(dynamic_rollout_log_prob)

    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = compose_runtime_config(
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
