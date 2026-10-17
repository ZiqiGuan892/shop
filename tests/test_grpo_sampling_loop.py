"""CPU behavior checks for bounded collection before a real optimizer update."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from shopping_grpo.training.grpo.sampling_loop import ShoppingDynamicSampler


class FakeDataProto:
    """Only the documented DataProto indexing/concatenation operations are mocked."""

    def __init__(self, tensors, non_tensors, meta_info=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors
        self.meta_info = dict(meta_info or {})

    def __len__(self):
        return len(self.non_tensor_batch["uid"])

    def select_idxs(self, indices):
        return type(self)(
            {k: v[indices] for k, v in self.batch.items()},
            {k: v[indices] for k, v in self.non_tensor_batch.items()},
            self.meta_info,
        )

    @classmethod
    def concat(cls, batches):
        # DataProto rejects inconsistent batch metadata; do not hide that class
        # of errors in a more permissive fake.
        if any(b.meta_info != batches[0].meta_info for b in batches[1:]):
            raise AssertionError("inconsistent generation metadata")
        return cls(
            {k: torch.cat([b.batch[k] for b in batches]) for k in batches[0].batch},
            {k: np.concatenate([b.non_tensor_batch[k] for b in batches])
             for k in batches[0].non_tensor_batch},
            batches[0].meta_info,
        )


class ConfigNode(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def sampling_config(*, target=2, n=4, max_batches=3, max_skips=10):
    return ConfigNode(
        shopping_dynamic_sampling=SimpleNamespace(
            metric="seq_reward", max_num_gen_batches=max_batches,
            max_consecutive_skipped_updates=max_skips, reward_tolerance=1e-8,
        ),
        data=SimpleNamespace(train_batch_size=target),
        actor_rollout_ref=SimpleNamespace(
            rollout=ConfigNode(n=n, calculate_log_probs=True),
        ),
        algorithm=SimpleNamespace(
            adv_estimator="grpo", use_kl_in_reward=False,
            rollout_correction={"bypass_mode": True},
        ),
        trainer=ConfigNode(critic_warmup=0),
    )


def make_batch(prefix="a", utilities=(0, 0, 0, 0, 0, 1, 0, 1), *, prompt=2,
               response=3, offset=10, invalid_rows=()):
    count = len(utilities)
    uids = np.array([f"{prefix}-{i // 4}" for i in range(count)], dtype=object)
    rows = torch.arange(offset, offset + count)
    prompts = rows[:, None].expand(count, prompt).clone()
    responses = (rows + 100)[:, None].expand(count, response).clone()
    rewards = torch.zeros(count, response)
    rewards[:, -1] = torch.tensor(utilities, dtype=torch.float32)
    positions = torch.arange(prompt + response).expand(count, 3, -1).clone()
    return FakeDataProto(
        {
            "prompts": prompts, "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": torch.ones(count, prompt + response, dtype=torch.int64),
            "position_ids": positions,
            "response_mask": torch.ones(count, response, dtype=torch.int64),
            "rollout_log_probs": (rows.float() / 100)[:, None].expand(count, response).clone(),
            "token_level_scores": rewards.clone(), "rm_scores": rewards.clone(),
        },
        {
            "uid": uids,
            "extra_info": np.array([{"task_id": int(row)} for row in rows], dtype=object),
            "shopping": np.array([
                {"infrastructure_invalid": i in invalid_rows,
                 "reward": {"terminal_utility": float(value),
                            "purchase_success": bool(value > 0),
                            "sampling_invalid": i in invalid_rows}}
                for i, value in enumerate(utilities)
            ], dtype=object),
        },
        {"global_token_num": [prompt + response] * count,
         "images_seqlens": [], "timing": {"generation": offset},
         "metrics": {"old_batch": offset}, "reward_extra_keys": ["shopping"],
         "temperature": 0.7},
    )


class Recorder:
    def __init__(self):
        self.records = []

    def log(self, *, data, step):
        self.records.append((step, dict(data)))


class ShoppingSamplingLoopTest(unittest.TestCase):
    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.logger = Recorder()

    def sampler(self, **kwargs):
        return ShoppingDynamicSampler(sampling_config(**kwargs), pad_token_id=99)

    def add(self, sampler, batch, *, epoch=0, step=7):
        return sampler.add(batch, batch.batch["rm_scores"], global_step=step,
                           logger=self.logger, epoch=epoch)

    def test_resampling_collects_complete_groups_and_all_aligned_fields(self):
        sampler = self.sampler()
        self.assertIsNone(self.add(sampler, make_batch("a", offset=10)))
        ready = self.add(sampler, make_batch("b", offset=20))
        expected = [14, 15, 16, 17, 24, 25, 26, 27]
        self.assertEqual(len(ready), 8)
        self.assertEqual([x["task_id"] for x in ready.non_tensor_batch["extra_info"]], expected)
        self.assertEqual(ready.non_tensor_batch["uid"].tolist(), ["a-1"] * 4 + ["b-1"] * 4)
        self.assertEqual(ready.batch["responses"][:, 0].tolist(), [x + 100 for x in expected])
        torch.testing.assert_close(ready.batch["rollout_log_probs"][:, 0], torch.tensor(expected) / 100)
        self.assertEqual(ready.batch["token_level_scores"].sum(-1).tolist(), [0, 1, 0, 1] * 2)
        self.assertEqual(ready.meta_info, {"reward_extra_keys": ["shopping"], "temperature": 0.7})
        self.assertEqual(sampler.last_metrics["group/resample_batches"], 1)
        self.assertEqual(sampler.last_metrics["group/generated"], 4)
        self.assertEqual(sampler.last_metrics["group/trained"], 2)
        self.assertEqual(sampler.generation_batches, 0)
        self.assertEqual(self.logger.records, [])

    def test_max_generation_batches_discards_incomplete_optimizer_batch(self):
        sampler = self.sampler()
        self.assertIsNone(self.add(sampler, make_batch("one")))
        for prefix in ("zero-a", "zero-b"):
            self.assertIsNone(self.add(sampler, make_batch(prefix, (0,) * 8)))
        self.assertEqual(sampler.batches, [])
        self.assertEqual(sampler.accepted, 0)
        self.assertEqual(sampler.skipped_updates, 1)
        self.assertEqual(self.logger.records[0][0], 7)
        self.assertEqual(self.logger.records[0][1]["training/global_step"], 7)
        self.assertEqual(self.logger.records[0][1]["training/optimizer_updated"], 0)
        self.assertEqual(self.logger.records[0][1]["group/trained"], 0)
        # A later successful batch must never reuse the discarded one-group buffer.
        ready = self.add(sampler, make_batch("fresh", (0, 1, 0, 1) * 2))
        self.assertEqual(ready.non_tensor_batch["uid"].tolist(), ["fresh-0"] * 4 + ["fresh-1"] * 4)
        self.assertEqual(sampler.consecutive_skips, 0)
        self.assertEqual(sampler.skipped_updates, 1)

    def test_ten_consecutive_skips_stop_without_a_ready_batch(self):
        sampler = self.sampler()
        for generation in range(29):
            self.assertIsNone(self.add(sampler, make_batch(str(generation), (0,) * 8)))
        with self.assertRaisesRegex(RuntimeError, "max_consecutive_skipped_updates=10"):
            self.add(sampler, make_batch("last", (0,) * 8))
        self.assertEqual(len(self.logger.records), 10)
        self.assertEqual({step for step, _ in self.logger.records}, {7})
        self.assertEqual(sampler.consecutive_skips, 10)

    def test_partial_collection_survives_epoch_boundary(self):
        sampler = self.sampler()
        self.assertIsNone(self.add(sampler, make_batch("epoch0"), epoch=0))
        ready = self.add(sampler, make_batch("epoch1"), epoch=1)
        self.assertEqual(ready.non_tensor_batch["uid"].tolist(), ["epoch0-1"] * 4 + ["epoch1-1"] * 4)
        self.assertEqual(sampler.skipped_updates, 0)

    def test_extra_kept_groups_are_capped_at_target_without_splitting_uid(self):
        sampler = self.sampler()
        self.assertIsNone(self.add(sampler, make_batch("first")))
        ready = self.add(sampler, make_batch("second", (0, 1, 0, 1) * 2))
        self.assertEqual(ready.non_tensor_batch["uid"].tolist(), ["first-1"] * 4 + ["second-0"] * 4)
        self.assertEqual(len(ready), 8)

    def test_any_invalid_rollout_rejects_entire_group(self):
        sampler = self.sampler(target=1)
        ready = self.add(sampler, make_batch("invalid", (0, 1, 0, 1) * 2, invalid_rows=(2,)))
        self.assertEqual(ready.non_tensor_batch["uid"].tolist(), ["invalid-1"] * 4)
        self.assertEqual(sampler.last_metrics["group/sampling_invalid"], 1)

    def test_incomplete_uid_is_rejected_before_collecting(self):
        sampler = self.sampler()
        with self.assertRaisesRegex(RuntimeError, "group size differs"):
            self.add(sampler, make_batch("bad", (0, 1, 0)))
        self.assertEqual(sampler.generation_batches, 0)

    def test_different_sequence_lengths_preserve_prompt_response_boundaries(self):
        sampler = self.sampler()
        self.assertIsNone(self.add(sampler, make_batch("short", prompt=2, response=4, offset=10)))
        ready = self.add(sampler, make_batch("long", prompt=4, response=2, offset=20))
        self.assertEqual(tuple(ready.batch["input_ids"].shape), (8, 8))
        self.assertEqual(ready.batch["input_ids"][0].tolist(), [99, 99, 14, 14, 114, 114, 114, 114])
        self.assertEqual(ready.batch["input_ids"][4].tolist(), [24, 24, 24, 24, 124, 124, 99, 99])
        self.assertEqual(ready.batch["attention_mask"][0].tolist(), [0, 0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(ready.batch["attention_mask"][4].tolist(), [1, 1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(ready.batch["position_ids"][0, 0].tolist(), [0, 0, 0, 1, 2, 3, 4, 5])
        self.assertEqual(ready.batch["position_ids"][4, 2].tolist(), [0, 1, 2, 3, 4, 5, 0, 0])
        self.assertEqual(ready.batch["response_mask"][4].tolist(), [1, 1, 0, 0])
        torch.testing.assert_close(ready.batch["rollout_log_probs"][4], torch.tensor([.24, .24, 0, 0]))
        self.assertEqual(ready.batch["rm_scores"][5].tolist(), [0, 1, 0, 0])

    def test_unsupported_variable_tensor_field_fails_closed(self):
        sampler = self.sampler()
        first, second = make_batch("a"), make_batch("b")
        first.batch["router_replay"] = torch.zeros(8, 3)
        second.batch["router_replay"] = torch.zeros(8, 5)
        self.assertIsNone(self.add(sampler, first))
        with self.assertRaisesRegex(ValueError, "unsupported variable shape.*router_replay"):
            self.add(sampler, second)

    def test_checkpoint_state_preserves_skip_budget_and_rejects_partial_collection(self):
        sampler = self.sampler(max_batches=1)
        self.assertIsNone(self.add(sampler, make_batch("skip", (0,) * 8)))
        restored = self.sampler()
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(restored.skipped_updates, 1)
        self.assertEqual(restored.consecutive_skips, 1)
        self.assertIsNone(self.add(restored, make_batch("pending")))
        with self.assertRaisesRegex(RuntimeError, "partially collected"):
            restored.state_dict()

    def test_final_incomplete_collection_is_logged_once_as_skipped(self):
        sampler = self.sampler()
        self.assertIsNone(self.add(sampler, make_batch("end")))
        sampler.discard_pending(global_step=7, logger=self.logger, reason="training_data_exhausted")
        sampler.discard_pending(global_step=7, logger=self.logger, reason="training_data_exhausted")
        self.assertEqual(len(self.logger.records), 1)
        self.assertEqual(sampler.accepted, 0)

    def test_diagnostics_separate_pending_next_step_from_committed_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampling.jsonl"
            sampler = ShoppingDynamicSampler(sampling_config(max_batches=1), diagnostics_path=path)
            self.add(sampler, make_batch("zero", (0,) * 8), step=7)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["event"], r["global_step"]) for r in records],
                         [("generation_batch", 8), ("skipped_update", 7)])
        self.assertEqual(len(records[0]["rollouts"]), 8)

    def test_global_rollout_cache_is_rejected_for_resampling_same_step(self):
        config = sampling_config()
        config.skip = {"rollout": {"enable": True}}
        with self.assertRaisesRegex(ValueError, "skip"):
            ShoppingDynamicSampler(config)

    def test_rollout_local_cache_is_rejected_for_resampling_same_step(self):
        config = sampling_config()
        config.actor_rollout_ref.rollout.skip = {"enable": True}
        with self.assertRaisesRegex(ValueError, "skip"):
            ShoppingDynamicSampler(config)


if __name__ == "__main__":
    unittest.main()
