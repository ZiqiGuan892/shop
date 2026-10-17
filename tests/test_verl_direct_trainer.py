"""Execute the shipped trainer methods with CPU workers and deterministic rollouts.

The function bodies are extracted from the actual server integration, so these
tests exercise its continue/update/checkpoint control flow without importing
Ray, vLLM, or launching training processes.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from shopping_grpo.training.grpo.sampling_loop import ShoppingDynamicSampler
from shopping_grpo.training.grpo.trainer_contract import (
    EXPECTED_TRAINER_SHA256, normalized_sha256,
)
from tests.test_grpo_sampling_loop import FakeDataProto, Recorder, make_batch, sampling_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "integrations/verl/ray_trainer.py"


class TensorBatch(dict):
    def __len__(self):
        return next(iter(self.values())).shape[0] if self.keys() else 0


class LoopDataProto(FakeDataProto):
    def __init__(self, tensors, non_tensors, meta_info=None):
        super().__init__(TensorBatch(tensors), non_tensors, meta_info)

    @classmethod
    def from_single_dict(cls, batch):
        return copy.deepcopy(batch)

    def repeat(self, repeat_times, interleave=True):
        assert interleave
        return self.select_idxs(np.repeat(np.arange(len(self)), repeat_times).tolist())

    def slice(self, start, end):
        return self.select_idxs(list(range(len(self)))[start:end])

    def union(self, other):
        return type(self)({**self.batch, **other.batch},
                          {**self.non_tensor_batch, **other.non_tensor_batch},
                          {**self.meta_info, **other.meta_info})


class Progress:
    def __init__(self, **kwargs):
        self.updates = 0
        self.closed = False

    def update(self, count):
        self.updates += count

    def close(self):
        self.closed = True


@contextlib.contextmanager
def timer(name, timings, **kwargs):
    try:
        yield
    finally:
        timings[name] = 0.01


def extract_methods(*names, **namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    methods = [copy.deepcopy(node) for node in trainer.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    result = {"__name__": "trainer_cpu_test", "os": os, "sys": sys, "json": json, "torch": torch,
              "np": np, "uuid": uuid, "DataProto": LoopDataProto, **namespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(SOURCE), "exec"), result)
    return result


def loop_config(*, batches=2, epochs=2, max_batches=3):
    base = sampling_config(max_batches=max_batches)
    return OmegaConf.create({
        "shopping_dynamic_sampling": {"enable": True, **vars(base.shopping_dynamic_sampling)},
        "data": vars(base.data),
        "actor_rollout_ref": {"rollout": {**vars(base.actor_rollout_ref.rollout), "temperature": .7},
                              "actor": {"policy_loss": {}, "checkpoint": {"async_save": False}}},
        "algorithm": {**vars(base.algorithm), "gamma": 1, "lam": 1, "norm_adv_by_std_in_grpo": False},
        "trainer": {"critic_warmup": 0, "project_name": "cpu", "experiment_name": "cpu",
                    "logger": ["console"], "val_before_train": False, "total_epochs": epochs,
                    "balance_batch": True, "save_freq": 1, "test_freq": 0,
                    "esi_redundant_time": 0, "resume_mode": "disable", "default_hdfs_dir": None,
                    "default_local_dir": "unused", "del_local_ckpt_after_load": False},
        "global_profiler": {"steps": None, "profile_continuous_steps": False},
    })


class FitHarness:
    def __init__(self, patterns, *, epochs=2, batches=2, total_steps=1, restored=None, fail_update=False):
        self.events = []
        self.config = loop_config(batches=batches, epochs=epochs)
        self.total_training_steps = total_steps
        self._dump_executor = SimpleNamespace(_shutdown=False)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.actor_rollout_wg = SimpleNamespace()
        self.use_rm = False
        self.use_reference_policy = True
        self.use_critic = False
        self.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 8)
        self.train_dataloader = [LoopDataProto(
            {"prompts": torch.full((2, 2), i, dtype=torch.long)},
            {"uid": np.array(["old", "old"], dtype=object),
             "multi_modal_inputs": np.array([{}, {}], dtype=object)},
        ) for i in range(batches)]
        self.train_dataset = SimpleNamespace(on_batch_end=lambda **kw: self.events.append("batch_end"))
        self.checkpoint_manager = SimpleNamespace(
            update_weights=lambda step: self.events.append(("sync", step)),
            sleep_replicas=lambda: self.events.append("sleep"),
        )
        self.patterns = iter(patterns)
        self.async_rollout_manager = SimpleNamespace(generate_sequences=self.generate)
        self.updates = []
        self.checkpoints = []
        self.restored = restored
        self.fail_update = fail_update
        self.progress = None
        self.timing_snapshots = []

    def generate(self, incoming):
        self.events.append("generate")
        pattern = next(self.patterns)
        utilities = {"zero": (0,) * 8, "one": (0,) * 4 + (0, 1, 0, 1),
                     "full": (0, 1, 0, 1) * 2}[pattern]
        batch = make_batch("rollout", utilities)
        batch.non_tensor_batch["uid"] = incoming.non_tensor_batch["uid"]
        batch.non_tensor_batch["multi_modal_inputs"] = np.array([{} for _ in range(8)], dtype=object)
        keys = ("full", "strict", "native", "semantic", "total", "efficiency", "penalty_overlong",
                "penalty_unfinished", "penalty_repeat", "repeat_action_rate", "r_type", "r_att",
                "r_option", "r_price")
        for info in batch.non_tensor_batch["shopping"]:
            info["reward"].update(dict.fromkeys(keys, 0))
        return LoopDataProto(batch.batch, batch.non_tensor_batch, batch.meta_info)

    def _load_checkpoint(self):
        if self.restored:
            self.global_steps, self._shopping_epoch, self._shopping_batches_in_epoch = self.restored

    def _get_gen_batch(self, batch):
        return copy.deepcopy(batch)

    def _balance_batch(self, batch, metrics):
        self.events.append("balance")

    def _compute_ref_log_prob(self, batch):
        self.events.append("reference")
        return LoopDataProto({"ref_log_prob": torch.zeros_like(batch.batch["rollout_log_probs"])}, {}, {})

    def _update_actor(self, batch):
        self.events.append("actor")
        if self.fail_update:
            raise RuntimeError("worker update failed")
        self.updates.append(copy.deepcopy(batch))
        return SimpleNamespace(meta_info={"metrics": {}})

    def _save_checkpoint(self):
        self.checkpoints.append((self.global_steps, self._shopping_epoch, self._shopping_batches_in_epoch))

    def _start_profiling(self, enabled):
        pass

    def _stop_profiling(self, enabled):
        pass

    def _shutdown_dump_executor(self):
        self.events.append("shutdown")


class DirectTrainerFitTest(unittest.TestCase):
    def setUp(self):
        capture = contextlib.redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        self.logger = Recorder()
        self.skip_steps = []
        tracking = types.ModuleType("verl.utils.tracking")
        tracking.Tracking = lambda **kwargs: self.logger
        bypass = types.ModuleType("verl.trainer.ppo.rollout_corr_helper")
        bypass.apply_bypass_mode = lambda batch, **kwargs: batch.batch.update(
            old_log_probs=batch.batch["rollout_log_probs"].clone())
        patch = mock.patch.dict(sys.modules, {"verl.utils.tracking": tracking,
                                            "verl.trainer.ppo.rollout_corr_helper": bypass})
        patch.start()
        self.addCleanup(patch.stop)

    def run_fit(self, harness):
        def progress(**kwargs):
            harness.progress = Progress(**kwargs)
            return harness.progress
        noop = lambda *args, **kwargs: {}
        def capture_timings(*, batch, timing_raw):
            harness.timing_snapshots.append(dict(timing_raw))
            return {}
        methods = extract_methods(
            "fit", tqdm=progress, pprint=lambda *args: None, marked_timer=timer,
            SkipManager=SimpleNamespace(init=lambda config: None, set_step=self.skip_steps.append),
            AdvantageEstimator=SimpleNamespace(REMAX="remax", GDPO="gdpo"),
            Role=SimpleNamespace(RefPolicy="ref"),
            extract_reward=lambda batch: (batch.batch["rm_scores"], {}),
            compute_advantage=lambda batch, **kwargs: batch,
            should_save_ckpt_esi=lambda **kwargs: False, reduce_metrics=lambda x: x,
            compute_data_metrics=noop, compute_timing_metrics=capture_timings, compute_throughout_metrics=noop,
            compute_variance_proxy_metrics=noop, compute_spec_decode_metrics=noop,
        )
        methods["fit"](harness)

    def test_resampling_defers_actor_reference_sleep_and_checkpoint_until_ready(self):
        harness = FitHarness(["one", "one"])
        self.run_fit(harness)
        self.assertEqual(harness.events[:6], [("sync", 0), "generate", "batch_end", "generate", "balance", "sleep"])
        self.assertLess(harness.events.index("sleep"), harness.events.index("reference"))
        self.assertLess(harness.events.index("reference"), harness.events.index("actor"))
        self.assertEqual(len(harness.updates), 1)
        self.assertEqual(harness.global_steps, 1)
        self.assertEqual(harness.checkpoints, [(1, 0, 2)])
        self.assertEqual(harness.progress.updates, 1)
        selected = harness.updates[0]
        self.assertEqual(selected.meta_info["global_token_num"], [5] * 8)
        torch.testing.assert_close(selected.batch["old_log_probs"], selected.batch["rollout_log_probs"])
        self.assertEqual(selected.batch["token_level_scores"].sum(-1).tolist(), [0, 1, 0, 1] * 2)
        self.assertEqual(len(harness.timing_snapshots), 1)
        self.assertAlmostEqual(harness.timing_snapshots[0]["gen"], 0.02)
        self.assertAlmostEqual(harness.timing_snapshots[0]["step"], 0.02)

    def test_max_batches_skip_does_not_commit_step_or_checkpoint(self):
        harness = FitHarness(["one", "zero", "zero"], epochs=1, batches=3)
        self.run_fit(harness)
        self.assertEqual(harness.global_steps, 0)
        self.assertEqual(harness.updates, [])
        self.assertEqual(harness.checkpoints, [])
        self.assertEqual(harness.progress.updates, 0)
        self.assertNotIn("reference", harness.events)
        self.assertEqual([x for x in harness.events if isinstance(x, tuple)], [("sync", 0)])
        self.assertEqual(self.skip_steps, [1, 1, 1, 1])
        self.assertEqual(self.logger.records[0][1]["training/optimizer_updated"], 0)

    def test_collection_crosses_epoch_without_discarding_valid_group(self):
        harness = FitHarness(["one", "one"], epochs=2, batches=1)
        self.run_fit(harness)
        self.assertEqual(harness.checkpoints, [(1, 1, 1)])
        self.assertEqual(len(harness.updates), 1)
        self.assertEqual(harness._shopping_sampler.skipped_updates, 0)

    def test_ten_skipped_updates_raise_without_weight_sync_or_checkpoint(self):
        harness = FitHarness(["zero"] * 30, epochs=1, batches=30)
        with self.assertRaisesRegex(RuntimeError, "max_consecutive_skipped_updates=10"):
            self.run_fit(harness)
        self.assertEqual(harness.global_steps, 0)
        self.assertEqual(harness.checkpoints, [])
        self.assertEqual(harness.updates, [])
        self.assertEqual(harness.events.count("generate"), 30)
        self.assertEqual({step for step, _ in self.logger.records}, {0})
        self.assertEqual(harness.events[-2:], ["sleep", "shutdown"])
        self.assertTrue(harness.progress.closed)

    def test_actor_failure_does_not_commit_optimizer_step_or_checkpoint(self):
        harness = FitHarness(["full"], fail_update=True)
        with self.assertRaisesRegex(RuntimeError, "worker update failed"):
            self.run_fit(harness)
        self.assertEqual(harness.global_steps, 0)
        self.assertEqual(harness.checkpoints, [])
        self.assertEqual(harness.progress.updates, 0)
        self.assertEqual(harness.events.count("sleep"), 1)
        self.assertEqual(harness.events[-1], "shutdown")
        self.assertTrue(harness.progress.closed)

    def test_generation_error_is_preserved_when_cleanup_also_fails(self):
        harness = FitHarness([])

        def generate_failure(batch):
            harness.events.append("generate")
            raise RuntimeError("original generation failure")

        def cleanup_failure():
            harness.events.append("sleep")
            raise ValueError("secondary cleanup failure")

        harness.async_rollout_manager.generate_sequences = generate_failure
        harness.checkpoint_manager.sleep_replicas = cleanup_failure
        with self.assertRaisesRegex(RuntimeError, "original generation failure"):
            self.run_fit(harness)
        self.assertEqual(harness.events[-2:], ["sleep", "shutdown"])
        self.assertEqual(harness.global_steps, 0)
        self.assertTrue(harness.progress.closed)

    def test_cleanup_failure_without_training_error_is_reported(self):
        harness = FitHarness(["one"], epochs=1, batches=1)

        def cleanup_failure():
            harness.events.append("sleep")
            raise ValueError("cleanup failure without prior exception")

        harness.checkpoint_manager.sleep_replicas = cleanup_failure
        with self.assertRaisesRegex(ValueError, "cleanup failure without prior exception"):
            self.run_fit(harness)
        self.assertEqual(harness.events[-2:], ["sleep", "shutdown"])
        self.assertTrue(harness.progress.closed)

    def test_resume_uses_saved_epoch_cursor_not_optimizer_step_quotient(self):
        harness = FitHarness(["full"], epochs=4, batches=10, total_steps=2, restored=(1, 3, 6))
        self.run_fit(harness)
        self.assertEqual(harness.checkpoints, [(2, 3, 7)])
        self.assertEqual(harness.global_steps, 2)
        self.assertEqual(harness.events[0], ("sync", 1))
        self.assertEqual(self.logger.records[-1][0], 2)

    def test_completed_resume_does_not_generate_again(self):
        harness = FitHarness([], total_steps=1, restored=(1, 0, 2))
        self.run_fit(harness)
        self.assertEqual(harness.updates, [])
        self.assertEqual(harness.checkpoints, [])
        self.assertEqual(harness.global_steps, 1)
        self.assertTrue(harness.progress.closed)


class DirectTrainerResumeTest(unittest.TestCase):
    def test_checkpoint_saves_consumed_batch_cursor_and_sampler_separate_from_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = FitHarness([], batches=10)
            harness.config.trainer.default_local_dir = directory
            harness.global_steps = 7
            harness._shopping_epoch = 3
            harness._shopping_batches_in_epoch = 9
            harness._shopping_sampler = ShoppingDynamicSampler(harness.config)
            harness._shopping_sampler.skipped_updates = 2
            harness.actor_rollout_wg.save_checkpoint = mock.Mock()
            loader = mock.MagicMock()
            loader.__len__.return_value = 10
            loader.state_dict.return_value = {"cursor": 9, "sampler_epoch": 3}
            harness.train_dataloader = loader
            fs = types.ModuleType("verl.utils.fs")
            fs.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
            save = extract_methods("_save_checkpoint", Role=SimpleNamespace(Critic="critic"))["_save_checkpoint"]
            with mock.patch.dict(sys.modules, {"verl.utils.fs": fs}):
                save(harness)
            folder = Path(directory) / "global_step_7"
            state = json.loads((folder / "shopping_sampler_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state, {"schema_version": 1, "global_step": 7, "epoch": 3,
                                     "batches_in_epoch": 9, "loader_batches": 10,
                                     "sampler": {"skipped_updates": 2, "consecutive_skips": 0}})
            self.assertEqual(torch.load(folder / "data.pt", weights_only=False), {"cursor": 9, "sampler_epoch": 3})
            self.assertEqual((Path(directory) / "latest_checkpointed_iteration.txt").read_text(), "7")
            self.assertFalse((folder / "shopping_sampler_state.json.tmp").exists())

    def test_missing_dynamic_resume_state_stops_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = FitHarness([], batches=10)
            harness.config.trainer.resume_mode = "resume_path"
            harness.config.trainer.resume_from_path = str(Path(directory) / "global_step_7")
            harness.config.trainer.default_local_dir = directory
            harness._shopping_sampler = ShoppingDynamicSampler(harness.config)
            harness.actor_rollout_wg.load_checkpoint = mock.Mock()
            load = extract_methods("_load_checkpoint", find_latest_ckpt_path=lambda path: None,
                                   Role=SimpleNamespace(Critic="critic"))["_load_checkpoint"]
            with self.assertRaisesRegex(RuntimeError, "requires shopping_sampler_state.json"):
                load(harness)
            harness.actor_rollout_wg.load_checkpoint.assert_not_called()

    def test_partial_epoch_restores_dataloader_cursor_independent_of_step_modulo(self):
        # global_step=10 is divisible by loader length, but dynamic sampling has
        # consumed 7 batches of epoch 3; the original modulo shortcut is wrong.
        self.check_resume(step=10, epoch=3, consumed=7, expect_epoch=3, expect_consumed=7, loads_cursor=True)

    def test_true_epoch_boundary_advances_epoch_without_loading_exhausted_cursor(self):
        self.check_resume(step=7, epoch=3, consumed=10, expect_epoch=4, expect_consumed=0, loads_cursor=False)

    def check_resume(self, *, step, epoch, consumed, expect_epoch, expect_consumed, loads_cursor):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / f"global_step_{step}"
            folder.mkdir()
            (folder / "data.pt").write_bytes(b"fake-state")
            (folder / "shopping_sampler_state.json").write_text(json.dumps({
                "schema_version": 1, "global_step": step, "epoch": epoch,
                "batches_in_epoch": consumed, "loader_batches": 10,
                "sampler": {"skipped_updates": 4, "consecutive_skips": 0},
            }), encoding="utf-8")
            harness = FitHarness([], batches=10)
            harness.config.trainer.resume_mode = "resume_path"
            harness.config.trainer.resume_from_path = str(folder)
            harness.config.trainer.default_local_dir = directory
            harness._shopping_sampler = ShoppingDynamicSampler(harness.config)
            loader = mock.MagicMock()
            loader.__len__.return_value = 10
            harness.train_dataloader = loader
            harness.actor_rollout_wg.load_checkpoint = mock.Mock()
            load = extract_methods("_load_checkpoint", find_latest_ckpt_path=lambda path: None,
                                   Role=SimpleNamespace(Critic="critic"))["_load_checkpoint"]
            with mock.patch.object(torch, "load", return_value={"cursor": consumed}) as load_tensor:
                load(harness)
            self.assertEqual(harness.global_steps, step)
            self.assertEqual(harness._shopping_epoch, expect_epoch)
            self.assertEqual(harness._shopping_batches_in_epoch, expect_consumed)
            self.assertEqual(harness._shopping_sampler.skipped_updates, 4)
            self.assertEqual(load_tensor.called, loads_cursor)
            if loads_cursor:
                loader.load_state_dict.assert_called_once_with({"cursor": consumed})
            else:
                loader.load_state_dict.assert_not_called()

    def test_repository_trainer_hash_matches_preflight_contract(self):
        self.assertEqual(normalized_sha256(SOURCE), EXPECTED_TRAINER_SHA256)
        compile(SOURCE.read_bytes(), str(SOURCE), "exec")


if __name__ == "__main__":
    unittest.main()
