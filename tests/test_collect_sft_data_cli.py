"""Offline tests for the two-step Teacher sampling/cleaning workflow."""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import clean_sft_data, collect_sft_data
from shopping_grpo.collection.sft import load_collection_exclusions, read_jsonl, write_jsonl
from shopping_grpo.evaluation.rollout import CollectionInfrastructureError
from tests.test_sft_collection import _accepted_trajectory


ROOT = Path(__file__).resolve().parents[1]


class TeacherDataWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="shopping-teacher-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "raw"
        self.tasks = self.root / "tasks.jsonl"
        write_jsonl(self.tasks, [{"task_id": x} for x in (1, 1, 2, 3, 4)])
        self.args = [
            "--tasks", str(self.tasks), "--output-dir", str(self.output),
            "--llm-base-url", "http://teacher.test/v1", "--api-key", "test-secret",
            "--model", "teacher-test", "--attempts-per-task", "2", "--workers", "2",
        ]
        self.addCleanup(patch.stopall)
        patch("scripts.collect_sft_data.load_collection_exclusions", return_value=({4}, [])).start()
        patch("scripts.clean_sft_data.load_collection_exclusions", return_value=({4}, [])).start()
        self.client = patch("scripts.collect_sft_data._make_client", return_value=object()).start()
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))

    @staticmethod
    def rollout(task, *, attempt_index, **kwargs):
        row = _accepted_trajectory(task["task_id"])
        row["attempt_index"] = attempt_index
        row["trajectory_id"] += f"-{attempt_index}"
        if task["task_id"] == 2:
            row["terminal_result"]["reward_detail"]["reward_valid"] = False
        return row

    def test_sampling_keeps_all_attempts_and_cleaning_is_offline_and_reproducible(self):
        with patch("scripts.collect_sft_data.collect_for_task", side_effect=self.rollout) as rollout:
            self.assertEqual(collect_sft_data.main(self.args), 0)
            self.assertEqual(rollout.call_count, 6)
        self.assertEqual(
            {p.name for p in self.output.iterdir()},
            {"raw.jsonl", "tasks.jsonl", "collection_config.json"},
        )
        raw = self.output / "raw.jsonl"
        before = raw.read_bytes()
        self.assertEqual(len(list(read_jsonl(raw))), 6)
        self.assertIn(b"Teacher private reasoning", before)
        self.assertIn(b"Reward: 1.0", before)
        config = (self.output / "collection_config.json").read_text(encoding="utf-8")
        self.assertNotIn("test-secret", config)

        clean = self.root / "clean"
        clean_args = ["--raw", str(raw), "--output-dir", str(clean), "--validation-ratio", "0.5"]
        self.client.reset_mock()
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network is forbidden")):
            self.assertEqual(clean_sft_data.main(clean_args), 0)
            first_outputs = {p.name: p.read_bytes() for p in clean.iterdir()}
            self.assertEqual(clean_sft_data.main(clean_args), 0)
        self.client.assert_not_called()
        self.assertEqual(first_outputs, {p.name: p.read_bytes() for p in clean.iterdir()})
        self.assertEqual(raw.read_bytes(), before)
        stats = json.loads((clean / "reject_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["accepted"], 2)
        self.assertEqual(stats["rejected"], 2)
        self.assertEqual(stats["duplicate_tasks_excluded"], 2)
        train = {r["task_id"] for r in read_jsonl(clean / "train.jsonl")}
        validation = {r["task_id"] for r in read_jsonl(clean / "validation.jsonl")}
        self.assertEqual(train | validation, {1, 3})
        self.assertFalse(train & validation)
        payload = (clean / "sft.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("Teacher private reasoning", payload)
        self.assertNotIn("Reward: 1.0", payload)

    def test_resume_makes_no_repeat_requests_even_for_failed_quality_attempts(self):
        with patch("scripts.collect_sft_data.collect_for_task", side_effect=self.rollout):
            self.assertEqual(collect_sft_data.main(self.args), 0)
        raw = self.output / "raw.jsonl"
        before = raw.read_bytes()
        with patch("scripts.collect_sft_data.collect_for_task") as rollout:
            self.assertEqual(collect_sft_data.main(self.args), 0)
        rollout.assert_not_called()
        self.assertEqual(raw.read_bytes(), before)

    def test_changed_teacher_cannot_silently_mix_into_raw(self):
        with patch("scripts.collect_sft_data.collect_for_task", side_effect=self.rollout):
            self.assertEqual(collect_sft_data.main(self.args), 0)
        with patch("scripts.collect_sft_data.collect_for_task") as rollout:
            self.assertEqual(collect_sft_data.main(self.args + ["--model", "another-model"]), 2)
        rollout.assert_not_called()
        self.assertIn("configuration changed", self.stderr.getvalue())

    def test_task_sources_only_contribute_ids_and_limit_applies_after_exclusion(self):
        write_jsonl(self.tasks, [
            {"task_id": 4}, {"task_id": 1, "prompt": [{"role": "user", "content": "old"}]},
            {"task_id": 1}, {"task_id": 2},
        ])
        args = collect_sft_data.parse_args(self.args + ["--limit", "2"])
        self.assertEqual(collect_sft_data.select_tasks(args, {4}), [{"task_id": 1}, {"task_id": 2}])
        args = collect_sft_data.parse_args(["--task-range", "0", "5", "--limit", "2"])
        self.assertEqual(collect_sft_data.select_tasks(args, {0, 1}), [{"task_id": 2}, {"task_id": 3}])

    def test_cleaner_rechecks_reserved_tasks_even_if_present_in_raw(self):
        raw = self.root / "raw.jsonl"
        write_jsonl(raw, [_accepted_trajectory(1), _accepted_trajectory(4)])
        clean = self.root / "clean"
        self.assertEqual(clean_sft_data.main(["--raw", str(raw), "--output-dir", str(clean)]), 0)
        self.assertEqual([r["task_id"] for r in read_jsonl(clean / "sft.jsonl")], [1])
        stats = json.loads((clean / "reject_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["held_out_excluded"], 1)

    def test_cleaner_will_not_overwrite_raw_or_accept_invalid_reward(self):
        raw = self.root / "sft.jsonl"
        row = _accepted_trajectory(1)
        row["terminal_result"]["reward_detail"]["reward_valid"] = False
        write_jsonl(raw, [row])
        before = raw.read_bytes()
        self.assertEqual(clean_sft_data.main(["--raw", str(raw), "--output-dir", str(self.root)]), 2)
        self.assertEqual(raw.read_bytes(), before)
        self.assertEqual(clean_sft_data.main([
            "--raw", str(raw), "--output-dir", str(self.root / "clean"),
        ]), 2)
        self.assertEqual(list(read_jsonl(self.root / "clean/train.jsonl")), [])

    def test_infrastructure_failure_stops_dispatch_and_saves_inflight_results(self):
        raw = self.root / "raw.jsonl"
        raw.touch()
        gate = threading.Barrier(2)
        clients = []

        def client_factory():
            client = object()
            clients.append(client)
            return client

        def rollout(task, **kwargs):
            gate.wait(timeout=5)
            row = self.rollout(task, **kwargs)
            row["error"] = {"type": "HTTPError", "message": "401"}
            return row

        with patch("scripts.collect_sft_data.collect_for_task", side_effect=rollout) as collect:
            with self.assertRaises(CollectionInfrastructureError):
                collect_sft_data.collect_raw_tasks(
                    tasks=[{"task_id": x} for x in (1, 2, 3)], client_factory=client_factory,
                    output_path=raw, base_url="http://shop.test", workers=2,
                )
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(len({id(c) for c in clients}), 2)
        self.assertEqual(len(list(read_jsonl(raw))), 2)

    def test_stop_drains_active_tasks_without_dispatching_next(self):
        raw = self.root / "raw.jsonl"
        raw.touch()
        stop = threading.Event()

        def rollout(task, **kwargs):
            stop.set()
            return self.rollout(task, **kwargs)

        with patch("scripts.collect_sft_data.collect_for_task", side_effect=rollout) as collect:
            with self.assertRaises(KeyboardInterrupt):
                collect_sft_data.collect_raw_tasks(
                    tasks=[{"task_id": 1}, {"task_id": 2}], client_factory=object,
                    output_path=raw, base_url="http://shop.test", stop_requested=stop,
                )
        self.assertEqual(collect.call_count, 1)
        self.assertEqual(len(list(read_jsonl(raw))), 1)

    def test_corrupt_raw_is_not_silently_skipped_on_resume(self):
        args = collect_sft_data.parse_args(self.args)
        raw = collect_sft_data.prepare_output(args, [{"task_id": 1}], [])
        raw.write_text('{"task_id":', encoding="utf-8")
        before = raw.read_bytes()
        with self.assertRaises(ValueError):
            collect_sft_data.prepare_output(args, [{"task_id": 1}], [])
        self.assertEqual(raw.read_bytes(), before)


class TeacherDataEntrypointTests(unittest.TestCase):
    def test_both_help_commands_work_without_installed_project_or_api_key(self):
        env = {k: v for k, v in os.environ.items() if k not in {
            "PYTHONPATH", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
        }}
        for name in ("collect_sft_data.py", "clean_sft_data.py"):
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts" / name), "--help"],
                cwd=ROOT.parent, env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_required_evaluation_and_grpo_exclusions_cannot_be_disabled(self):
        excluded, audit = load_collection_exclusions()
        for relative in ("data/evaluation/tasks.jsonl", "data/grpo/train.jsonl", "data/grpo/validation.jsonl"):
            self.assertTrue({row["task_id"] for row in read_jsonl(ROOT / relative)} <= excluded)
        self.assertEqual(len(audit), 3)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_collection_exclusions([Path(tmp) / "missing.jsonl"])
        with patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                load_collection_exclusions()


if __name__ == "__main__":
    unittest.main()
