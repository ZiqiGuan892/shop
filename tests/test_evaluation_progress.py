"""Verify console traces against collected records, using no model or environment service."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest
from urllib.error import URLError
from unittest.mock import patch

from scripts import evaluate_shop_benchmark
from shopping_grpo.evaluation.progress import EvaluationProgress
from shopping_grpo.evaluation.rollout import CollectionInfrastructureError, collect_tasks


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class ScriptedShop:
    def __init__(self, terminal_step=7, **kwargs):
        self.terminal_step = terminal_step
        self.steps = 0
        self.released = False

    def reset(self, task_id):
        return {"env_idx": 0, "instruction": f"fixture user request {task_id}"}

    def step(self, action):
        self.steps += 1
        return {
            "instruction": f"raw environment feedback {self.steps}\nsecond line",
            "reward": self.steps / 10,
            "done": self.steps == self.terminal_step,
        }

    def release(self):
        self.released = True


class ScriptedModel:
    def __init__(self, clock, progress=None, response="tool"):
        self.clock = clock
        self.progress = progress
        self.response = response
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        self.clock.now += 10
        if self.progress is not None:
            self.progress.heartbeat()
        if self.response == "error":
            raise URLError("fixture connection unavailable")
        if self.response == "final":
            return {"role": "assistant", "content": "fixture final response"}
        name = "open_product" if self.response == "blocked" else "search_products"
        args = {"asin": "not-visible"} if name == "open_product" else {"query": f"query {self.calls}"}
        return {
            "role": "assistant", "content": f"fixture model reply {self.calls}",
            "tool_calls": [{
                "id": f"call_{self.calls}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        }

    def project_observation(self, tool_name, observation, parameters=None):
        return "visible " + observation, {"truncated": False}


def snapshots(text):
    return [json.loads(block) for block in re.findall(r"\[rollout\] (.*?)\n\[/rollout\]", text, re.S)]


class EvaluationProgressTest(unittest.TestCase):
    def run_collection(self, every=3, response="tool", terminal_step=7, initial=(), enabled=True):
        clock = FakeClock()
        stream = io.StringIO()
        progress = EvaluationProgress(interval=0, rollout_print_every=every, stream=stream, clock=clock)
        client = ScriptedModel(clock, progress if enabled else None, response=response)
        environments = []

        def factory(**kwargs):
            env = ScriptedShop(terminal_step=terminal_step, **kwargs)
            environments.append(env)
            return env

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "trajectories.jsonl"
            output.write_text("".join(json.dumps(row) + "\n" for row in initial), encoding="utf-8")
            with progress:
                written = collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}, {"task_id": 3}], client=client,
                    output_path=output, base_url="http://fixture.invalid", env_factory=factory,
                    max_steps=35, progress=progress if enabled else None,
                )
            saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        return written, saved, stream.getvalue(), client, environments

    def test_interval_prints_actual_new_records_and_final_partial_block(self):
        written, saved, text, client, environments = self.run_collection()
        self.assertEqual(saved, written)
        self.assertEqual(client.calls, 21)
        self.assertTrue(all(env.released for env in environments))
        chunks = snapshots(text)
        for trajectory in written:
            parts = [chunk for chunk in chunks if chunk["task_id"] == trajectory["task_id"]]
            self.assertEqual([part["recorded_steps"] for part in parts], [3, 6, 7])
            self.assertEqual([part["offsets"]["steps"] for part in parts], [0, 3, 6])
            for field in EvaluationProgress._trace_lists:
                self.assertEqual([entry for part in parts for entry in part[field]], trajectory[field])
            self.assertEqual(parts[0]["initial_result"], trajectory["initial_result"])
            self.assertEqual(parts[-1]["terminal_result"], trajectory["terminal_result"])
            self.assertEqual(parts[-1]["final_reward"], trajectory["final_reward"])
            self.assertEqual(parts[-1]["status"], trajectory["status"])
        self.assertIn("raw environment feedback 1", text)
        self.assertIn("visible raw environment feedback 1", text)
        self.assertIn("阶段=模型请求", text)
        self.assertIn("环境步=0/35", text)

    def test_reporting_does_not_change_trajectory_schema_or_behavior(self):
        observed, _, _, observed_client, _ = self.run_collection()
        plain, _, _, plain_client, _ = self.run_collection(enabled=False)
        for left, right in zip(observed, plain):
            for volatile in ("trajectory_id", "created_at"):
                left.pop(volatile)
                right.pop(volatile)
            self.assertEqual(left, right)
        self.assertEqual(observed_client.calls, plain_client.calls)

    def test_resume_counts_matching_records_and_estimates_only_this_run(self):
        _, _, text, client, _ = self.run_collection(
            terminal_step=1,
            initial=({"task_id": 1}, {"task_id": 999}, {"task_id": 1, "attempt_index": 7}),
        )
        self.assertEqual(client.calls, 2)
        self.assertIn("已有记录=1 | 待运行=2", text)
        self.assertIn("ETA≈待本次首条任务完成", text)
        self.assertIn("已记录=2/3 (66.7%) | 本次完成=1 | 已用=00:00:10 | ETA≈00:00:10", text)
        self.assertIn("已记录=3/3 (100.0%) | 本次完成=2 | 已用=00:00:20 | ETA≈00:00:00", text)

    def test_terminal_step_on_interval_does_not_repeat_messages(self):
        written, _, text, _, _ = self.run_collection(terminal_step=3)
        parts = [chunk for chunk in snapshots(text) if chunk["task_id"] == 1]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1]["steps"], [])
        self.assertEqual(parts[1]["messages"], [])
        self.assertTrue(parts[1]["final"])
        self.assertEqual(parts[1]["status"], written[0]["status"])

    def test_zero_step_final_response_and_blocked_calls_are_printed(self):
        for response, expected_status in (("final", "assistant_final"), ("blocked", "invalid_action_limit")):
            with self.subTest(response=response):
                written, _, text, _, _ = self.run_collection(response=response)
                chunks = snapshots(text)
                self.assertEqual(len(chunks), 3)
                self.assertEqual(chunks[0]["steps"], [])
                self.assertEqual(chunks[0]["status"], expected_status)
                self.assertEqual(chunks[0]["messages"], written[0]["messages"])
                self.assertEqual(chunks[0]["blocked_tool_calls"], written[0]["blocked_tool_calls"])

    def test_zero_interval_disables_trace_but_keeps_task_progress(self):
        _, _, text, _, _ = self.run_collection(every=0, terminal_step=1)
        self.assertEqual(snapshots(text), [])
        self.assertIn("已写入 task_id=1", text)
        self.assertIn("采集完成", text)

    def test_infrastructure_failure_prints_error_and_still_stops_collection(self):
        clock = FakeClock()
        stream = io.StringIO()
        progress = EvaluationProgress(interval=0, stream=stream, clock=clock)
        client = ScriptedModel(clock, response="error")
        env = ScriptedShop()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "trajectories.jsonl"
            with self.assertRaises(CollectionInfrastructureError), progress:
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}], client, output, "http://fixture.invalid",
                    env_factory=lambda **kwargs: env, progress=progress,
                )
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
        self.assertTrue(env.released)
        self.assertEqual(client.calls, 1)
        self.assertEqual(snapshots(stream.getvalue())[0]["error"]["type"], "URLError")
        self.assertIn("已中断 (CollectionInfrastructureError)", stream.getvalue())
        self.assertNotIn("采集完成", stream.getvalue())

    def test_heartbeat_thread_runs_while_waiting_and_exits_on_close(self):
        observed = threading.Event()

        class Stream(io.StringIO):
            def write(self, text):
                result = super().write(text)
                if "进行中" in text:
                    observed.set()
                return result

        with EvaluationProgress(interval=0.01, stream=Stream()) as progress:
            progress("start", total=1, existing=0)
            progress("task_start", task_id=1, attempt_index=0, max_steps=35)
            self.assertTrue(observed.wait(2))
        self.assertFalse(progress.thread.is_alive())

    def test_closed_console_does_not_change_evaluation_result(self):
        class ClosedStream:
            def write(self, text):
                raise BrokenPipeError("fixture closed pipe")

        clock = FakeClock()
        with tempfile.TemporaryDirectory() as temporary:
            with EvaluationProgress(interval=0, stream=ClosedStream(), clock=clock) as progress:
                written = collect_tasks(
                    [{"task_id": 1}], ScriptedModel(clock, response="final"),
                    Path(temporary) / "trajectories.jsonl", "http://fixture.invalid",
                    env_factory=ScriptedShop, progress=progress,
                )
        self.assertEqual(written[0]["status"], "assistant_final")
        self.assertIsNone(written[0]["error"])

    def test_cli_preserves_json_summary_stdout_and_prints_rollout_to_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        client = ScriptedModel(FakeClock())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, output, summary = root / "tasks.jsonl", root / "raw.jsonl", root / "summary.json"
            tasks.write_text('{"task_id": 1}\n', encoding="utf-8")
            argv = [
                "evaluate_shop_benchmark.py", "--benchmark", str(tasks), "--output", str(output),
                "--summary", str(summary), "--model", "fixture-model", "--api-key", "fixture-key",
                "--llm-base-url", "http://fixture.invalid/v1", "--progress-interval", "0",
                "--rollout-print-every", "1",
            ]

            def collect(*args, **kwargs):
                return collect_tasks(
                    *args, **kwargs, env_factory=lambda **env: ScriptedShop(terminal_step=1, **env),
                )

            with (
                patch.object(sys, "argv", argv),
                patch.object(evaluate_shop_benchmark, "OpenAIChatClient", return_value=client),
                patch.object(evaluate_shop_benchmark, "collect_tasks", side_effect=collect),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                evaluate_shop_benchmark.main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result, json.loads(summary.read_text(encoding="utf-8")))
            self.assertEqual(result["protocol"]["max_steps"], 35)
            self.assertEqual(result["protocol"]["model"], "fixture-model")
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(snapshots(stderr.getvalue())[0]["steps"], row["steps"])
        self.assertNotIn("fixture-key", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
