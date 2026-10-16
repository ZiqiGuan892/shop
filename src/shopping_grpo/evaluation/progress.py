"""Console progress and unmodified rollout excerpts for sequential evaluation."""

import json
import sys
import threading
import time


def _duration(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class EvaluationProgress:
    """Observe collection without modifying trajectories or making API requests."""

    _trace_lists = (
        "messages", "steps", "blocked_tool_calls", "tool_call_truncations",
        "context_compactions", "context_turn_tokens",
    )

    def __init__(self, interval=15, rollout_print_every=5, stream=None, clock=None):
        if interval < 0 or rollout_print_every < 0:
            raise ValueError("progress interval and rollout print interval must be nonnegative")
        self.interval = interval
        self.rollout_print_every = rollout_print_every
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.started = None
        self.total = self.existing = self.finished = 0
        self.finished_seconds = 0.0
        self.current = None
        self.output_available = True

    def __enter__(self):
        if self.interval:
            self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            if self.started is not None:
                label = "采集完成" if exc_type is None else f"已中断 ({exc_type.__name__})"
                self._write(f"[评测] {label} | {self._totals()}")

    def __call__(self, event, **data):
        with self.lock:
            now = self.clock()
            if event == "start":
                self.started = now
                self.total = data["total"]
                self.existing = data["existing"]
                self._write(
                    f"[评测] 总任务尝试={self.total} | 已有记录={self.existing} | "
                    f"待运行={self.total - self.existing} | "
                    f"轨迹打印间隔={self.rollout_print_every} 环境步（0=关闭）"
                )
            elif event == "task_start":
                self.current = dict(data, started=now, phase="环境重置", phase_started=now, steps=0)
                self.cursors = dict.fromkeys(self._trace_lists, 0)
                self.trace_started = False
                self._write(f"[评测] 开始 task_id={data['task_id']} | {self._totals()}")
            elif event == "phase":
                self.current.update(data, phase_started=now)
            elif event == "step":
                trajectory = data["trajectory"]
                self.current["steps"] = len(trajectory["steps"])
                if (
                    self.rollout_print_every
                    and len(trajectory["steps"]) - self.cursors["steps"] >= self.rollout_print_every
                ):
                    self._trace(trajectory, final=False)
            elif event == "task_end":
                trajectory = data["trajectory"]
                elapsed = now - self.current["started"]
                self.finished += 1
                self.finished_seconds += elapsed
                if self.rollout_print_every:
                    self._trace(trajectory, final=True)
                self._write(
                    f"[评测] 已写入 task_id={trajectory['task_id']} | "
                    f"status={trajectory['status']} | steps={len(trajectory['steps'])} | "
                    f"reward={trajectory['final_reward']} | 本条耗时={_duration(elapsed)} | "
                    f"{self._totals()}"
                )
                self.current = None

    def _totals(self):
        completed = self.existing + self.finished
        percent = 100 * completed / self.total if self.total else 100.0
        remaining = self.total - completed
        if remaining == 0:
            eta = "00:00:00"
        elif self.finished:
            eta = _duration(self.finished_seconds / self.finished * remaining)
        else:
            eta = "待本次首条任务完成"
        elapsed = _duration(self.clock() - self.started)
        return (
            f"已记录={completed}/{self.total} ({percent:.1f}%) | "
            f"本次完成={self.finished} | 已用={elapsed} | ETA≈{eta}"
        )

    def heartbeat(self):
        """Also usable synchronously in tests, without sleeping or making requests."""
        with self.lock:
            if self.current is None:
                return
            now = self.clock()
            self._write(
                f"[评测] 进行中 task_id={self.current['task_id']} | "
                f"环境步={self.current['steps']}/{self.current['max_steps']} | "
                f"阶段={self.current['phase']} | "
                f"阶段耗时={_duration(now - self.current['phase_started'])} | "
                f"{self._totals()}"
            )

    def _heartbeat_loop(self):
        while not self.stop.wait(self.interval):
            self.heartbeat()

    def _trace(self, trajectory, final):
        # Serialize only newly collected entries. No reconstruction, truncation,
        # extra model calls, or observer-specific fields in the saved trajectory.
        payload = {
            "trajectory_id": trajectory["trajectory_id"],
            "task_id": trajectory["task_id"],
            "attempt_index": trajectory["attempt_index"],
            "final": final,
            "recorded_steps": len(trajectory["steps"]),
            "offsets": dict(self.cursors),
        }
        for name in self._trace_lists:
            payload[name] = trajectory[name][self.cursors[name]:]
            self.cursors[name] = len(trajectory[name])
        if not self.trace_started:
            payload["initial_result"] = trajectory["initial_result"]
            self.trace_started = True
        if final:
            for name in ("status", "done", "final_reward", "terminal_result", "error", "release_error"):
                payload[name] = trajectory[name]
        self._write("[rollout] " + json.dumps(payload, ensure_ascii=False, indent=2) + "\n[/rollout]")

    def _write(self, text):
        if not self.output_available:
            return
        try:
            print(text, file=self.stream, flush=True)
        except (OSError, UnicodeError):
            # A closed pipe or terminal must not turn a shopping task into an error.
            self.output_available = False
