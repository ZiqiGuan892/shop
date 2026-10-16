#!/usr/bin/env python3
"""Step 1: sample Teacher trajectories only; retain successes and failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# Run with the user's active Python, including simulator-only installations.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.sft import load_collection_exclusions, read_jsonl, write_jsonl
from shopping_grpo.evaluation.rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    _is_infrastructure_failure,
    append_jsonl,
    collect_for_task,
    load_tasks,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks", type=Path, help="JSONL task list; only task_id is used.")
    source.add_argument(
        "--task-range", type=int, nargs=2, metavar=("START", "END"),
        help="Environment task IDs in [START, END), before exclusions and --limit.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft-collection/raw"))
    parser.add_argument(
        "--exclude-tasks", type=Path, action="append", default=[],
        help="Additional task JSONL to exclude; repeatable. Evaluation and GRPO are always excluded.",
    )
    parser.add_argument("--limit", type=int, help="Maximum distinct tasks after exclusions.")
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.add_argument("--base-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default="high")
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--context-compaction", action="store_true")
    parser.add_argument("--observation-token-budget", type=int, default=0)
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    return parser.parse_args(argv)


def _validate_args(args):
    for name in ("attempts_per_task", "workers", "timeout", "max_tokens", "max_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be nonnegative")
    if args.task_range and not 0 <= args.task_range[0] < args.task_range[1]:
        raise ValueError("--task-range requires 0 <= START < END")
    if not args.llm_base_url or not args.api_key:
        raise ValueError("Set OPENAI_BASE_URL and OPENAI_API_KEY (or the corresponding CLI flags)")


def select_tasks(args, excluded_ids):
    candidates = (
        load_tasks(args.tasks) if args.tasks else
        ({"task_id": task_id} for task_id in range(*args.task_range))
    )
    selected = []
    seen = set()
    for row in candidates:
        task_id = int(row["task_id"])
        if task_id < 0:
            raise ValueError("task_id must be nonnegative")
        if task_id in excluded_ids or task_id in seen:
            continue
        seen.add(task_id)
        # Never reuse old messages, answers or prompts from a supplied dataset.
        # The shopping instruction comes from env.reset(task_id).
        selected.append({"task_id": task_id})
        if args.limit is not None and len(selected) >= args.limit:
            break
    if not selected:
        raise ValueError("No tasks remain after excluding evaluation/GRPO/additional reserved tasks")
    return selected


def _make_client(args):
    return OpenAIChatClient(
        model=args.model, base_url=args.llm_base_url, api_key=args.api_key,
        temperature=args.temperature, top_p=args.top_p, timeout=args.timeout,
        max_tokens=args.max_tokens, thinking=args.thinking, reasoning_effort=args.reasoning_effort,
        context_window=args.context_window or None,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget or None,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )


def prepare_output(args, tasks, exclusions):
    """Freeze the task list and sampling recipe, without storing credentials."""
    config_keys = (
        "model", "llm_base_url", "base_url", "attempts_per_task", "temperature", "top_p",
        "timeout", "max_tokens", "max_steps", "thinking", "reasoning_effort", "context_window",
        "context_safety_margin", "context_compaction", "observation_token_budget",
        "observation_detail_token_budget", "observation_generic_token_budget",
        "observation_search_top_k",
    )
    config = {key: getattr(args, key) for key in config_keys}
    config["task_ids"] = [row["task_id"] for row in tasks]
    config["exclusions"] = exclusions
    config["environment_manifest_sha256"] = hashlib.sha256(
        (ROOT / "data/environment.json").read_bytes()
    ).hexdigest()
    output = args.output_dir
    raw = output / "raw.jsonl"
    manifest = output / "collection_config.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("collection_config") != config:
            raise ValueError("Sampling configuration changed; use a new --output-dir to avoid mixing runs")
    elif raw.exists() and raw.stat().st_size:
        raise ValueError("Existing raw.jsonl has no collection_config.json; use a new --output-dir")

    # Fail before appending to malformed or unrelated raw data.
    if raw.exists():
        allowed = set(config["task_ids"])
        for row in read_jsonl(raw):
            if int(row["task_id"]) not in allowed:
                raise ValueError("Existing raw.jsonl contains a task outside this sampling task list")
        if raw.stat().st_size:
            with raw.open("rb") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    raise ValueError("raw.jsonl has an unterminated last line; inspect it before resuming")
    output.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "schema_version": "shopping-teacher-sampling-v1",
            "collection_config": config,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output / "tasks.jsonl", tasks)
    raw.touch(exist_ok=True)
    return raw


def collect_raw_tasks(
    *, tasks, client_factory, output_path, base_url, max_steps=35,
    attempts_per_task=1, workers=1, progress_interval=15,
    stop_requested=None,
):
    """Append every attempt; quality filtering belongs exclusively to step 2."""
    completed = {
        (int(row["task_id"]), int(row.get("attempt_index", 0)))
        for row in read_jsonl(output_path)
    }
    attempts = [(task, attempt) for task in tasks for attempt in range(attempts_per_task)]
    remaining = [(task, attempt) for task, attempt in attempts
                 if (task["task_id"], attempt) not in completed]
    existing = len(attempts) - len(remaining)
    candidates = iter(remaining)
    pending = {}
    written = 0
    started = time.monotonic()
    interrupted = infrastructure_failed = False
    stop_requested = stop_requested if stop_requested is not None else threading.Event()
    print(f"[教师采样] 总尝试={len(attempts)} 已有={existing} 待运行={len(remaining)} "
          f"并发={workers} 原始文件={output_path}", file=sys.stderr, flush=True)

    def report():
        elapsed = time.monotonic() - started
        eta = f"{elapsed / written * (len(remaining) - written):.0f}s" if written else "待首条完成"
        active = ",".join(f"{task['task_id']}:{attempt}" for task, attempt in pending.values())
        print(f"[教师采样] 已记录={existing + written}/{len(attempts)} "
              f"已用={elapsed:.0f}s ETA≈{eta} 运行中(task:attempt)={active or '-'}",
              file=sys.stderr, flush=True)

    def submit_available(executor):
        while len(pending) < workers and not stop_requested.is_set():
            try:
                task, attempt = next(candidates)
            except StopIteration:
                break
            future = executor.submit(
                collect_for_task, task, client=client_factory(), base_url=base_url,
                max_steps=max_steps, attempt_index=attempt,
            )
            pending[future] = (task, attempt)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_available(executor)
        while pending:
            try:
                ready, _ = wait(pending, timeout=progress_interval or None, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                interrupted = True
                print("[教师采样] 停止派发新任务；等待运行中的请求结束、保存轨迹并释放环境。",
                      file=sys.stderr, flush=True)
                continue
            for future in ready:
                task, attempt = pending.pop(future)
                trajectory = future.result()
                append_jsonl(output_path, [trajectory])
                written += 1
                infrastructure_failed |= (
                    _is_infrastructure_failure(trajectory)
                    or (trajectory.get("error") or {}).get("type") == "HTTPError"
                )
                print(f"[教师采样] 写入 task_id={task['task_id']} attempt={attempt} "
                      f"status={trajectory.get('status')} steps={len(trajectory.get('steps') or [])}",
                      file=sys.stderr, flush=True)
            interrupted |= stop_requested.is_set()
            if not interrupted and not infrastructure_failed:
                submit_available(executor)
            report()
    if infrastructure_failed:
        raise CollectionInfrastructureError("Service failure: raw attempts saved; no further tasks scheduled")
    if interrupted:
        raise KeyboardInterrupt
    return written


def main(argv=None):
    args = parse_args(argv)
    previous_handlers = {}
    try:
        _validate_args(args)
        excluded, exclusion_audit = load_collection_exclusions(args.exclude_tasks)
        tasks = select_tasks(args, excluded)
        _make_client(args)  # Validate model options before creating files; makes no request.
        raw = prepare_output(args, tasks, exclusion_audit)
        stop_requested = threading.Event()

        def request_stop(signum, frame):
            stop_requested.set()
            print("[教师采样] 停止派发新任务；正在保存运行中的任务并释放环境，请等待。",
                  file=sys.stderr, flush=True)

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        written = collect_raw_tasks(
            tasks=tasks, client_factory=lambda: _make_client(args), output_path=raw,
            base_url=args.base_url, max_steps=args.max_steps,
            attempts_per_task=args.attempts_per_task, workers=args.workers,
            progress_interval=args.progress_interval,
            stop_requested=stop_requested,
        )
    except KeyboardInterrupt:
        return 130
    except (ValueError, KeyError, OSError, CollectionInfrastructureError) as exc:
        print(f"Sampling stopped: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({"collected_raw": written, "raw": str(raw)}, ensure_ascii=False), flush=True)
    print("下一步：运行 scripts/clean_sft_data.py --raw <上面的原始文件路径>",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
