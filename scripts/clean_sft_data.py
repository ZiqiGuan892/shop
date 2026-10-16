#!/usr/bin/env python3
"""Step 2: clean saved Teacher trajectories offline and split SFT data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.sft import build_collection_artifacts, load_collection_exclusions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Step 1 raw.jsonl; never modified.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft-collection/clean"))
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-tasks", type=Path, action="append", default=[],
        help="Additional reserved task JSONL; repeatable. Evaluation and GRPO are always excluded.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if not 0 <= args.validation_ratio < 1:
            raise ValueError("--validation-ratio must be in [0, 1)")
        excluded, exclusion_audit = load_collection_exclusions(args.exclude_tasks)
        manifest = args.raw.parent / "collection_config.json"
        collection_config = {}
        if manifest.exists():
            collection_config = json.loads(manifest.read_text(encoding="utf-8"))["collection_config"]
        print(f"[SFT 清洗] 原始文件={args.raw} 输出目录={args.output_dir}",
              file=sys.stderr, flush=True)
        summary = build_collection_artifacts(
            raw_path=args.raw, output_dir=args.output_dir, held_out_task_ids=excluded,
            validation_ratio=args.validation_ratio, seed=args.seed,
            collection_config=collection_config, exclusion_audit=exclusion_audit,
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"Cleaning stopped: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["accepted"]:
        print("没有通过验收的轨迹；请查看 rejected.jsonl / reject_stats.json。",
              file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
