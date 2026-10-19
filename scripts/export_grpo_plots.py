#!/usr/bin/env python3
"""Export one GRPO run's recorded metrics to PNG/CSV without a web server."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.grpo.plots import (
    PlotExporter, StaticPlotMonitor, check_plot_dependencies,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="one training output directory")
    parser.add_argument("--output", type=Path, help="default: <run-dir>/plots")
    parser.add_argument("--source", choices=("auto", "tensorboard", "diagnostics"), default="auto")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                        help="refresh periodically until Ctrl+C; e.g. --watch 600")
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error("--run-dir must be an existing training output directory")
    if not math.isfinite(args.watch) or args.watch < 0:
        parser.error("--watch must be finite and nonnegative")
    try:
        check_plot_dependencies()
        if args.watch:
            monitor = StaticPlotMonitor(args.run_dir, args.watch, output=args.output, source=args.source)
            monitor.start()
            try:
                while monitor.thread.is_alive():
                    monitor.thread.join(timeout=60)
                if monitor.error:
                    raise RuntimeError(monitor.error)
            except KeyboardInterrupt:
                pass
            finally:
                monitor.stop()
        else:
            exporter = PlotExporter(args.run_dir, output=args.output, source=args.source)
            result = exporter.refresh()
            if result is None:
                raise ValueError("No recorded scalar metrics yet; check the run directory or use --watch 600")
            print(f"Exported {len(result['metrics'])} metric plots through step "
                  f"{result['latest_step']} to {exporter.output}")
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"Static plot export failed: {exc}\n")


if __name__ == "__main__":
    main()
