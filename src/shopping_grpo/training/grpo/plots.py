"""CPU-only static plots of recorded GRPO metrics; no model or web server."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def check_plot_dependencies():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Static plots require Matplotlib in the training Python: '
            'python -m pip install "matplotlib>=3.8,<4" '
            '(or disable automatic export with --plot-interval 0)'
        ) from exc


class MetricReader:
    """Read one run incrementally, preserving repeated steps and nonfinite values."""

    def __init__(self, run_dir, source="auto"):
        self.run_dir = Path(run_dir)
        self.source = source
        self.active_source = None
        self.accumulator = None
        self.offset = 0
        self.jsonl_series = defaultdict(list)

    def _tensorboard(self):
        directory = self.run_dir / "tensorboard"
        if not any(directory.glob("events.out.tfevents.*")):
            return {}
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        if self.accumulator is None:
            self.accumulator = EventAccumulator(
                str(directory), size_guidance={"scalars": 0, "tensors": 0},
            )
        self.accumulator.Reload()
        series = {
            tag: [(event.step, event.value, event.wall_time) for event in self.accumulator.Scalars(tag)]
            for tag in self.accumulator.Tags()["scalars"]
        }
        # PyTorch's default add_scalar uses simple_value; also accept its new_style format.
        from tensorboard.util.tensor_util import make_ndarray

        for tag in self.accumulator.Tags()["tensors"]:
            if tag in series:
                continue
            points = []
            for event in self.accumulator.Tensors(tag):
                value = make_ndarray(event.tensor_proto)
                if value.size == 1 and value.dtype.kind in "biuf":
                    points.append((event.step, float(value.item()), event.wall_time))
            if points:
                series[tag] = points
        return series

    def _diagnostics(self):
        path = self.run_dir / "training_diagnostics.jsonl"
        if not path.is_file():
            return {}
        if path.stat().st_size < self.offset:
            raise ValueError("training_diagnostics.jsonl was truncated; restart the exporter")
        with path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    break  # A live writer may not have finished the last record yet.
                record = json.loads(line)
                if record.get("event") in ("optimizer_step", "skipped_update"):
                    step = record["global_step"]
                    if not isinstance(step, int) or isinstance(step, bool):
                        raise ValueError("diagnostic global_step must be an integer")
                    for tag, value in record.get("metrics", {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            self.jsonl_series[tag].append((step, float(value), None))
                self.offset = handle.tell()
        return dict(self.jsonl_series)

    def read(self):
        if self.source == "tensorboard":
            self.active_source = "tensorboard"
            return self._tensorboard()
        diagnostics = self._diagnostics()
        if self.source == "diagnostics":
            self.active_source = "diagnostics"
            return diagnostics
        tensorboard = self._tensorboard()
        if not diagnostics or not tensorboard:
            self.active_source = "diagnostics" if diagnostics else "tensorboard"
            return diagnostics or tensorboard
        # JSONL is flushed on each event; TB may still buffer the same records.
        # Keep every JSONL event at a tag/step, and use TB for uncovered points
        # (notably initial validation). Never average or deduplicate skipped steps.
        series = dict(tensorboard)
        for tag, points in diagnostics.items():
            steps = {point[0] for point in points}
            additional = [point for point in tensorboard.get(tag, []) if point[0] not in steps]
            series[tag] = sorted([*additional, *points], key=lambda point: point[0])
        self.active_source = "tensorboard+diagnostics"
        return series


def _atomic_write(path, writer):
    """Replace only a completed artifact, so previews never see a partial PNG."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".plot-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _draw_metric(axis, tag, points):
    from matplotlib.ticker import MaxNLocator

    # Sort by optimizer step while preserving arrival order for repeated steps.
    points = sorted(points, key=lambda point: point[0])
    values = [value if math.isfinite(value) else math.nan for _, value, _ in points]
    axis.plot([step for step, _, _ in points], values, color="#2563eb", linewidth=1.4,
              marker=".", markersize=3)
    axis.set_title(tag, fontsize=10, wrap=True)
    axis.set_xlabel("Optimizer step")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, alpha=0.2)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    nonfinite = sum(not math.isfinite(value) for _, value, _ in points)
    if nonfinite:
        axis.text(0.02, 0.96, f"NaN/Inf: {nonfinite} (gaps; see CSV)",
                  transform=axis.transAxes, va="top", color="#b91c1c", fontsize=9)


def _filename(tag):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", tag).strip("_")[:90] or "metric"
    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:10]
    return f"{slug}--{digest}.png"


def render_plots(series, output, *, source, run_name):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for tag, points in sorted(series.items()):
        figure = Figure(figsize=(8, 4.5), layout="constrained")
        FigureCanvasAgg(figure)
        _draw_metric(figure.subplots(), tag, points)
        figure.suptitle(run_name, fontsize=10, color="#64748b")
        filename = _filename(tag)
        _atomic_write(output / filename, lambda path: figure.savefig(path, format="png", dpi=150))
        figure.clear()
        metadata[tag] = {
            "file": filename, "points": len(points),
            "first_step": min(point[0] for point in points),
            "last_step": max(point[0] for point in points),
            "nonfinite": sum(not math.isfinite(point[1]) for point in points),
        }

    overview = (
        "reward/terminal_utility_mean", "reward/strict_mean", "reward/purchase_success_rate",
        "actor/pg_loss", "actor/grad_norm", "shopping_dynamic_sampling/skipped_updates_total",
    )
    figure = Figure(figsize=(15, 9), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, tag in zip(figure.subplots(2, 3).flat, overview):
        if tag in series:
            _draw_metric(axis, tag, series[tag])
        else:
            axis.set_title(tag, fontsize=10)
            axis.text(0.5, 0.5, "Not recorded", transform=axis.transAxes, ha="center")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"{run_name} | Recorded GRPO metrics (no smoothing)", fontsize=14)
    _atomic_write(output / "summary.png", lambda path: figure.savefig(path, format="png", dpi=150))
    figure.clear()

    def write_csv(path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("tag", "step", "value", "wall_time"))
            for tag, points in sorted(series.items()):
                writer.writerows((tag, *point) for point in points)

    _atomic_write(output / "metrics.csv", write_csv)
    manifest = {
        "source": source, "run": run_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_step": max(item["last_step"] for item in metadata.values()),
        "metrics": metadata,
    }
    _atomic_write(output / "index.json", lambda path: path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    ))
    return manifest


class PlotExporter:
    def __init__(self, run_dir, *, output=None, source="auto"):
        self.run_dir = Path(run_dir)
        self.output = Path(output) if output is not None else self.run_dir / "plots"
        self.reader = MetricReader(run_dir, source)
        self.fingerprint = None

    def refresh(self):
        series = {tag: points for tag, points in self.reader.read().items() if points}
        if not series:
            return None
        fingerprint = hashlib.sha256(json.dumps(
            [self.reader.active_source, series], sort_keys=True,
        ).encode("utf-8")).digest()
        if fingerprint == self.fingerprint:
            return None
        manifest = render_plots(series, self.output, source=self.reader.active_source,
                                run_name=self.run_dir.name)
        self.fingerprint = fingerprint
        return manifest


class StaticPlotMonitor:
    """One CPU thread owned by the launcher; failures never mask training errors."""

    def __init__(self, run_dir, interval=600, *, output=None, source="auto"):
        self.exporter = PlotExporter(run_dir, output=output, source=source)
        self.interval = interval
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="grpo-static-plots", daemon=True)
        self.error = None
        self.log_path = (Path(output) / "export.log") if output else (Path(run_dir) / "plots.log")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            print("Static plot export is still finishing; rerun export_grpo_plots.py if needed.", flush=True)

    def _refresh(self, log):
        try:
            result = self.exporter.refresh()
            if result:
                message = (
                    f"Static plots updated: {self.exporter.output} "
                    f"(step={result['latest_step']}, metrics={len(result['metrics'])}, "
                    f"source={result['source']})"
                )
                print(message, flush=True)
                log.write(message + "\n")
        except Exception as exc:
            message = f"Static plot export failed (training continues): {type(exc).__name__}: {exc}"
            print(message, flush=True)
            log.write(message + "\n")

    def _run(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"Static plot refresh interval: {self.interval}s\n")
                self._refresh(log)
                while not self.stopped.wait(self.interval):
                    self._refresh(log)
                self._refresh(log)  # Last readable records on success, failure, or Ctrl+C.
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"Static plot monitor stopped (training continues): {self.error}", flush=True)
