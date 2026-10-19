"""Read recorded scalars and render CPU-only artifacts; never launch training."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from shopping_grpo.training.grpo import plots


ROOT = Path(__file__).resolve().parents[1]


class GRPOPlotsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shopping-grpo-plots-")
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "training run"
        self.run.mkdir()
        self.diagnostics = self.run / "training_diagnostics.jsonl"

    def write_records(self, *records):
        with self.diagnostics.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def assert_png(self, path):
        data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertGreaterEqual(width, 800)
        self.assertGreaterEqual(height, 450)

    def test_tensorboard_reads_simple_and_tensor_scalars_without_vectors(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.compat.proto.types_pb2 import DT_FLOAT
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        writer = EventFileWriter(str(self.run / "tensorboard"))
        try:
            writer.add_event(Event(wall_time=100, step=50, summary=Summary(value=[
                Summary.Value(tag="actor/pg_loss", simple_value=-0.125),
                Summary.Value(
                    tag="reward/strict_mean",
                    tensor=TensorProto(dtype=DT_FLOAT, tensor_content=struct.pack("<f", 0.75)),
                    metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="scalars")),
                ),
                Summary.Value(
                    tag="not_a_scalar",
                    tensor=TensorProto(dtype=DT_FLOAT,
                                       tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2)]),
                                       tensor_content=struct.pack("<ff", 0.25, 0.5)),
                ),
            ])))
            writer.flush()
            reader = plots.MetricReader(self.run)
            series = reader.read()
            self.assertEqual(reader.active_source, "tensorboard")
            self.assertEqual(series["actor/pg_loss"], [(50, -0.125, 100)])
            self.assertEqual(series["reward/strict_mean"], [(50, 0.75, 100)])
            self.assertNotIn("not_a_scalar", series)

            writer.add_event(Event(wall_time=101, step=50, summary=Summary(value=[
                Summary.Value(tag="actor/pg_loss", simple_value=-0.25),
            ])))
            writer.flush()
            self.assertEqual(reader.read()["actor/pg_loss"], [
                (50, -0.125, 100), (50, -0.25, 101),
            ])
        finally:
            writer.close()

    def test_diagnostics_preserve_repeated_steps_nonfinite_and_skip_raw_batches(self):
        self.write_records(
            {"event": "generation_batch", "global_step": 5, "metrics": {"reward/raw": 0.4}},
            {"event": "optimizer_step", "global_step": 5,
             "metrics": {"reward/strict_mean": 0.5, "actor/pg_loss": float("nan"),
                         "enabled": True, "description": "not a scalar"}},
            {"event": "skipped_update", "global_step": 5,
             "metrics": {"reward/strict_mean": 0.0, "actor/pg_loss": float("inf")}},
            {"event": "optimizer_step", "global_step": 6,
             "metrics": {"actor/pg_loss": float("-inf")}},
        )
        reader = plots.MetricReader(self.run)
        series = reader.read()
        self.assertEqual(reader.active_source, "diagnostics")
        self.assertEqual(set(series), {"reward/strict_mean", "actor/pg_loss"})
        self.assertEqual(series["reward/strict_mean"], [(5, 0.5, None), (5, 0.0, None)])
        self.assertTrue(math.isnan(series["actor/pg_loss"][0][1]))
        self.assertEqual([point[1] for point in series["actor/pg_loss"][1:]],
                         [float("inf"), float("-inf")])
        self.assertEqual(len(reader.read()["actor/pg_loss"]), 3)

    def test_auto_combines_buffered_tensorboard_with_latest_jsonl_without_duplicates(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        writer = EventFileWriter(str(self.run / "tensorboard"))
        try:
            def record(step, **metrics):
                writer.add_event(Event(wall_time=100 + step, step=step, summary=Summary(value=[
                    Summary.Value(tag=tag, simple_value=value) for tag, value in metrics.items()
                ])))

            record(0, **{"val/reward": 0.5})
            record(49, **{"actor/pg_loss": 0.75})
            record(50, **{"actor/pg_loss": 0.25, "sampling/skipped": 1})
            writer.flush()
            self.write_records(
                {"event": "optimizer_step", "global_step": 50, "metrics": {"actor/pg_loss": 0.25}},
                {"event": "skipped_update", "global_step": 50, "metrics": {"sampling/skipped": 1}},
                {"event": "optimizer_step", "global_step": 51, "metrics": {"actor/pg_loss": 0.125}},
            )
            reader = plots.MetricReader(self.run)
            series = reader.read()
            self.assertEqual(reader.active_source, "tensorboard+diagnostics")
            self.assertEqual(series["val/reward"], [(0, 0.5, 100)])
            self.assertEqual(series["actor/pg_loss"], [(49, 0.75, 149), (50, 0.25, None), (51, 0.125, None)])
            self.assertEqual(series["sampling/skipped"], [(50, 1.0, None)])

            self.write_records(
                {"event": "skipped_update", "global_step": 50, "metrics": {"sampling/skipped": 2}},
                {"event": "skipped_update", "global_step": 50, "metrics": {"sampling/skipped": 3}},
            )
            series = reader.read()
            self.assertEqual(series["sampling/skipped"], [(50, 1.0, None), (50, 2.0, None), (50, 3.0, None)])
            # TB flushing its delayed copies must not duplicate already plotted JSONL records.
            record(50, **{"sampling/skipped": 2})
            record(50, **{"sampling/skipped": 3})
            record(51, **{"actor/pg_loss": 0.125})
            writer.flush()
            self.assertEqual(reader.read(), series)
            self.assertEqual(plots.MetricReader(self.run, "diagnostics").read()["actor/pg_loss"],
                             [(50, 0.25, None), (51, 0.125, None)])
        finally:
            writer.close()

    def test_partial_final_jsonl_record_is_consumed_once_after_completion(self):
        self.write_records({"event": "optimizer_step", "global_step": 1, "metrics": {"loss": 1.5}})
        fragment = json.dumps({"event": "optimizer_step", "global_step": 2, "metrics": {"loss": 0.5}})
        with self.diagnostics.open("ab") as handle:
            handle.write(fragment[:24].encode())
        reader = plots.MetricReader(self.run, "diagnostics")
        self.assertEqual(reader.read()["loss"], [(1, 1.5, None)])
        self.assertEqual(reader.read()["loss"], [(1, 1.5, None)])
        with self.diagnostics.open("ab") as handle:
            handle.write(fragment[24:].encode() + b"\n")
        self.assertEqual(reader.read()["loss"], [(1, 1.5, None), (2, 0.5, None)])
        self.assertEqual(len(reader.read()["loss"]), 2)

    def test_truncated_jsonl_is_reported_instead_of_mixing_two_runs(self):
        self.write_records({"event": "optimizer_step", "global_step": 1, "metrics": {"loss": 1.0}})
        reader = plots.MetricReader(self.run, "diagnostics")
        reader.read()
        self.diagnostics.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "truncated"):
            reader.read()

    def test_explicit_tensorboard_source_does_not_silently_use_jsonl(self):
        self.write_records({"event": "optimizer_step", "global_step": 1, "metrics": {"loss": 1.0}})
        reader = plots.MetricReader(self.run, "tensorboard")
        self.assertEqual(reader.read(), {})
        self.assertEqual(reader.active_source, "tensorboard")

    def test_real_artifacts_are_safe_and_export_only_after_new_metrics(self):
        collision_tags = ["../../actor/pg_loss", "..\\..\\actor\\pg_loss", "actor:pg_loss", "actor/pg_loss"]
        metrics = dict.fromkeys(collision_tags, 0.25)
        metrics["reward/strict_mean"] = float("nan")
        self.write_records({"event": "optimizer_step", "global_step": 50, "metrics": metrics},
                           {"event": "skipped_update", "global_step": 50,
                            "metrics": {"reward/strict_mean": float("inf")}})
        exporter = plots.PlotExporter(self.run)
        manifest = exporter.refresh()
        self.assertEqual(manifest["source"], "diagnostics")
        self.assertEqual(manifest["latest_step"], 50)
        self.assertEqual(manifest["metrics"]["reward/strict_mean"]["nonfinite"], 2)
        self.assert_png(exporter.output / "summary.png")
        filenames = [manifest["metrics"][tag]["file"] for tag in collision_tags]
        self.assertEqual(len(set(filenames)), len(collision_tags))
        for entry in manifest["metrics"].values():
            path = exporter.output / entry["file"]
            self.assertEqual(path.resolve().parent, exporter.output.resolve())
            self.assert_png(path)
        saved = json.loads((exporter.output / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)
        with (exporter.output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        reward_rows = [row for row in rows if row["tag"] == "reward/strict_mean"]
        self.assertEqual([row["step"] for row in reward_rows], ["50", "50"])
        self.assertTrue(math.isnan(float(reward_rows[0]["value"])))
        self.assertEqual(float(reward_rows[1]["value"]), float("inf"))
        self.assertFalse(list(exporter.output.glob(".plot-*")))
        with patch.object(plots, "render_plots", side_effect=AssertionError("redrew unchanged data")):
            self.assertIsNone(exporter.refresh())
        self.write_records({"event": "optimizer_step", "global_step": 51, "metrics": {"actor/pg_loss": 0.1}})
        with patch.object(plots, "render_plots", wraps=plots.render_plots) as render:
            self.assertEqual(exporter.refresh()["latest_step"], 51)
            render.assert_called_once()

    def test_empty_run_does_not_create_plot_directory(self):
        exporter = plots.PlotExporter(self.run)
        self.assertIsNone(exporter.refresh())
        self.assertFalse(exporter.output.exists())

    def test_monitor_final_export_failure_does_not_mask_training_failure(self):
        first_refresh = threading.Event()
        monitor = plots.StaticPlotMonitor(self.run)
        self.assertEqual(monitor.interval, 600)

        def failed_refresh():
            first_refresh.set()
            raise ValueError("malformed live metric")

        with patch.object(monitor.exporter, "refresh", side_effect=failed_refresh) as refresh, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "original training failure"):
                monitor.start()
                try:
                    self.assertTrue(first_refresh.wait(timeout=5), "monitor did not start")
                    raise RuntimeError("original training failure")
                finally:
                    monitor.stop()
            self.assertEqual(refresh.call_count, 2)
        self.assertFalse(monitor.thread.is_alive())
        log = (self.run / "plots.log").read_text(encoding="utf-8")
        self.assertIn("600s", log)
        self.assertEqual(log.count("training continues"), 2)

    def test_monitor_log_open_failure_is_exposed_and_does_not_wait_forever(self):
        monitor = plots.StaticPlotMonitor(self.run)
        monitor.log_path.mkdir()
        with patch.object(monitor.exporter, "refresh") as refresh, patch("builtins.print"):
            monitor.start()
            monitor.thread.join(timeout=5)
            self.assertFalse(monitor.thread.is_alive())
            self.assertIsInstance(monitor.error, str)
            self.assertTrue(monitor.error)
            refresh.assert_not_called()
            monitor.stop()

    def test_explicit_output_keeps_monitor_log_outside_source_run(self):
        target = Path(self.temporary.name) / "offline chart directory"
        monitor = plots.StaticPlotMonitor(self.run, output=target)
        first_refresh = threading.Event()
        with patch.object(monitor.exporter, "refresh", side_effect=lambda: first_refresh.set() and None):
            monitor.start()
            try:
                self.assertTrue(first_refresh.wait(timeout=5))
            finally:
                monitor.stop()
        self.assertIsNone(monitor.error)
        self.assertEqual(monitor.log_path, target / "export.log")
        self.assertTrue((target / "export.log").is_file())
        self.assertFalse((self.run / "plots.log").exists())

    def test_cli_exports_existing_run_without_training_or_web_server(self):
        self.write_records({"event": "optimizer_step", "global_step": 59,
                            "metrics": {"actor/pg_loss": 0.25}})
        target = self.run / "exported charts"
        result = subprocess.run([
            sys.executable, "-B", str(ROOT / "scripts/export_grpo_plots.py"),
            "--run-dir", str(self.run), "--output", str(target), "--source", "diagnostics",
        ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("through step 59", result.stdout)
        self.assert_png(target / "summary.png")
        self.assertTrue((target / "metrics.csv").is_file())
        self.assertTrue((target / "index.json").is_file())
        self.assertFalse((self.run / "train.log").exists())

    def test_cli_watch_reports_failed_monitor_instead_of_waiting(self):
        from scripts import export_grpo_plots

        errors = io.StringIO()
        with patch.object(export_grpo_plots, "StaticPlotMonitor") as monitor_class, \
                patch.object(export_grpo_plots, "check_plot_dependencies"), \
                patch.object(sys, "argv", ["export_grpo_plots.py", "--run-dir", str(self.run), "--watch", "600"]), \
                patch.object(sys, "stderr", errors):
            monitor = monitor_class.return_value
            monitor.thread.is_alive.return_value = False
            monitor.error = "PermissionError: cannot open export.log"
            with self.assertRaises(SystemExit) as failure:
                export_grpo_plots.main()
            self.assertEqual(failure.exception.code, 1)
            monitor.start.assert_called_once_with()
            monitor.stop.assert_called_once_with()
            monitor.thread.join.assert_not_called()
        self.assertIn("cannot open export.log", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
