"""Focused data-selection, completion-log and desktop process regression checks.

The GUI checks call this application's Tk callbacks directly with a hidden root.
Subprocesses and the adaptation engine are replaced with controlled fakes; no
GPU experiment or external application is launched by this suite.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

TEST_V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_V1))
sys.path.insert(0, str(TEST_V1 / "code"))

from Paradigm.tta_composition.config import CompositionConfig
from Paradigm.tta_composition.data_selection import TargetCase, discover_targets
from Paradigm.tta_composition.gui_process import ProcessEvent, RunProcess
from Paradigm.tta_composition.gui_hyperparameters import HyperparameterDialog, default_hyperparameters, parse_hyperparameters
from Paradigm.tta_composition.result_summary import format_result


def write_image(path: Path, color: int = 255) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=color).save(path)
    return path


class DataSelectionChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="tta data with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "custom images"
        self.masks = self.root / "custom masks"

    def test_custom_discovery_is_sorted_and_does_not_open_masks(self) -> None:
        write_image(self.images / "B.PNG")
        write_image(self.images / "a.png")
        write_image(self.masks / "A_mask.png")
        with patch.object(Image, "open", side_effect=AssertionError("Discovery read image/mask contents")):
            cases = discover_targets(self.root, "custom", self.images, self.masks)
        self.assertEqual([case.image_path.stem for case in cases], ["a", "B"])
        self.assertEqual(len(cases[0].mask_paths), 1)
        self.assertEqual(cases[1].mask_paths, ())

    def test_custom_images_without_masks_support_prediction_only(self) -> None:
        write_image(self.images / "case.png")
        cases = discover_targets(self.root, "custom", self.images)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].mask_paths, ())

    def test_duplicate_image_stems_are_rejected(self) -> None:
        write_image(self.images / "same.png")
        write_image(self.images / "same.jpg")
        with self.assertRaises(ValueError):
            discover_targets(self.root, "custom", self.images)

    def test_ambiguous_mask_matches_are_rejected(self) -> None:
        write_image(self.images / "case.png")
        write_image(self.masks / "case.png")
        write_image(self.masks / "case_mask.png")
        with self.assertRaises(ValueError):
            discover_targets(self.root, "custom", self.images, self.masks)

    def test_missing_or_empty_image_folder_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            discover_targets(self.root, "custom", self.images)
        self.images.mkdir()
        with self.assertRaises(ValueError):
            discover_targets(self.root, "custom", self.images)

    def test_explicit_missing_mask_folder_is_rejected(self) -> None:
        write_image(self.images / "case.png")
        with self.assertRaises(FileNotFoundError):
            discover_targets(self.root, "custom", self.images, self.masks)

    def test_montgomery_requires_both_lung_masks_for_evaluation(self) -> None:
        dataset = self.root / "data" / "raw" / "montgomery"
        write_image(dataset / "images" / "one.png")
        write_image(dataset / "images" / "two.png")
        write_image(dataset / "left_mask" / "one.png")
        write_image(dataset / "left_mask" / "two.png")
        write_image(dataset / "right_mask" / "two.png")
        cases = discover_targets(self.root, "montgomery")
        self.assertEqual([len(case.mask_paths) for case in cases], [0, 2])

    def test_sz_cxr_uses_single_masks(self) -> None:
        dataset = self.root / "data" / "raw" / "sz_cxr"
        write_image(dataset / "images" / "case.png")
        write_image(dataset / "masks" / "case_mask.png")
        cases = discover_targets(self.root, "sz_cxr")
        self.assertEqual(len(cases[0].mask_paths), 1)

    def test_builtin_image_override_uses_masks_from_selected_dataset(self) -> None:
        alternate = self.root / "another Montgomery"
        write_image(alternate / "images" / "case.png")
        left = write_image(alternate / "left_mask" / "case.png")
        right = write_image(alternate / "right_mask" / "case.png")
        cases = discover_targets(self.root, "montgomery", alternate / "images")
        self.assertEqual(set(cases[0].mask_paths), {left, right})

    def test_selected_data_paths_roundtrip_in_config(self) -> None:
        config = CompositionConfig(dataset="custom", image_dir=self.images, mask_dir=self.masks)
        config_file = self.root / "config.json"
        config_file.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        restored = CompositionConfig.from_json(config_file)
        self.assertEqual(restored.dataset, "custom")
        self.assertEqual(restored.image_dir, self.images)
        self.assertEqual(restored.mask_dir, self.masks)


class SummaryChecks(unittest.TestCase):
    @staticmethod
    def summary(evaluated: int = 2) -> dict[str, object]:
        return {
            "event": "completed", "output": "C:/result with spaces", "methods": [],
            "cases": 2, "evaluated_cases": evaluated, "device": "cpu", "dataset": "custom",
            "metrics": {"Dice_mean": 0.8125, "Dice_std": 0.025, "HD95_mean": 4.125,
                        "HD95_std": 0.125, "JI_mean": 0.685, "JI_std": 0.05} if evaluated else {},
            "elapsed_seconds": 1.25,
        }

    def test_completion_text_contains_quality_metrics_and_output(self) -> None:
        text = format_result(self.summary())
        self.assertIn("Dice", text)
        self.assertIn("HD95", text)
        self.assertIn("0.8125", text)
        self.assertIn("C:/result with spaces", text)
        self.assertIn("2", text)

    def test_unlabelled_completion_does_not_claim_a_dice_score(self) -> None:
        text = format_result(self.summary(0))
        self.assertIn("2", text)
        self.assertNotIn("Dice 0", text)
        self.assertNotIn("0.8125", text)
        self.assertTrue(any(word in text for word in ("mask", "標籤", "評估", "遮罩")))


class RunnerChecks(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="tta cli data with spaces ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_unlabelled_cases_are_skipped_without_opening_any_masks(self) -> None:
        from Paradigm.tta_composition import run

        case = TargetCase("unlabelled", self.directory / "missing-image.png")
        with patch.object(Image, "open", side_effect=AssertionError("Unlabelled case read a file")):
            rows = run._evaluate(self.directory, [case], 16, lambda *_: {})
        self.assertEqual(rows, [])

    def test_binary_zero_one_lung_masks_are_unioned_after_adaptation(self) -> None:
        import numpy as np
        from Paradigm.tta_composition import run

        left = np.zeros((16, 16), dtype=np.uint8)
        left[1:4, 1:4] = 1
        right = np.zeros((16, 16), dtype=np.uint8)
        right[8:12, 8:12] = 1
        paths = (self.directory / "left.png", self.directory / "right.png")
        Image.fromarray(left).save(paths[0])
        Image.fromarray(right).save(paths[1])
        target = run._load_target_mask(paths, 16)
        self.assertEqual(int(target.sum()), 9 + 16)
        self.assertEqual(target.dtype, np.dtype(bool))

    def test_explicit_empty_methods_override_a_saved_config(self) -> None:
        from Paradigm.tta_composition import run

        config = CompositionConfig(methods=("SmaRT",))
        path = self.directory / "selected config.json"
        path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        args = run._parse_args(["--config", str(path), "--methods", "", "--dataset", "custom",
                                "--image-dir", str(self.directory), "--device", "cpu"])
        selected = run._config_from_args(args)
        self.assertEqual(selected.methods, ())
        self.assertEqual(selected.dataset, "custom")
        self.assertEqual(selected.image_dir, self.directory)

    def test_cli_adaptation_steps_override_a_saved_config(self) -> None:
        from Paradigm.tta_composition import run

        config = CompositionConfig(methods=("DLTTA",), adaptation_steps=2)
        path = self.directory / "iteration config.json"
        path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        selected = run._config_from_args(run._parse_args([
            "--config", str(path), "--adaptation-steps", "5",
        ]))
        self.assertEqual(selected.adaptation_steps, 5)
        with self.assertRaises(ValueError):
            run._config_from_args(run._parse_args(["--adaptation-steps", "0"]))

    def test_cli_saves_final_metrics_and_reads_masks_after_all_predictions(self) -> None:
        import torch
        from types import SimpleNamespace
        from Paradigm.tta_composition import run
        from Paradigm.tta_composition.contracts import UpdateRecord

        images = self.directory / "custom images"
        masks = self.directory / "custom masks"
        write_image(images / "a.png", 127)
        write_image(images / "b.png", 128)
        write_image(masks / "a_mask.png")
        checkpoint = self.directory / "fake checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint hash fixture")
        output = self.directory / "completed output"
        adapted: list[str] = []

        class FakeReference(torch.nn.Module):
            def forward(self, image, return_features=False):
                logits = torch.cat((torch.zeros_like(image), torch.ones_like(image)), dim=1)
                return (logits, image) if return_features else logits

        class FakeEngine:
            def __init__(self, bundle, config):
                self.config = config

            def step(self, image, sample_id):
                adapted.append(sample_id)
                prediction = torch.ones((1, self.config.image_size, self.config.image_size), dtype=torch.long)
                record = UpdateRecord(len(adapted), {}, {}, {}, {}, 0, 0.001)
                return prediction, record, {}

            def audit_state(self):
                return {"stub_engine": True}

        original_open = Image.open

        def guard_mask_reads(path, *args, **kwargs):
            if Path(path).parent == masks:
                self.assertEqual(adapted, ["a", "b"], "Mask contents were read before all adaptation steps")
            return original_open(path, *args, **kwargs)

        arguments = ["run", "--dataset", "custom", "--image-dir", str(images),
                     "--mask-dir", str(masks), "--checkpoint", str(checkpoint),
                     "--output", str(output), "--device", "cpu", "--methods", "GraTa"]
        captured = io.StringIO()
        bundle = SimpleNamespace(reference=FakeReference())
        with patch.object(sys, "argv", arguments), patch.object(run, "build_model_bundle", return_value=bundle), \
                patch.object(run, "CompositionEngine", FakeEngine), patch.object(Image, "open", guard_mask_reads), \
                redirect_stdout(captured):
            code = run.main()
        self.assertEqual(code, 0)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["event"], "completed")
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["evaluated_cases"], 1)
        self.assertEqual(summary["metrics"]["Dice_mean"], 1.0)
        self.assertEqual(summary["metrics"]["HD95_mean"], 0.0)
        self.assertEqual(json.loads(captured.getvalue().splitlines()[-1]), summary)
        self.assertEqual(len(list((output / "predictions" / "GraTa").glob("*.png"))), 2)
        self.assertTrue((output / "metrics.csv").is_file())
        self.assertTrue((output / "data_manifest.json").is_file())
        self.assertEqual(summary["baseline"]["cases"], 2)
        self.assertEqual(summary["comparison"]["evaluated_cases"], 1)
        self.assertEqual(summary["comparison"]["delta"]["Dice_mean"], 0.0)
        self.assertTrue((output / "comparison.csv").is_file())


class FinishedChild:
    def __init__(self, code: int = 0) -> None:
        self.stdout: io.StringIO | BlockingStdout = io.StringIO("first log line\nsecond log line\n")
        self.returncode: int | None = None
        self.exit_code = code

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self.exit_code
        return self.exit_code

    def terminate(self) -> None:
        self.exit_code = -15
        self.returncode = -15

    def kill(self) -> None:
        self.terminate()


class BlockingStdout:
    def __init__(self, stopped: threading.Event) -> None:
        self.stopped = stopped

    def __iter__(self):
        self.stopped.wait(5)
        return iter(())

    def close(self) -> None:
        self.stopped.set()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


class BlockingChild(FinishedChild):
    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()
        self.stdout = BlockingStdout(self.stopped)

    def wait(self, timeout: float | None = None) -> int:
        if not self.stopped.wait(timeout or 5):
            raise AssertionError("Fake child failed to stop")
        return super().wait(timeout)

    def terminate(self) -> None:
        super().terminate()
        self.stopped.set()


def collect_until_terminal(runner: RunProcess) -> list[ProcessEvent]:
    events = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        event = runner.events.get(timeout=max(deadline - time.monotonic(), 0.01))
        events.append(event)
        if event.kind in ("exit", "error"):
            return events
    raise AssertionError("Runner did not report a terminal event")


class ProcessChecks(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="tta subprocess paths with spaces ")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "fresh run"

    def test_command_keeps_paths_as_list_arguments_and_streams_logs(self) -> None:
        runner = RunProcess(Path(sys.executable), TEST_V1)
        arguments = ["--image-dir", "C:/data with spaces", "--methods", ""]
        with patch("Paradigm.tta_composition.gui_process.subprocess.Popen", return_value=FinishedChild()) as popen:
            runner.start(arguments, self.output)
            events = collect_until_terminal(runner)
        command = popen.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertIn("C:/data with spaces", command)
        self.assertFalse(popen.call_args.kwargs.get("shell", False))
        self.assertEqual([event.payload for event in events if event.kind == "line"],
                         ["first log line\n", "second log line\n"])
        self.assertEqual(events[-1].payload, 0)
        runner.close()

    def test_nonzero_child_exit_remains_a_failure(self) -> None:
        runner = RunProcess(Path(sys.executable), TEST_V1)
        with patch("Paradigm.tta_composition.gui_process.subprocess.Popen", return_value=FinishedChild(3)):
            runner.start([], self.output)
            events = collect_until_terminal(runner)
        self.assertEqual(events[-1].kind, "exit")
        self.assertEqual(events[-1].payload, 3)
        runner.close()

    def test_cancel_stops_child_and_reports_termination(self) -> None:
        runner = RunProcess(Path(sys.executable), TEST_V1)
        child = BlockingChild()
        with patch("Paradigm.tta_composition.gui_process.subprocess.Popen", return_value=child):
            runner.start([], self.output)
            self.assertTrue(runner.running)
            runner.cancel()
            events = collect_until_terminal(runner)
        self.assertEqual(events[-1].kind, "exit")
        self.assertNotEqual(events[-1].payload, 0)
        self.assertFalse(runner.running)
        runner.close()

    def test_launch_failure_is_preserved_in_execution_log(self) -> None:
        runner = RunProcess(Path(sys.executable), TEST_V1)
        with patch("Paradigm.tta_composition.gui_process.subprocess.Popen", side_effect=OSError("fake launch error")):
            with self.assertRaises(OSError):
                runner.start([], self.output)
        self.assertFalse(runner.running)
        self.assertIn("fake launch error", (self.output / "execution.log").read_text(encoding="utf-8"))

    def test_gui_configuration_is_written_before_child_starts(self) -> None:
        runner = RunProcess(Path(sys.executable), TEST_V1)
        payload = {"image_size": 128, "normalization": "percentile", "label": "超參數"}
        config_path = self.output / "gui_config.json"

        def launch(*args, **kwargs):
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), payload)
            self.assertIn(str(config_path), args[0])
            return FinishedChild()

        with patch("Paradigm.tta_composition.gui_process.subprocess.Popen", side_effect=launch):
            runner.start(["--config", str(config_path)], self.output, payload)
            events = collect_until_terminal(runner)
        self.assertEqual(events[-1].payload, 0)
        runner.close()

    def test_nonfinite_configuration_is_rejected_before_child_starts(self) -> None:
        runner = RunProcess(Path(sys.executable), TEST_V1)
        with patch("Paradigm.tta_composition.gui_process.subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                runner.start([], self.output, {"learning_rate": float("nan")})
        popen.assert_not_called()


class FakeRunProcess:
    def __init__(self, *args, **kwargs) -> None:
        self.events: queue.Queue[ProcessEvent] = queue.Queue()
        self.running = False
        self.arguments: list[str] = []
        self.output_dir: Path | None = None
        self.config_payload: dict | None = None
        self.cancelled = False

    def start(self, arguments: list[str], output_dir: Path, config_payload: dict | None = None) -> None:
        self.arguments = arguments
        self.output_dir = output_dir
        self.config_payload = config_payload
        self.running = True

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.running = False


class GUIFunctionChecks(unittest.TestCase):
    def setUp(self) -> None:
        from Paradigm.tta_composition import gui

        self.temporary = tempfile.TemporaryDirectory(prefix="tta gui with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.images = self.directory / "images"
        write_image(self.images / "case.png")
        self.checkpoint = self.directory / "fake checkpoint.pth"
        self.checkpoint.write_bytes(b"fake checkpoint - never loaded by GUI tests")
        self.process_patch = patch.object(gui, "RunProcess", FakeRunProcess)
        self.process_patch.start()
        self.addCleanup(self.process_patch.stop)
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk unavailable: {error}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = gui.CompositionGUI(self.root)
        self.app.python = Path(sys.executable)
        self.app.checkpoint_var.set(str(self.checkpoint))
        self.app.output_var.set(str(self.directory / "results"))
        self.app.dataset_var.set("自訂資料夾")
        self.app.image_dir_var.set(str(self.images))
        self.app.mask_dir_var.set("")
        self.app.max_cases_var.set("1")

    def test_gui_uses_launching_interpreter_in_an_independent_clone(self) -> None:
        from Paradigm.tta_composition import gui

        isolated_python = str(self.directory / "isolated runtime" / "python.exe")
        with patch.object(gui.sys, "executable", isolated_python):
            isolated_app = gui.CompositionGUI(self.root)
        self.addCleanup(lambda: self.root.after_cancel(isolated_app._poll_id))
        self.assertEqual(isolated_app.python, Path(isolated_python))

    def _select_sz_cxr_fixture(self) -> Path:
        dataset = self.directory / "data" / "raw" / "sz_cxr"
        write_image(dataset / "images" / "case.png")
        write_image(dataset / "masks" / "case_mask.png", 0)
        self.app.test_v1 = self.directory
        self.app.dataset_var.set("SZ-CXR")
        self.app._dataset_changed()
        return dataset

    def test_sz_cxr_image_browser_uses_selected_sibling_masks(self) -> None:
        original = self._select_sz_cxr_fixture()
        alternate = self.directory / "alternate SZ-CXR"
        write_image(alternate / "images" / "case.png")
        selected_mask = write_image(alternate / "masks" / "case_mask.png")
        with patch("Paradigm.tta_composition.gui.filedialog.askdirectory", return_value=str(alternate / "images")):
            self.app._browse_images()
        case = self.app._discover_data()[0]
        self.assertEqual(case.image_path, alternate / "images" / "case.png")
        self.assertEqual(case.mask_paths, (selected_mask,))
        self.assertNotIn(original / "masks" / "case_mask.png", case.mask_paths)
        self.assertEqual(self.app.mask_dir_var.get(), "")
        self.assertIn("自動使用", self.app.data_info_var.get())
        self.app._start()
        self.assertNotIn("--mask-dir", self.app.process.arguments)

    def test_sz_cxr_image_browser_without_sibling_masks_is_prediction_only(self) -> None:
        self._select_sz_cxr_fixture()
        alternate_images = self.directory / "unlabelled SZ-CXR" / "images"
        write_image(alternate_images / "case.png")
        with patch("Paradigm.tta_composition.gui.filedialog.askdirectory", return_value=str(alternate_images)):
            self.app._browse_images()
        self.assertEqual(self.app._discover_data()[0].mask_paths, ())
        self.assertIn("可評估 0 張", self.app.data_info_var.get())
        self.app._start()
        self.assertTrue(self.app.process.running)
        self.assertNotIn("--mask-dir", self.app.process.arguments)

    def test_manually_changed_image_folder_clears_old_explicit_masks_without_scanning(self) -> None:
        old_masks = self.directory / "old masks"
        write_image(old_masks / "case_mask.png")
        self.app.mask_dir_var.set(str(old_masks))
        alternate_images = self.directory / "new images"
        write_image(alternate_images / "case.png")
        with patch("Paradigm.tta_composition.gui.discover_targets", side_effect=AssertionError("Path edit scanned data")):
            self.app.image_dir_var.set(str(alternate_images))
        self.assertEqual(self.app.mask_dir_var.get(), "")
        self.app._check_data()
        self.assertEqual(self.app._discover_data()[0].mask_paths, ())

    def test_same_image_folder_preserves_selected_masks_and_runner_arguments(self) -> None:
        masks = self.directory / "selected masks"
        selected_mask = write_image(masks / "case_mask.png")
        with patch("Paradigm.tta_composition.gui.filedialog.askdirectory", return_value=str(masks)):
            self.app._browse_masks()
        self.app.image_dir_var.set(str(self.images))
        self.assertEqual(self.app.mask_dir_var.get(), str(masks))
        self.assertEqual(self.app._discover_data()[0].mask_paths, (selected_mask,))
        self.app._start()
        arguments = self.app.process.arguments
        self.assertEqual(arguments[arguments.index("--mask-dir") + 1], str(masks))

    def test_custom_selection_and_all_cases_reach_runner_arguments(self) -> None:
        self.app.all_cases_var.set(True)
        self.app._start()
        arguments = self.app.process.arguments
        self.assertIn("--dataset", arguments)
        self.assertEqual(arguments[arguments.index("--dataset") + 1], "custom")
        self.assertEqual(arguments[arguments.index("--image-dir") + 1], str(self.images))
        self.assertNotIn("--max-cases", arguments)
        self.assertEqual(arguments[arguments.index("--methods") + 1], "")

    def test_completion_updates_both_summary_and_log_with_dice(self) -> None:
        self.app._start()
        runner = self.app.process
        summary = SummaryChecks.summary()
        summary["output"] = str(self.app.current_output)
        assert self.app.current_output is not None, "GUI did not create an output directory"
        self.app.current_output.mkdir(parents=True)
        runner.events.put(ProcessEvent("line", json.dumps(summary) + "\n"))
        runner.running = False
        runner.events.put(ProcessEvent("exit", 0))
        self.app._drain_output()
        self.assertIn("Dice", self.app.summary_var.get())
        self.assertIn("0.8125", self.app.summary_var.get())
        self.assertIn("Dice", self.app.log.get("1.0", "end"))
        self.assertIn("完成", self.app.status_var.get())
        self.assertEqual(str(self.app.stop_button.cget("state")), "disabled")
        final_log = (self.app.current_output / "execution.log").read_text(encoding="utf-8")
        self.assertIn("Dice：0.8125", final_log)
        self.assertIn("狀態：完成", final_log)

    def test_stop_is_reported_as_cancelled_instead_of_success(self) -> None:
        self.app._start()
        runner = self.app.process
        self.app._stop()
        self.assertTrue(runner.cancelled)
        runner.running = False
        runner.events.put(ProcessEvent("exit", -15))
        self.app._drain_output()
        self.assertTrue(any(word in self.app.status_var.get() for word in ("停止", "取消")))
        self.assertEqual(str(self.app.run_button.cget("state")), "normal")

    def test_failed_exit_does_not_show_stale_success_metrics(self) -> None:
        self.app.summary_var.set("old Dice 0.999")
        self.app._start()
        runner = self.app.process
        runner.running = False
        runner.events.put(ProcessEvent("exit", 3))
        self.app._drain_output()
        self.assertIn("失敗", self.app.status_var.get())
        self.assertNotIn("0.999", self.app.summary_var.get())

    def test_exit_without_a_summary_is_incomplete(self) -> None:
        self.app._start()
        self.app.process.running = False
        self.app.process.events.put(ProcessEvent("exit", 0))
        self.app._drain_output()
        self.assertIn("不完整", self.app.status_var.get())
        self.assertIn("不完整", self.app.summary_var.get())

    def test_saved_summary_is_used_when_completion_line_is_missing(self) -> None:
        self.app._start()
        assert self.app.current_output is not None, "GUI did not create an output directory"
        self.app.current_output.mkdir(parents=True)
        summary = SummaryChecks.summary(0)
        summary["output"] = str(self.app.current_output)
        (self.app.current_output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.app.process.running = False
        self.app.process.events.put(ProcessEvent("exit", 0))
        self.app._drain_output()
        self.assertIn("完成", self.app.status_var.get())
        self.assertIn("未計算", self.app.summary_var.get())

    def test_invalid_seed_is_rejected_before_process_starts(self) -> None:
        self.app.seed_var.set("4294967296")
        with patch("Paradigm.tta_composition.gui.messagebox.showerror") as error:
            self.app._start()
        error.assert_called_once()
        self.assertFalse(self.app.process.running)

    def test_missing_data_is_rejected_before_process_starts(self) -> None:
        self.app.image_dir_var.set(str(self.directory / "not present"))
        with patch("Paradigm.tta_composition.gui.messagebox.showerror") as error:
            self.app._start()
        error.assert_called_once()
        self.assertFalse(self.app.process.running)

    def test_progress_log_updates_case_count_and_status(self) -> None:
        self.app._start()
        self.app.process.events.put(ProcessEvent("line", json.dumps({
            "event": "progress", "completed": 1, "total": 2, "image_id": "case",
        }) + "\n"))
        self.app._drain_output()
        self.assertIn("1/2", self.app.status_var.get())
        self.assertEqual(float(self.app.progress.cget("value")), 1.0)
        self.assertIn("case", self.app.log.get("1.0", "end"))

    def test_nondefault_hyperparameters_reach_runner_configuration(self) -> None:
        values = default_hyperparameters()
        values.update(image_size=128, normalization="percentile", testfit_confidence=0.95,
                      vptta_memory_size=6, smart_structure_weight=0.25, adaptation_steps=5)
        values["lrs"]["VPTTA"] = 0.005
        self.app.hyperparameter_values = values
        self.app.method_vars["VPTTA"].set(True)
        self.app._start()
        runner = self.app.process
        self.assertTrue(runner.running)
        self.assertIsNotNone(runner.config_payload)
        payload = runner.config_payload
        self.assertEqual(payload["image_size"], 128)
        self.assertEqual(payload["normalization"], "percentile")
        self.assertEqual(payload["testfit_confidence"], 0.95)
        self.assertEqual(payload["vptta_memory_size"], 6)
        self.assertEqual(payload["smart_structure_weight"], 0.25)
        self.assertEqual(payload["adaptation_steps"], 5)
        self.assertEqual(payload["lrs"]["VPTTA"], 0.005)
        self.assertEqual(payload["methods"], ["VPTTA"])
        config_path = Path(runner.arguments[runner.arguments.index("--config") + 1])
        self.assertEqual(config_path, runner.output_dir / "gui_config.json")
        self.assertEqual(Path(runner.arguments[runner.arguments.index("--baseline-cache") + 1]),
                         self.directory / "results" / "baselines")
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        from Paradigm.tta_composition.run import _config_from_args, _parse_args

        restored = _config_from_args(_parse_args(["--config", str(config_path)]))
        self.assertEqual(restored.image_size, 128)
        self.assertEqual(restored.testfit_confidence, 0.95)
        self.assertEqual(restored.lrs["VPTTA"], 0.005)
        self.assertEqual(restored.smart_structure_weight, 0.25)
        self.assertEqual(restored.adaptation_steps, 5)

    def test_invalid_hyperparameters_are_rejected_before_process_starts(self) -> None:
        for updates in ({"image_size": 127}, {"testfit_confidence": 1.2},
                        {"smart_ema_decay": float("inf")}, {"vptta_prompt_strength": float("nan")},
                        {"adaptation_steps": 0}, {"adaptation_steps": "1.5"}):
            with self.subTest(updates=updates):
                values = default_hyperparameters()
                values.update(updates)
                self.app.hyperparameter_values = values
                with patch("Paradigm.tta_composition.gui.messagebox.showerror") as error:
                    self.app._start()
                error.assert_called_once()
                self.assertFalse(self.app.process.running)
        values = default_hyperparameters()
        values["lrs"]["SmaRT"] = float("nan")
        self.app.hyperparameter_values = values
        with patch("Paradigm.tta_composition.gui.messagebox.showerror") as error:
            self.app._start()
        error.assert_called_once()
        self.assertFalse(self.app.process.running)

    def test_hyperparameter_editor_applies_and_cancel_preserves_settings(self) -> None:
        values = default_hyperparameters()
        values["grata_noise_std"] = 0.07
        with patch("Paradigm.tta_composition.gui.show_hyperparameter_dialog", return_value=values):
            self.app._edit_hyperparameters()
        self.assertEqual(self.app.hyperparameter_values["grata_noise_std"], 0.07)
        self.assertIn("已調整", self.app.hyperparameter_hint_var.get())
        with patch("Paradigm.tta_composition.gui.show_hyperparameter_dialog", return_value=None):
            self.app._edit_hyperparameters()
        self.assertEqual(self.app.hyperparameter_values["grata_noise_std"], 0.07)

    def test_hyperparameter_editor_is_disabled_during_run(self) -> None:
        self.app._start()
        self.assertEqual(str(self.app.hyperparameters_button.cget("state")), "disabled")
        with patch("Paradigm.tta_composition.gui.show_hyperparameter_dialog") as dialog:
            self.app._edit_hyperparameters()
        dialog.assert_not_called()
        self.app.process.running = False
        self.app._handle_exit(3)
        self.assertEqual(str(self.app.hyperparameters_button.cget("state")), "normal")

    def test_dialog_apply_validates_and_does_not_mutate_original_values(self) -> None:
        values = default_hyperparameters()
        dialog = HyperparameterDialog(self.root, values)
        dialog.variables["testfit_confidence"].set("0.91")
        dialog.variables["lr_GraTa"].set("0.002")
        dialog.variables["adaptation_steps"].set("4")
        dialog._apply()
        self.assertEqual(dialog.result["testfit_confidence"], 0.91)
        self.assertEqual(dialog.result["lrs"]["GraTa"], 0.002)
        self.assertEqual(dialog.result["adaptation_steps"], 4)
        self.assertEqual(values, default_hyperparameters())

    def test_dialog_defaults_and_cancel_do_not_apply_changes(self) -> None:
        values = default_hyperparameters()
        values["testfit_confidence"] = 0.93
        dialog = HyperparameterDialog(self.root, values)
        dialog.variables["image_size"].set("128")
        dialog._restore_defaults()
        self.assertEqual(parse_hyperparameters(dialog._read_values()), default_hyperparameters())
        self.assertIsNone(dialog.result)
        dialog._cancel()
        self.assertIsNone(dialog.result)
        self.assertEqual(values["testfit_confidence"], 0.93)

    def test_dialog_invalid_apply_keeps_editor_open(self) -> None:
        dialog = HyperparameterDialog(self.root, default_hyperparameters())
        try:
            dialog.variables["grata_noise_std"].set("inf")
            with patch("Paradigm.tta_composition.gui_hyperparameters.messagebox.showerror") as error:
                dialog._apply()
            error.assert_called_once()
            self.assertTrue(dialog.winfo_exists())
            self.assertIsNone(dialog.result)
        finally:
            dialog._cancel()

    def test_baseline_progress_and_cache_ready_do_not_complete_experiment(self) -> None:
        self.app._start()
        for event in (
            {"event": "baseline", "status": "checking", "total": 7},
            {"event": "baseline", "status": "building", "total": 7},
            {"event": "baseline_progress", "completed": 4, "total": 7, "image_id": "base-case"},
        ):
            self.app._handle_line(json.dumps(event))
        self.assertIn("4/7", self.app.status_var.get())
        self.assertEqual(float(self.app.progress.cget("value")), 4)
        self.assertIn("base-case", self.app.log.get("1.0", "end"))
        for status in ("reused", "ready"):
            self.app._handle_line(json.dumps({"event": "baseline", "status": status, "total": 7}))
            self.assertTrue(self.app._running)
            self.assertIsNone(self.app._completion)
            self.assertEqual(float(self.app.progress.cget("value")), 7)
        self.app._handle_line(json.dumps({"event": "progress", "phase": "tta", "completed": 1,
                                         "total": 1, "image_id": "case"}))
        self.assertIn("TTA", self.app.status_var.get())
        self.assertEqual(float(self.app.progress.cget("maximum")), 1)

    def test_source_only_progress_and_evaluation_status_remain_nonterminal(self) -> None:
        self.app._start()
        self.app._handle_line(json.dumps({"event": "progress", "phase": "source_only", "completed": 1,
                                         "total": 1, "image_id": "case"}))
        self.assertIn("Source-only", self.app.status_var.get())
        self.app._handle_line(json.dumps({"event": "status", "message": "正在評估 Baseline 與本次結果…"}))
        self.assertIn("正在評估", self.app.status_var.get())
        self.assertTrue(self.app._running)
        self.assertIsNone(self.app._completion)


if __name__ == "__main__":
    unittest.main(verbosity=2)
