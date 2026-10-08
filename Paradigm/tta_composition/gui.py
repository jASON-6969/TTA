"""Desktop GUI for selecting and running TTA composition experiments."""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import CompositionConfig
from .contracts import METHODS
from .data_selection import discover_targets
from .gui_hyperparameters import default_hyperparameters, format_hyperparameter_hint, parse_hyperparameters, show_hyperparameter_dialog
from .gui_process import RunProcess
from .result_summary import format_result


class CompositionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TTA 組合框架")
        self.root.geometry("1100x840")
        self.root.minsize(960, 760)
        self.test_v1 = Path(__file__).resolve().parents[2]
        self.python = Path(sys.executable)
        self.checkpoint_var = tk.StringVar(
            value=str(self.test_v1 / "base_model" / "benchmark_cxr_base_no_source_aug" / "source_checkpoint.pth")
        )
        self.output_var = tk.StringVar(value=str(self.test_v1 / "composition_runs"))
        self.dataset_var = tk.StringVar(value="Montgomery")
        self.image_dir_var = tk.StringVar()
        self.mask_dir_var = tk.StringVar()
        self._previous_image_dir = self.image_dir_var.get()
        self.image_dir_var.trace_add("write", self._image_dir_changed)
        self.data_info_var = tk.StringVar(value="")
        self.all_cases_var = tk.BooleanVar(value=False)
        self.max_cases_var = tk.StringVar(value="8")
        self.seed_var = tk.StringVar(value="42")
        self.device_var = tk.StringVar(value="auto")
        self.hyperparameter_values = default_hyperparameters()
        self.hyperparameter_hint_var = tk.StringVar(value=format_hyperparameter_hint(self.hyperparameter_values))
        self.method_vars = {name: tk.BooleanVar(value=False) for name in METHODS}
        self.process = RunProcess(self.python, self.test_v1)
        self.current_output: Path | None = None
        self.summary_var = tk.StringVar(value="執行完成後，這裡會顯示 Base、TTA 與差異（Δ）。")
        self._running = False
        self._completion: dict | None = None
        self._read_error: str | None = None
        self._settings_states: list[tuple[tk.Widget, str]] = []
        self._build()
        self._dataset_changed()
        self._poll_id = self.root.after(120, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        shell = ttk.Frame(self.root, padding=18)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=1)

        title = ttk.Label(shell, text="TTA 組合框架", font=("Segoe UI", 19, "bold"))
        title.grid(row=0, column=0, sticky="w")
        ttk.Label(
            shell,
            text="先建立全資料集 Source-only baseline，再以本次影像比較 Base 與 TTA；結果與日誌會自動保存。",
        ).grid(row=1, column=0, sticky="w", pady=(3, 16))

        settings = ttk.LabelFrame(shell, text="實驗設定", padding=12)
        self.settings = settings
        settings.grid(row=2, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)

        method_row = ttk.Frame(settings)
        method_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Label(method_row, text="適配方法").pack(side="left", padx=(0, 12))
        for name in METHODS:
            ttk.Checkbutton(method_row, text=name, variable=self.method_vars[name]).pack(side="left", padx=(0, 10))
        ttk.Button(method_row, text="全選", command=self._select_all).pack(side="right", padx=(6, 0))
        ttk.Button(method_row, text="清除", command=self._clear_methods).pack(side="right")

        self._path_row(settings, 1, "Checkpoint", self.checkpoint_var, self._browse_checkpoint)
        self._path_row(settings, 2, "輸出目錄", self.output_var, self._browse_output)

        data_row = ttk.Frame(settings)
        data_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        ttk.Label(data_row, text="目標資料").pack(side="left", padx=(0, 14))
        dataset_box = ttk.Combobox(
            data_row, textvariable=self.dataset_var,
            values=("Montgomery", "SZ-CXR", "自訂資料夾"), state="readonly", width=18,
        )
        dataset_box.pack(side="left")
        dataset_box.bind("<<ComboboxSelected>>", self._dataset_changed)
        ttk.Button(data_row, text="檢查資料", command=self._check_data).pack(side="left", padx=10)
        self._path_row(settings, 4, "影像資料夾", self.image_dir_var, self._browse_images)
        self._path_row(settings, 5, "評估 mask（可選）", self.mask_dir_var, self._browse_masks)
        hints = ttk.Frame(settings)
        hints.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(3, 4))
        ttk.Label(hints, textvariable=self.data_info_var, wraplength=670).pack(side="left", fill="x", expand=True)
        ttk.Label(hints, textvariable=self.hyperparameter_hint_var, foreground="#52616b").pack(side="right", padx=(12, 0))

        numeric = ttk.Frame(settings)
        numeric.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(numeric, text="影像上限").pack(side="left")
        self.case_box = ttk.Spinbox(numeric, from_=1, to=10000, width=8, textvariable=self.max_cases_var)
        self.case_box.pack(side="left", padx=(8, 8))
        ttk.Checkbutton(numeric, text="全部影像", variable=self.all_cases_var, command=self._toggle_all_cases).pack(side="left", padx=(0, 18))
        ttk.Label(numeric, text="Seed").pack(side="left")
        ttk.Spinbox(numeric, from_=0, to=2147483647, width=10, textvariable=self.seed_var).pack(side="left", padx=(8, 20))
        ttk.Label(numeric, text="運算裝置").pack(side="left")
        ttk.Combobox(numeric, textvariable=self.device_var, values=("auto", "cpu", "cuda"), state="readonly", width=9).pack(side="left", padx=8)
        self.hyperparameters_button = ttk.Button(numeric, text="超參數…", command=self._edit_hyperparameters)
        self.hyperparameters_button.pack(side="right")

        status_frame = ttk.LabelFrame(shell, text="執行狀態", padding=10)
        status_frame.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(2, weight=1)
        controls = ttk.Frame(status_frame)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(controls, text="▶  開始執行", command=self._start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="■  停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(controls, mode="determinate", maximum=100, length=180)
        self.progress.pack(side="left", padx=(10, 0))
        self.status_var = tk.StringVar(value="就緒")
        self.open_output_button = ttk.Button(controls, text="開啟結果資料夾", command=self._open_output, state="disabled")
        self.open_output_button.pack(side="left", padx=10)
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")

        ttk.Label(status_frame, textvariable=self.summary_var, wraplength=1000, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10)
        )

        self.log = tk.Text(status_frame, height=16, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=2, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        ttk.Label(shell, text="未勾選方法 = Source-only。Mask 僅供完成適配後評估；HD95 為 resize 後的像素距離。", foreground="#52616b").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )

    @staticmethod
    def _path_row(parent: tk.Misc, row: int, label: str, variable: tk.StringVar, browse) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="瀏覽…", command=browse).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _select_all(self) -> None:
        for variable in self.method_vars.values():
            variable.set(True)

    def _clear_methods(self) -> None:
        for variable in self.method_vars.values():
            variable.set(False)

    def _edit_hyperparameters(self) -> None:
        if self._running:
            return
        values = show_hyperparameter_dialog(self.root, self.hyperparameter_values)
        if values is not None:
            self.hyperparameter_values = values
            self.hyperparameter_hint_var.set(format_hyperparameter_hint(values))

    def _browse_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="選擇 checkpoint",
            filetypes=(("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")),
            initialdir=str(Path(self.checkpoint_var.get()).parent),
        )
        if selected:
            self.checkpoint_var.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(title="選擇輸出目錄", initialdir=self.output_var.get())
        if selected:
            self.output_var.set(selected)

    def _dataset_key(self) -> str:
        return {"Montgomery": "montgomery", "SZ-CXR": "sz_cxr", "自訂資料夾": "custom"}[self.dataset_var.get()]

    def _dataset_changed(self, _event=None) -> None:
        key = self._dataset_key()
        self.mask_dir_var.set("")
        if key == "custom":
            self.image_dir_var.set("")
        else:
            dataset_root = self.test_v1 / "data" / "raw" / key
            self.image_dir_var.set(str(dataset_root / "images"))
        self._refresh_data_info()

    def _image_dir_changed(self, *_args) -> None:
        image_dir = self.image_dir_var.get()
        if image_dir != self._previous_image_dir:
            self._previous_image_dir = image_dir
            self.mask_dir_var.set("")

    def _refresh_data_info(self) -> None:
        try:
            records = self._discover_data()
            paired = sum(bool(record.mask_paths) for record in records)
            hint = "Mask 配對：相同檔名或 <影像名稱>_mask。"
            if not self.mask_dir_var.get().strip():
                if self._dataset_key() == "montgomery":
                    hint = "Montgomery 自動使用影像資料夾旁的 left_mask／right_mask。"
                elif self._dataset_key() == "sz_cxr":
                    hint = "SZ-CXR 自動使用影像資料夾旁的 masks；無標籤時僅儲存預測。"
            self.data_info_var.set(f"影像 {len(records)} 張；可評估 {paired} 張。" + hint)
        except (OSError, ValueError) as error:
            self.data_info_var.set(f"資料尚未就緒：{error}")

    def _discover_data(self):
        images = self.image_dir_var.get().strip()
        masks = self.mask_dir_var.get().strip()
        if not images:
            raise ValueError("請選擇影像資料夾。")
        return discover_targets(
            self.test_v1, self._dataset_key(), Path(images).expanduser().resolve(),
            Path(masks).expanduser().resolve() if masks else None,
        )

    def _check_data(self) -> None:
        self._refresh_data_info()

    def _browse_images(self) -> None:
        selected = filedialog.askdirectory(title="選擇目標影像資料夾")
        if selected:
            self.image_dir_var.set(selected)
            self._refresh_data_info()

    def _browse_masks(self) -> None:
        selected = filedialog.askdirectory(title="選擇評估 mask 資料夾")
        if selected:
            self.mask_dir_var.set(selected)
            self._refresh_data_info()

    def _toggle_all_cases(self) -> None:
        self.case_box.configure(state="disabled" if self.all_cases_var.get() else "normal")

    def _set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self._settings_states.clear()
            pending = list(self.settings.winfo_children())
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if isinstance(widget, tk.Widget) and "state" in widget.keys():
                    self._settings_states.append((widget, str(widget.cget("state"))))
                    widget.configure({"state": "disabled"})
        else:
            for widget, state in self._settings_states:
                widget.configure({"state": state})
            self._settings_states.clear()
            self._toggle_all_cases()
        self.run_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _open_output(self) -> None:
        if self.current_output is not None and self.current_output.is_dir():
            os.startfile(str(self.current_output))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if self._running:
            return
        if not self.python.exists():
            messagebox.showerror("找不到 Python", f"專案環境不存在：\n{self.python}")
            return
        checkpoint = Path(self.checkpoint_var.get()).expanduser().resolve()
        if not checkpoint.is_file():
            messagebox.showerror("Checkpoint 無效", f"找不到 checkpoint：\n{checkpoint}")
            return
        try:
            max_cases = None if self.all_cases_var.get() else int(self.max_cases_var.get())
            seed = int(self.seed_var.get())
            if (max_cases is not None and max_cases < 1) or not 0 <= seed <= 4294967295:
                raise ValueError
        except ValueError:
            messagebox.showerror("設定無效", "影像上限須大於 0；Seed 須介於 0 與 4294967295。")
            return
        try:
            hyperparameters = parse_hyperparameters(self.hyperparameter_values)
        except ValueError as error:
            messagebox.showerror("超參數無效", str(error))
            return
        try:
            records = self._discover_data()
        except (OSError, ValueError) as error:
            messagebox.showerror("資料無效", str(error))
            return
        if not self.output_var.get().strip():
            messagebox.showerror("設定無效", "請選擇輸出目錄。")
            return
        selected = [name for name in METHODS if self.method_vars[name].get()]
        label = "source_only" if not selected else "_".join(name.lower() for name in selected)
        output_root = Path(self.output_var.get()).expanduser().resolve()
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = output_root / f"seed_{seed}_{label}_{run_stamp}"
        try:
            config = CompositionConfig(
                **hyperparameters, methods=tuple(selected), checkpoint=checkpoint, output=output,
                seed=seed, max_cases=max_cases, device=self.device_var.get(), dataset=self._dataset_key(),
                image_dir=Path(self.image_dir_var.get().strip()).expanduser().resolve(),
                mask_dir=Path(self.mask_dir_var.get().strip()).expanduser().resolve() if self.mask_dir_var.get().strip() else None,
            )
        except ValueError as error:
            messagebox.showerror("設定無效", str(error))
            return
        arguments = [
            "--config", str(output / "gui_config.json"), "--baseline-cache", str(output_root / "baselines"),
            "--methods", ",".join(selected), "--checkpoint", str(checkpoint),
            "--seed", str(seed),
            "--output", str(output), "--device", self.device_var.get(),
            "--dataset", self._dataset_key(),
            "--image-dir", str(Path(self.image_dir_var.get().strip()).expanduser().resolve()),
        ]
        if max_cases is not None:
            arguments.extend(["--max-cases", str(max_cases)])
        if self.mask_dir_var.get().strip():
            arguments.extend(["--mask-dir", str(Path(self.mask_dir_var.get().strip()).expanduser().resolve())])
        self.current_output = output
        self._completion = None
        self._read_error = None
        self.summary_var.set("準備全資料集 Source-only baseline；完成後執行本次實驗。")
        self._append_log(f"\n開始實驗｜{self.dataset_var.get()}｜方法：{', '.join(selected) or 'Source-only'}\n")
        self._append_log(f"影像：{self.image_dir_var.get()}\n結果目錄：{output}\n")
        total = len(records) if max_cases is None else min(len(records), max_cases)
        self.status_var.set(f"檢查 Source-only baseline…（全資料集 {len(records)} 張；本次 {total} 張）")
        self._set_running(True)
        self.progress.configure(value=0, maximum=len(records))
        try:
            self.process.start(arguments, output, config.to_dict())
        except (OSError, ValueError, RuntimeError) as error:
            self._finish(f"無法啟動 runner：{error}", failed=True)
            return
        self.open_output_button.configure(state="normal")

    def _handle_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            self._append_log(line)
            return
        if not isinstance(event, dict):
            self._append_log(line)
            return
        if event.get("event") == "baseline":
            total = int(event.get("total", 0))
            status = event.get("status")
            messages = {
                "checking": f"檢查 Source-only baseline…（全資料集 {total} 張）",
                "building": f"建立 Source-only baseline…（全資料集 {total} 張）",
                "reused": f"已重用 Source-only baseline（{total} 張）；準備本次實驗",
                "ready": f"Source-only baseline 就緒（{total} 張）；準備本次實驗",
            }
            if status in messages:
                self.progress.configure(value=total if status in ("reused", "ready") else 0, maximum=total)
                if not self.process.cancelled:
                    self.status_var.set(messages[status])
                self._append_log(messages[status] + "\n")
            else:
                self._append_log(line)
        elif event.get("event") == "baseline_progress":
            completed, total = int(event["completed"]), int(event["total"])
            self.progress.configure(value=completed, maximum=total)
            if not self.process.cancelled:
                self.status_var.set(f"Source-only baseline {completed}/{total}；完成後執行本次實驗")
            self._append_log(f"[Base {completed}/{total}] {event['image_id']}：已完成預測\n")
        elif event.get("event") == "progress":
            completed, total = int(event["completed"]), int(event["total"])
            phase = "Source-only" if event.get("phase") == "source_only" else "TTA 適配"
            self.progress.configure(value=completed, maximum=total)
            if not self.process.cancelled:
                self.status_var.set(f"{phase} {completed}/{total}；完成後評估")
            self._append_log(f"[{phase} {completed}/{total}] {event['image_id']}：已完成預測\n")
        elif event.get("event") == "status":
            message = str(event.get("message", ""))
            if not self.process.cancelled:
                self.status_var.set(message)
            self._append_log(message + "\n")
        elif event.get("event") == "completed":
            self._completion = event
        else:
            self._append_log(line)

    def _drain_output(self) -> None:
        try:
            while True:
                event = self.process.events.get_nowait()
                if event.kind == "line":
                    self._handle_line(str(event.payload))
                elif event.kind == "error":
                    self._read_error = str(event.payload)
                    self._append_log(f"讀取執行日誌失敗：{event.payload}\n")
                elif event.kind == "exit":
                    self._handle_exit(int(event.payload))
        except queue.Empty:
            pass
        self._poll_id = self.root.after(120, self._drain_output)

    def _handle_exit(self, code: int) -> None:
        if self.process.cancelled:
            self.summary_var.set("實驗已停止；已保存的預測保留在結果目錄，未產生完整評估。")
            self._finish("已停止")
            return
        if code != 0 or self._read_error:
            self.summary_var.set("實驗失敗；請查看下方錯誤日誌。")
            self._finish(f"失敗（exit {code}）", failed=True)
            return
        try:
            if self._completion is None and self.current_output is not None:
                self._completion = json.loads((self.current_output / "summary.json").read_text(encoding="utf-8"))
            if self._completion is None:
                raise ValueError("Runner 未回傳完成摘要。")
            text = format_result(self._completion)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.summary_var.set(f"結果摘要不完整：{error}")
            self._finish("結果摘要不完整", failed=True)
            return
        self.summary_var.set(text)
        self._append_log("\n" + text + "\n")
        self._persist_log(text)
        self.progress.configure(value=self.progress.cget("maximum"))
        self._finish("完成")

    def _persist_log(self, text: str) -> None:
        if self.current_output is not None and self.current_output.is_dir():
            try:
                with (self.current_output / "execution.log").open("a", encoding="utf-8") as stream:
                    stream.write(text + "\n")
            except OSError as error:
                self._append_log(f"無法保存最後摘要：{error}\n")

    def _finish(self, message: str, failed: bool = False) -> None:
        self._set_running(False)
        if failed:
            self.progress.configure(value=0)
        self.status_var.set(message)
        self._append_log(f"狀態：{message}\n")
        self._persist_log(f"狀態：{message}")
        if message.startswith("無法啟動"):
            messagebox.showerror("啟動失敗", message)

    def _stop(self) -> None:
        if self._running:
            self.process.cancel()
            self.stop_button.configure(state="disabled")
            self.status_var.set("正在停止…")

    def _close(self) -> None:
        self.root.after_cancel(self._poll_id)
        self.process.close()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    CompositionGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

