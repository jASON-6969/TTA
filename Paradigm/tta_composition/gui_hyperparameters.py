"""Hyperparameter fields, shared validation and the desktop editor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping
import tkinter as tk
from tkinter import messagebox, ttk

from .config import CompositionConfig
from .contracts import METHODS


@dataclass(frozen=True)
class HyperparameterField:
    name: str
    label: str
    value_type: type
    tab: str
    method: str | None = None
    choices: tuple[str, ...] = ()


FIELDS = (
    HyperparameterField("image_size", "影像大小（像素，16 的倍數）", int, "一般"),
    HyperparameterField("normalization", "影像正規化", str, "一般", choices=("zscore", "percentile")),
    HyperparameterField("adaptation_steps", "每張影像更新次數", int, "一般"),
    HyperparameterField("lr_VPTTA", "學習率", float, "VPTTA", method="VPTTA"),
    HyperparameterField("vptta_memory_size", "Prompt 記憶容量", int, "VPTTA"),
    HyperparameterField("vptta_prompt_size", "Prompt 大小", int, "VPTTA"),
    HyperparameterField("vptta_prompt_strength", "Prompt 強度", float, "VPTTA"),
    HyperparameterField("lr_DLTTA", "學習率", float, "DLTTA", method="DLTTA"),
    HyperparameterField("dltta_memory_size", "分布記憶容量", int, "DLTTA"),
    HyperparameterField("lr_TestFit", "學習率", float, "TestFit", method="TestFit"),
    HyperparameterField("testfit_confidence", "偽標籤信心門檻", float, "TestFit"),
    HyperparameterField("testfit_feature_weight", "特徵損失權重", float, "TestFit"),
    HyperparameterField("lr_GraTa", "學習率", float, "GraTa", method="GraTa"),
    HyperparameterField("grata_consistency_weight", "一致性損失權重", float, "GraTa"),
    HyperparameterField("grata_feature_weight", "特徵損失權重", float, "GraTa"),
    HyperparameterField("grata_noise_std", "增強雜訊標準差", float, "GraTa"),
    HyperparameterField("lr_SmaRT", "學習率", float, "SmaRT", method="SmaRT"),
    HyperparameterField("smart_ema_decay", "EMA 衰減係數", float, "SmaRT"),
    HyperparameterField("smart_structure_weight", "結構損失權重", float, "SmaRT"),
    HyperparameterField("smart_consistency_weight", "一致性損失權重", float, "SmaRT"),
)


def _field_value(values: Mapping[str, Any], field: HyperparameterField) -> Any:
    return values["lrs"][field.method] if field.method else values[field.name]


def _values_from_config(config: CompositionConfig) -> dict[str, Any]:
    payload = config.to_dict()
    return {
        "lrs": dict(config.lrs),
        **{field.name: payload[field.name] for field in FIELDS if field.method is None},
    }


def default_hyperparameters() -> dict[str, Any]:
    """Read defaults from the runner's configuration, never from UI constants."""
    return _values_from_config(CompositionConfig())


def parse_hyperparameters(values: Mapping[str, Any]) -> dict[str, Any]:
    """Parse UI values and delegate parameter ranges to CompositionConfig."""
    merged = default_hyperparameters()
    merged.update({name: value for name, value in values.items() if name != "lrs"})
    merged["lrs"].update(values.get("lrs", {}))
    parsed: dict[str, Any] = {"lrs": {}}
    for field in FIELDS:
        try:
            value = field.value_type(str(_field_value(merged, field)).strip())
            if field.value_type is float and not math.isfinite(value):
                raise ValueError("數值必須是有限值")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{field.tab}／{field.label}：請輸入有效數值或選項。") from error
        if field.method:
            parsed["lrs"][field.method] = value
        else:
            parsed[field.name] = value
    return _values_from_config(CompositionConfig(**parsed))


def format_hyperparameter_hint(values: Mapping[str, Any]) -> str:
    defaults = default_hyperparameters()
    changed = sum(_field_value(values, field) != _field_value(defaults, field) for field in FIELDS)
    suffix = "預設值" if not changed else f"已調整 {changed} 項"
    return f"{values['image_size']} px · {values['normalization']} · {suffix}"


class HyperparameterDialog(tk.Toplevel):
    """Edit a private copy; only Apply returns a validated configuration."""

    def __init__(self, parent: tk.Misc, values: Mapping[str, Any]):
        super().__init__(parent)
        self.title("超參數設定")
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self.result: dict[str, Any] | None = None
        self.variables: dict[str, tk.StringVar] = {}
        shell = ttk.Frame(self, padding=14)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="設定會套用到下一次實驗；未勾選方法的設定會保留。").pack(anchor="w", pady=(0, 10))
        self.notebook = ttk.Notebook(shell, width=500, height=250)
        self.notebook.pack(fill="both", expand=True)
        for tab in ("一般", *METHODS):
            frame = ttk.Frame(self.notebook, padding=14)
            frame.columnconfigure(1, weight=1)
            self.notebook.add(frame, text=tab)
            for row, field in enumerate(field for field in FIELDS if field.tab == tab):
                variable = tk.StringVar(self, value=str(_field_value(values, field)))
                self.variables[field.name] = variable
                ttk.Label(frame, text=field.label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=7)
                widget: ttk.Entry | ttk.Combobox
                if field.choices:
                    widget = ttk.Combobox(frame, textvariable=variable, values=field.choices, state="readonly", width=18)
                else:
                    widget = ttk.Entry(frame, textvariable=variable, width=20)
                widget.grid(row=row, column=1, sticky="ew", pady=7)
        controls = ttk.Frame(shell)
        controls.pack(fill="x", pady=(12, 0))
        ttk.Button(controls, text="恢復預設", command=self._restore_defaults).pack(side="left")
        ttk.Button(controls, text="套用", command=self._apply).pack(side="right", padx=(8, 0))
        ttk.Button(controls, text="取消", command=self._cancel).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _event: self._cancel())

    def _read_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {"lrs": {}}
        for field in FIELDS:
            value = self.variables[field.name].get()
            if field.method:
                values["lrs"][field.method] = value
            else:
                values[field.name] = value
        return values

    def _restore_defaults(self) -> None:
        defaults = default_hyperparameters()
        for field in FIELDS:
            self.variables[field.name].set(str(_field_value(defaults, field)))

    def _apply(self) -> None:
        try:
            self.result = parse_hyperparameters(self._read_values())
        except ValueError as error:
            messagebox.showerror("超參數無效", str(error), parent=self)
            return
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


def show_hyperparameter_dialog(parent: tk.Misc, values: Mapping[str, Any]) -> dict[str, Any] | None:
    dialog = HyperparameterDialog(parent, values)
    dialog.grab_set()
    parent.wait_window(dialog)
    return dialog.result
