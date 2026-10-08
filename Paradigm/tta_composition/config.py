"""Named configuration for the composition runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
from typing import Iterable

from .contracts import METHODS
from .data_selection import DATASETS


DEFAULT_LRS = {"VPTTA": 5e-3, "DLTTA": 5e-4, "TestFit": 5e-4, "GraTa": 5e-4, "SmaRT": 5e-4}
# Experimental defaults; historical run settings are recorded in each config.json.


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(name: str, value: object, *, minimum: float = 0.0, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
        finite = math.isfinite(numeric)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")
    if numeric < minimum or (maximum is not None and numeric > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {interval}")


@dataclass
class CompositionConfig:
    methods: tuple[str, ...] = ()
    checkpoint: Path | None = None
    output: Path | None = None
    seed: int = 42
    image_size: int = 256
    normalization: str = "zscore"
    max_cases: int | None = None
    device: str = "auto"
    dataset: str = "montgomery"
    image_dir: Path | None = None
    mask_dir: Path | None = None
    baseline_cache: Path | None = None
    lrs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LRS))
    smart_ema_decay: float = 0.99
    vptta_memory_size: int = 16
    vptta_prompt_size: int = 8
    vptta_prompt_strength: float = 0.05
    dltta_memory_size: int = 16
    testfit_confidence: float = 0.95
    testfit_feature_weight: float = 0.50
    grata_consistency_weight: float = 1.0
    grata_feature_weight: float = 0.50
    grata_noise_std: float = 0.08
    smart_structure_weight: float = 0.50
    smart_consistency_weight: float = 0.50
    adaptation_steps: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.lrs, dict):
            raise ValueError("lrs must be a method-to-learning-rate dictionary")
        if any(method not in METHODS for method in self.lrs):
            raise ValueError(f"Learning rates must use method names from {METHODS}")
        for method, default in DEFAULT_LRS.items():
            self.lrs.setdefault(method, default)
        selected = tuple(self.methods)
        unknown = sorted(set(selected) - set(METHODS))
        if unknown:
            raise ValueError(f"Unknown method(s): {unknown}; choices={METHODS}")
        if len(set(selected)) != len(selected):
            raise ValueError("Methods must be unique")
        _positive_integer("image_size", self.image_size)
        if self.image_size % 16:
            raise ValueError("image_size must be a multiple of 16 for the five-level UNet")
        if self.normalization not in ("zscore", "percentile"):
            raise ValueError("normalization must be zscore or percentile")
        if self.max_cases is not None:
            _positive_integer("max_cases", self.max_cases)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 4294967295:
            raise ValueError("seed must be an integer in [0, 4294967295]")
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{self.dataset}'; choices={DATASETS}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        _finite_number("smart_ema_decay", self.smart_ema_decay, maximum=1.0)
        if self.smart_ema_decay == 0:
            raise ValueError("smart_ema_decay must be in (0, 1]")
        for method in METHODS:
            _finite_number(f"Learning rate for {method}", self.lrs[method])
        for name in ("vptta_memory_size", "vptta_prompt_size", "dltta_memory_size", "adaptation_steps"):
            _positive_integer(name, getattr(self, name))
        _finite_number("testfit_confidence", self.testfit_confidence, maximum=1.0)
        for name in (
            "vptta_prompt_strength", "testfit_feature_weight", "grata_consistency_weight",
            "grata_feature_weight", "grata_noise_std", "smart_structure_weight", "smart_consistency_weight",
        ):
            _finite_number(name, getattr(self, name))

    @property
    def canonical_methods(self) -> tuple[str, ...]:
        selected = set(self.methods)
        return tuple(method for method in METHODS if method in selected)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["methods"] = list(self.methods)
        payload["checkpoint"] = str(self.checkpoint) if self.checkpoint else None
        payload["output"] = str(self.output) if self.output else None
        payload["image_dir"] = str(self.image_dir) if self.image_dir else None
        payload["mask_dir"] = str(self.mask_dir) if self.mask_dir else None
        payload["baseline_cache"] = str(self.baseline_cache) if self.baseline_cache else None
        return payload

    @classmethod
    def from_json(cls, path: Path) -> "CompositionConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "methods" in payload:
            payload["methods"] = tuple(payload["methods"])
        for key in ("checkpoint", "output", "image_dir", "mask_dir", "baseline_cache"):
            if key in payload:
                payload[key] = Path(payload[key]) if payload[key] else None
        return cls(**payload)


def parse_methods(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        if not value.strip():
            return ()
        values = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        values = tuple(value)
    aliases = {"Smart": "SmaRT", "smart": "SmaRT", "grata": "GraTa", "testfit": "TestFit", "dltta": "DLTTA", "vptta": "VPTTA"}
    values = tuple(aliases.get(value, value) for value in values)
    unknown = sorted(set(values) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}; choices={METHODS}")
    if len(set(values)) != len(values):
        raise ValueError("Methods must be unique")
    return values

