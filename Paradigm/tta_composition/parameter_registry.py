"""Parameter ownership validation for composition experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

from torch import nn


@dataclass(frozen=True)
class ParameterOwner:
    method: str
    names: tuple[str, ...]


class ParameterRegistry:
    """Maps every trainable parameter to at most one adapting method."""

    def __init__(
        self,
        model: nn.Module,
        smart_module: nn.Module | None = None,
        extras: dict[str, nn.Parameter] | None = None,
    ):
        self.model = model
        self.smart_module = smart_module
        self.parameters: dict[str, nn.Parameter] = {
            name: parameter for name, parameter in model.named_parameters()
        }
        if smart_module is not None:
            self.parameters.update(
                {f"smart.{name}": parameter for name, parameter in smart_module.named_parameters()}
            )
        if extras:
            self.parameters.update(extras)
        self._owners: dict[str, str] = {}

    @property
    def owners(self) -> dict[str, str]:
        return dict(self._owners)

    def names_for(self, method: str) -> tuple[str, ...]:
        return tuple(name for name, owner in self._owners.items() if owner == method)

    def claim(self, method: str, names: Iterable[str]) -> ParameterOwner:
        names = tuple(names)
        unknown = sorted(set(names) - set(self.parameters))
        if unknown:
            raise KeyError(f"{method} claimed unknown parameters: {unknown}")
        collisions = sorted(name for name in names if name in self._owners and self._owners[name] != method)
        if collisions:
            owners = {name: self._owners[name] for name in collisions}
            raise ValueError(f"Parameter ownership collision: {owners}; new owner={method}")
        for name in names:
            self._owners[name] = method
        return ParameterOwner(method, names)

    def validate_complete(self) -> None:
        duplicates = [name for name, count in self._counts().items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate parameter owners: {duplicates}")

    def _counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for name in self._owners:
            counts[name] += 1
        return counts

    def parameter(self, name: str) -> nn.Parameter:
        return self.parameters[name]

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "owner": self._owners.get(name),
                "shape": list(parameter.shape),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in self.parameters.items()
        }

