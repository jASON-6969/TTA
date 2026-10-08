"""Central proposal/commit engine for arbitrary TTA method subsets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math
import time
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from .audit import finite_gradient_norm
from .config import CompositionConfig
from .contracts import StepContext, UpdateProposal, UpdateRecord
from .lr_policy import LearningRateDecision, combine_learning_rates
from .model_bridge import ModelBundle, bn_parameter_names, checked_probabilities, model_forward, non_bn_parameter_names, set_student_modes, source_buffers_snapshot
from .parameter_registry import ParameterRegistry
from .methods import DLTTA, GraTa, SmaRT, TestFit, VPTTA


class CompositionEngine:
    """Own model state, method state and the single update transaction."""

    def __init__(self, bundle: ModelBundle, config: CompositionConfig):
        self.bundle = bundle
        self.config = config
        self.methods = config.canonical_methods
        self.step_index = 0
        self._initial_student = deepcopy(bundle.student.state_dict())
        self._initial_smart = deepcopy(bundle.smart.state_dict())
        self._initial_ema = deepcopy(bundle.ema.state_dict()) if bundle.ema is not None else None

        prompt_device = bundle.student.device if hasattr(bundle.student, "device") else next(bundle.student.parameters()).device
        self.vptta = VPTTA(
            prompt_device, config.lrs["VPTTA"], config.vptta_memory_size,
            prompt_size=config.vptta_prompt_size, prompt_strength=config.vptta_prompt_strength,
        )
        self.dltta = DLTTA(config.lrs["DLTTA"], config.dltta_memory_size)
        self._initialize_signal_methods()
        self._method_objects = {
            "VPTTA": self.vptta,
            "DLTTA": self.dltta,
            "TestFit": self.testfit,
            "GraTa": self.grata,
            "SmaRT": self.smart,
        }
        extras = {"vptta.prompt": self.vptta.prompt}
        self.registry = ParameterRegistry(bundle.student, bundle.smart, extras=extras)
        self._configure_ownership()
        self._source_buffers = source_buffers_snapshot(bundle.reference)
        self._student_source_buffers = source_buffers_snapshot(bundle.student)

    def _initialize_signal_methods(self) -> None:
        self.testfit = TestFit(
            self.config.lrs["TestFit"], self.config.testfit_confidence,
            feature_weight=self.config.testfit_feature_weight,
        )
        self.grata = GraTa(
            self.config.lrs["GraTa"], consistency_weight=self.config.grata_consistency_weight,
            feature_weight=self.config.grata_feature_weight,
        )
        self.smart = SmaRT(
            self.bundle.smart, self.config.lrs["SmaRT"], structure_weight=self.config.smart_structure_weight,
            consistency_weight=self.config.smart_consistency_weight,
        )

    def _configure_ownership(self) -> None:
        model_non_bn = non_bn_parameter_names(self.bundle.student)
        model_bn = bn_parameter_names(self.bundle.student)
        if "VPTTA" in self.methods:
            self.registry.claim("VPTTA", self.vptta.parameter_names())
        if "SmaRT" in self.methods:
            self.registry.claim("SmaRT", self.smart.parameter_names())
        if "TestFit" in self.methods:
            self.registry.claim("TestFit", model_non_bn)
            self.testfit.bind_parameter_names(model_non_bn)
        if "GraTa" in self.methods:
            self.registry.claim("GraTa", model_bn)
            self.grata.bind_parameter_names(model_bn)
        elif "DLTTA" in self.methods and "TestFit" not in self.methods:
            # DLTTA needs a host only when it is the sole model adapter.
            self.registry.claim("DLTTA", model_bn)
            self.dltta.bind_host(model_bn)
        if "DLTTA" in self.methods and "GraTa" not in self.methods and "TestFit" in self.methods:
            # TestFit owns the convolutional group; DLTTA remains an LR controller.
            self.dltta.bind_host(())
        for name, parameter in self.registry.parameters.items():
            parameter.requires_grad = name in self.registry.owners
        set_student_modes(self.bundle)

    @property
    def parameter_ownership(self) -> dict[str, str]:
        return self.registry.owners

    def _active(self, name: str) -> bool:
        return name in self.methods

    def _make_context(self, image: Tensor, sample_id: str) -> StepContext:
        device = next(self.bundle.student.parameters()).device
        image = image.to(device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        transformed = image
        if self._active("VPTTA"):
            transformed = self.vptta.prepare(image, self.config.seed + self.step_index)
        if self._active("VPTTA"):
            # Reference weights remain frozen, but the prompt must receive the
            # adaptation signal through this forward path.
            reference_logits, reference_features = model_forward(self.bundle.reference, transformed)
        else:
            with torch.no_grad():
                reference_logits, reference_features = model_forward(self.bundle.reference, transformed.detach())
        set_student_modes(self.bundle)
        student_logits, student_features = model_forward(self.bundle.student, transformed)
        smart_logits = smart_features = ema_logits = ema_features = None
        if self._active("SmaRT"):
            smart_view = self.bundle.smart.augment(transformed, self.config.seed + 200000 + self.step_index)
            smart_image = self.bundle.smart.transform(smart_view)
            smart_logits, smart_features = model_forward(self.bundle.student, smart_image)
            smart_logits = self.bundle.smart.correct(smart_logits)
            if self.bundle.ema is not None:
                with torch.no_grad():
                    ema_logits, ema_features = model_forward(self.bundle.ema, smart_image.detach())
        augmented_logits = augmented_features = None
        if self._active("GraTa"):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.seed + 100000 + self.step_index)
            noise = torch.randn(transformed.shape, generator=generator, dtype=transformed.dtype).to(device) * self.config.grata_noise_std
            noisy = transformed + noise
            augmented_logits, augmented_features = model_forward(self.bundle.student, noisy)
        return StepContext(
            step=self.step_index + 1,
            sample_id=sample_id,
            image=image,
            transformed_image=transformed,
            reference_logits=reference_logits,
            reference_features=reference_features,
            student_logits=student_logits,
            student_features=student_features,
            smart_logits=smart_logits,
            smart_features=smart_features,
            ema_logits=ema_logits,
            ema_features=ema_features,
            augmented_logits=augmented_logits,
            augmented_features=augmented_features,
            rng_seed=self.config.seed + self.step_index,
        )

    def _proposals(self, context: StepContext) -> list[UpdateProposal]:
        proposals: list[UpdateProposal] = []
        parameters = self.registry.parameters
        for name in self.methods:
            proposal = self._method_objects[name].propose(context, parameters)
            proposal.validate()
            proposals.append(proposal)
        return proposals

    def _decision_for_owner(self, owner: str, proposals: list[UpdateProposal]) -> LearningRateDecision:
        fallback = self.config.lrs[owner]
        hints = {
            proposal.method: proposal.lr_hint
            for proposal in proposals
            if proposal.lr_hint is not None
            and (proposal.method == owner or (proposal.method == "DLTTA" and owner != "VPTTA"))
        }
        return combine_learning_rates(owner, hints, fallback)

    def _commit(self, context: StepContext, proposals: list[UpdateProposal], started: float) -> UpdateRecord:
        by_name: dict[str, Tensor] = {}
        losses: dict[str, float] = {}
        hints: dict[str, float] = {}
        for proposal in proposals:
            if proposal.loss is not None:
                losses[proposal.method] = proposal.loss
            if proposal.lr_hint is not None:
                hints[proposal.method] = proposal.lr_hint
            for name, gradient in proposal.gradients.items():
                if name in by_name:
                    raise ValueError(f"Multiple proposals attempted to update {name}")
                by_name[name] = gradient

        decisions: dict[str, LearningRateDecision] = {}
        for owner in sorted(set(self.registry.owners.values())):
            decisions[owner] = self._decision_for_owner(owner, proposals)

        # Reject non-serializable audit values before changing any parameter.
        gradient_norms = {name: finite_gradient_norm(gradient, name) for name, gradient in by_name.items()}
        final_lrs: dict[str, float] = {}
        updated = 0
        with torch.no_grad():
            for name, gradient in by_name.items():
                owner = self.registry.owners[name]
                decision = decisions[owner]
                final_lrs[owner] = decision.final_lr
                if decision.skipped:
                    continue
                parameter = self.registry.parameter(name)
                parameter.add_(gradient, alpha=-decision.final_lr)
                updated += 1
        for method in self.methods:
            self._method_objects[method].after_commit(context)
        if self._active("SmaRT") and self.bundle.ema is not None:
            with torch.no_grad():
                for ema_parameter, student_parameter in zip(self.bundle.ema.parameters(), self.bundle.student.parameters()):
                    ema_parameter.mul_(self.config.smart_ema_decay).add_(student_parameter, alpha=1.0 - self.config.smart_ema_decay)
        self.step_index += 1
        return UpdateRecord(
            step=self.step_index,
            losses=losses,
            lr_hints=hints,
            final_learning_rates=final_lrs,
            gradient_norms=gradient_norms,
            updated_parameters=updated,
            elapsed_seconds=time.perf_counter() - started,
        )

    def step(self, image: Tensor, sample_id: str) -> tuple[Tensor, UpdateRecord, dict[str, float]]:
        """Adapt repeatedly, then publish one prediction and one prompt memory entry."""
        started = time.perf_counter()
        records: list[UpdateRecord] = []
        for _ in range(self.config.adaptation_steps):
            iteration_started = time.perf_counter()
            context = self._make_context(image, sample_id)
            proposals = self._proposals(context)
            records.append(self._commit(context, proposals, iteration_started))

        prediction = self._predict(context.image)
        for method in self.methods:
            self._method_objects[method].after_prediction(context)
        metrics: dict[str, float] = {}
        for proposal in proposals:
            metrics.update({f"{proposal.method}.{key}": value for key, value in proposal.metrics.items()})
        metrics["adaptation_iterations"] = float(self.config.adaptation_steps)
        record = replace(
            records[-1], iterations=tuple(records),
            elapsed_seconds=time.perf_counter() - started,
        )
        return prediction, record, metrics

    @staticmethod
    def _checked_probabilities(logits: Tensor, model_name: str) -> Tensor:
        return checked_probabilities(logits, model_name)

    def _predict(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            transformed = image
            if self._active("VPTTA"):
                transformed = self.vptta.prepare(image, self.config.seed + self.step_index)
            student_logits, _ = model_forward(self.bundle.student, transformed)
            reference_logits, _ = model_forward(self.bundle.reference, transformed.detach())
            if self._active("SmaRT"):
                smart_view = self.bundle.smart.augment(transformed, self.config.seed + 200000 + self.step_index)
                smart_image = self.bundle.smart.transform(smart_view)
                student_logits, _ = model_forward(self.bundle.student, smart_image)
                student_logits = self.bundle.smart.correct(student_logits)
            student_probabilities = self._checked_probabilities(student_logits, "student")
            if self._active("TestFit"):
                reference_probabilities = self._checked_probabilities(reference_logits, "reference")
                probabilities = 0.5 * student_probabilities + 0.5 * reference_probabilities
            else:
                probabilities = student_probabilities
            return probabilities.argmax(dim=1).cpu()

    def reset(self) -> None:
        self.bundle.student.load_state_dict(self._initial_student)
        self.bundle.smart.load_state_dict(self._initial_smart)
        if self.bundle.ema is not None and self._initial_ema is not None:
            self.bundle.ema.load_state_dict(self._initial_ema)
        self.vptta.reset()
        self.dltta.reset()
        self._initialize_signal_methods()
        self._method_objects.update({"TestFit": self.testfit, "GraTa": self.grata, "SmaRT": self.smart})
        self.step_index = 0
        self._configure_ownership()

    def audit_state(self) -> dict[str, object]:
        current_buffers = source_buffers_snapshot(self.bundle.reference)
        source_unchanged = all(torch.equal(current_buffers[name], value) for name, value in self._source_buffers.items())
        current_student_buffers = source_buffers_snapshot(self.bundle.student)
        student_source_unchanged = all(
            torch.equal(current_student_buffers[name], value)
            for name, value in self._student_source_buffers.items()
        )
        return {
            "methods": list(self.methods),
            "parameter_ownership": self.parameter_ownership,
            "reference_source_buffers_unchanged": source_unchanged,
            "student_source_bn_buffers_unchanged": student_source_unchanged,
            "step": self.step_index,
            "adaptation_steps": self.config.adaptation_steps,
            "vptta_memory": len(self.vptta.memory),
            "dltta_memory": len(self.dltta.memory),
            "has_smart_ema": self.bundle.ema is not None,
        }

