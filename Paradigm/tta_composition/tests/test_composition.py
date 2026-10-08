from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image

TEST_V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_V1))
sys.path.insert(0, str(TEST_V1 / "code"))

from feature_alignment_2d_pilot.data import ImageOnlyDataset, Record
from Paradigm.tta_composition.config import CompositionConfig
from Paradigm.tta_composition.engine import CompositionEngine
from Paradigm.tta_composition.lr_policy import combine_learning_rates
from Paradigm.tta_composition.methods.smart import count_foreground_components, two_lung_structure_loss
from Paradigm.tta_composition.model_bridge import ModelBundle, SmartAdapter, bn_parameter_names, non_bn_parameter_names


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, image, return_features=False):
        features = self.bn(self.conv(image))
        logits = self.head(features)
        return (logits, features) if return_features else logits


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def make_engine(methods=(), *, adaptation_steps=3):
    torch.manual_seed(7)
    reference = TinyUNet()
    student = TinyUNet()
    student.load_state_dict(reference.state_dict())
    bundle = ModelBundle(reference, student, SmartAdapter(), None, None, freeze_bn)
    return CompositionEngine(bundle, CompositionConfig(
        methods=tuple(methods), image_size=16, device="cpu", adaptation_steps=adaptation_steps,
    ))


def test_all_32_subsets_can_step_and_reset():
    names = ("VPTTA", "DLTTA", "TestFit", "GraTa", "SmaRT")
    for adaptation_steps in (1, 3, 5):
        for mask in range(32):
            selected = tuple(name for index, name in enumerate(names) if mask & (1 << index))
            engine = make_engine(selected, adaptation_steps=adaptation_steps)
            prediction, record, _ = engine.step(torch.randn(1, 16, 16), "sample")
            assert prediction.shape == (1, 16, 16)
            assert record.step == adaptation_steps
            assert [iteration.step for iteration in record.iterations] == list(range(1, adaptation_steps + 1))
            engine.reset()
            assert engine.step_index == 0


def test_source_only_rejects_nonfinite_model_outputs():
    for value in (float("nan"), float("inf"), -float("inf")):
        engine = make_engine()
        with torch.no_grad():
            engine.bundle.student.head.bias[0] = value
            engine.bundle.reference.head.bias[0] = value
        with unittest.TestCase().assertRaisesRegex(FloatingPointError, "Non-finite student logits"):
            engine.step(torch.randn(1, 16, 16), "invalid-checkpoint")


def test_prediction_rejects_nonfinite_logits_after_adaptation():
    engine = make_engine(("GraTa",))
    commit = engine._commit

    def overflow_after_commit(*args, **kwargs):
        record = commit(*args, **kwargs)
        if engine.step_index == engine.config.adaptation_steps:
            with torch.no_grad():
                engine.bundle.student.head.bias.fill_(float("inf"))
        return record

    with patch.object(engine, "_commit", overflow_after_commit), \
            unittest.TestCase().assertRaisesRegex(FloatingPointError, "Non-finite student logits"):
        engine.step(torch.randn(1, 16, 16), "overflow-after-update")


def test_testfit_rejects_nonfinite_reference_outputs():
    engine = make_engine(("TestFit",))
    commit = engine._commit

    def corrupt_reference_after_commit(*args, **kwargs):
        record = commit(*args, **kwargs)
        if engine.step_index == engine.config.adaptation_steps:
            with torch.no_grad():
                engine.bundle.reference.head.bias.fill_(float("nan"))
        return record

    with patch.object(engine, "_commit", corrupt_reference_after_commit), \
            unittest.TestCase().assertRaisesRegex(FloatingPointError, "Non-finite reference logits"):
        engine.step(torch.randn(1, 16, 16), "invalid-reference")


def test_parameter_ownership_is_unique_and_source_buffers_are_unchanged():
    engine = make_engine(("VPTTA", "DLTTA", "TestFit", "GraTa", "SmaRT"))
    owners = engine.parameter_ownership
    engine.registry.validate_complete()
    assert set(engine.registry.names_for("VPTTA")) == {"vptta.prompt"}
    assert set(engine.registry.names_for("TestFit")) == set(non_bn_parameter_names(engine.bundle.student))
    assert set(engine.registry.names_for("GraTa")) == set(bn_parameter_names(engine.bundle.student))
    assert set(engine.registry.names_for("SmaRT")) == {
        f"smart.{name}" for name, _ in engine.bundle.smart.named_parameters()
    }
    assert engine.registry.names_for("DLTTA") == ()
    groups = [set(engine.registry.names_for(method)) for method in ("VPTTA", "TestFit", "GraTa", "SmaRT")]
    assert sum(len(group) for group in groups) == len(owners)
    assert all(left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :])
    for name, parameter in engine.registry.parameters.items():
        assert parameter.requires_grad is (name in owners)
    engine.step(torch.randn(1, 16, 16), "sample")
    assert engine.audit_state()["reference_source_buffers_unchanged"] is True
    assert engine.audit_state()["student_source_bn_buffers_unchanged"] is True


def test_dltta_uses_bn_host_when_it_is_the_only_model_adapter():
    engine = make_engine(("DLTTA",))
    assert set(engine.registry.names_for("DLTTA")) == set(bn_parameter_names(engine.bundle.student))
    assert engine.registry.names_for("TestFit") == ()
    assert engine.registry.names_for("GraTa") == ()


def test_target_adaptation_dataset_does_not_load_mask(tmp_path):
    image_path = tmp_path / "target.png"
    Image.new("L", (8, 8), color=127).save(image_path)
    record = Record("target-0", image_path, tmp_path / "missing-mask.png", "Montgomery")

    image, sample_id = ImageOnlyDataset([record], image_size=16, normalization="zscore")[0]

    assert image.shape == (1, 16, 16)
    assert image.isfinite().all()
    assert sample_id == "target-0"


def test_method_order_is_canonical_and_reset_reproduces_prediction():
    image = torch.randn(1, 16, 16)
    first = make_engine(("SmaRT", "GraTa", "VPTTA", "DLTTA", "TestFit"))
    prediction_a, _, _ = first.step(image, "sample")
    first.reset()
    prediction_b, _, _ = first.step(image, "sample")
    assert first.methods == ("VPTTA", "DLTTA", "TestFit", "GraTa", "SmaRT")
    assert torch.equal(prediction_a, prediction_b)


def test_vptta_prompt_receives_reference_signal():
    engine = make_engine(("VPTTA",))
    engine.step(torch.randn(1, 16, 16), "sample")
    assert engine.vptta.memory
    assert torch.isfinite(engine.vptta.prompt).all()
    assert not torch.equal(engine.vptta.prompt, torch.zeros_like(engine.vptta.prompt))


def test_lr_hints_use_geometric_mean_and_zero_skips():
    decision = combine_learning_rates("GraTa", {"GraTa": 1e-4, "DLTTA": 1e-2}, 1e-3)
    assert abs(decision.final_lr - 1e-3) < 1e-10
    assert combine_learning_rates("GraTa", {"GraTa": 0.0, "DLTTA": 1e-4}, 1e-3).skipped


def test_two_lung_structure_allows_zero_one_or_two_components():
    empty = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    one = empty.clone()
    one[:, :, 1:3, 1:3] = True
    two = one.clone()
    two[:, :, 5:7, 5:7] = True
    assert count_foreground_components(empty) == 0
    assert count_foreground_components(one) == 1
    assert count_foreground_components(two) == 2
    loss, metrics = two_lung_structure_loss(torch.cat([~empty, empty], dim=1).float())
    assert torch.isfinite(loss)
    assert metrics["foreground_components"] == 0

