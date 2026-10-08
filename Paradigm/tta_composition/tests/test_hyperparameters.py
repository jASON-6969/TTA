"""Configuration and adaptation regressions, using small CPU-only models."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

TEST_V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_V1))

from Paradigm.tta_composition.config import CompositionConfig, DEFAULT_LRS
from Paradigm.tta_composition.contracts import METHODS, StepContext
from Paradigm.tta_composition.engine import CompositionEngine
from Paradigm.tta_composition.methods.grata import GraTa
from Paradigm.tta_composition.methods.smart import SmaRT, two_lung_structure_loss
from Paradigm.tta_composition.methods.testfit import TestFit
from Paradigm.tta_composition.methods.vptta import VPTTA
from Paradigm.tta_composition.model_bridge import ModelBundle, SmartAdapter


def configured(**overrides) -> CompositionConfig:
    values = dict(
        methods=METHODS, image_size=32, normalization="percentile", seed=17,
        lrs={name: rate * 2 for name, rate in DEFAULT_LRS.items()},
        smart_ema_decay=0.8, vptta_memory_size=2, vptta_prompt_size=4,
        vptta_prompt_strength=0.15, dltta_memory_size=3, testfit_confidence=0.65,
        testfit_feature_weight=0.3, grata_consistency_weight=0.7,
        grata_feature_weight=0.2, grata_noise_std=0.15,
        smart_structure_weight=0.4, smart_consistency_weight=0.5,
        adaptation_steps=5,
    )
    values.update(overrides)
    return CompositionConfig(**values)


class TinyModel(nn.Module):
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


def make_engine(config: CompositionConfig) -> CompositionEngine:
    torch.manual_seed(7)
    reference = TinyModel().eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    bundle = ModelBundle(
        reference, deepcopy(reference), SmartAdapter(), deepcopy(reference), None, freeze_bn,
    )
    return CompositionEngine(bundle, config)


def loss_context(parameter: nn.Parameter) -> StepContext:
    pattern = torch.linspace(-0.7, 1.3, 16).reshape(1, 1, 4, 4)
    logits = torch.cat((parameter * pattern, -parameter * pattern), dim=1)
    teacher = torch.cat((torch.full_like(pattern, -2), torch.full_like(pattern, 2)), dim=1)
    features = parameter * pattern
    return StepContext(
        step=1, sample_id="loss-probe", image=pattern, transformed_image=pattern,
        reference_logits=teacher, reference_features=torch.zeros_like(features),
        student_logits=logits, student_features=features,
        augmented_logits=logits + torch.cat((pattern, -pattern), dim=1),
        augmented_features=features * 2, smart_logits=logits, ema_logits=teacher,
    )


class ConfigurationTests(unittest.TestCase):
    def test_all_nondefault_settings_round_trip_through_json(self):
        config = configured(
            checkpoint=Path("model.pth"), output=Path("results"), image_dir=Path("images"),
            mask_dir=Path("masks"), baseline_cache=Path("baseline-cache"), max_cases=5,
        )
        with tempfile.TemporaryDirectory(prefix="tta_config_test_") as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            loaded = CompositionConfig.from_json(path)
        self.assertEqual(loaded.to_dict(), config.to_dict())
        self.assertEqual(loaded.baseline_cache, Path("baseline-cache"))
        self.assertEqual(loaded.canonical_methods, METHODS)

    def test_partial_learning_rates_keep_other_defaults(self):
        config = CompositionConfig(lrs={"TestFit": 0.0})
        self.assertEqual(config.lrs, {**DEFAULT_LRS, "TestFit": 0.0})
        with self.assertRaises(ValueError):
            CompositionConfig(lrs={"Testfit": 0.1})

    def test_rejects_nonfinite_negative_or_nonnumeric_rates_and_weights(self):
        fields = (
            "vptta_prompt_strength", "testfit_feature_weight", "grata_consistency_weight",
            "grata_feature_weight", "grata_noise_std", "smart_structure_weight",
            "smart_consistency_weight", "testfit_confidence", "smart_ema_decay",
        )
        for field in fields:
            for value in (float("nan"), float("inf"), -float("inf"), -0.1, True, "0.1"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    CompositionConfig(**{field: value})
        for method in METHODS:
            for value in (float("nan"), float("inf"), -float("inf"), -0.1, True, "0.1"):
                with self.subTest(method=method, value=value), self.assertRaises(ValueError):
                    CompositionConfig(lrs={method: value})

    def test_rejects_invalid_integer_and_bounded_settings(self):
        for field in ("image_size", "max_cases", "vptta_memory_size", "vptta_prompt_size", "dltta_memory_size", "adaptation_steps"):
            for value in (0, -1, 1.5, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    CompositionConfig(**{field: value})
        for field, value in (
            ("image_size", 17), ("normalization", "unknown"), ("testfit_confidence", 1.1),
            ("smart_ema_decay", 0), ("smart_ema_decay", 1.1),
            ("seed", -1), ("seed", 4294967296), ("seed", True), ("seed", 1.5),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                CompositionConfig(**{field: value})
        for confidence in (0.0, 1.0):
            CompositionConfig(testfit_confidence=confidence, smart_ema_decay=1.0, seed=0)
        CompositionConfig(seed=4294967295, image_size=16)


class AdaptationSettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_prompt_size_strength_and_memory_change_real_transformation(self):
        method = VPTTA(torch.device("cpu"), memory_size=2, prompt_size=4, prompt_strength=0.15)
        image = torch.zeros(1, 1, 16, 16)
        with torch.no_grad():
            method.prompt.fill_(1)
        transformed = method.prepare(image, 0)
        self.assertEqual(method.prompt.shape, (1, 1, 4, 4))
        self.assertTrue(torch.allclose(transformed, image + 0.15 * torch.tanh(torch.ones_like(image))))
        for _ in range(4):
            method.after_prediction(None)
        self.assertEqual(len(method.memory), 2)
        method.reset()
        self.assertEqual(method.prompt_strength, 0.15)
        self.assertEqual(len(method.memory), 0)
        self.assertTrue(torch.equal(method.prepare(image, 0), image))

    def test_testfit_threshold_and_feature_weight_change_loss_and_gradient(self):
        parameter = nn.Parameter(torch.tensor(0.7))
        context = loss_context(parameter)
        proposals = []
        for confidence, feature_weight in ((1.0, 0.0), (1.0, 0.3), (0.5, 0.3)):
            method = TestFit(confidence=confidence, feature_weight=feature_weight)
            method.bind_parameter_names(("value",))
            proposals.append(method.propose(context, {"value": parameter}))
        self.assertEqual(proposals[0].loss, 0.0)
        self.assertGreater(proposals[1].loss, 0.0)
        self.assertGreater(proposals[2].loss, proposals[1].loss)
        self.assertEqual(proposals[0].metrics["pseudo_valid_fraction"], 0.0)
        self.assertEqual(proposals[2].metrics["pseudo_valid_fraction"], 1.0)
        self.assertGreater(float(proposals[1].gradients["value"].abs()), 0.0)

    def test_grata_weights_change_loss_and_gradient(self):
        parameter = nn.Parameter(torch.tensor(0.7))
        context = loss_context(parameter)
        proposals = []
        for consistency_weight, feature_weight in ((0.0, 0.0), (0.7, 0.0), (0.0, 0.2)):
            method = GraTa(consistency_weight=consistency_weight, feature_weight=feature_weight)
            method.bind_parameter_names(("value",))
            proposals.append(method.propose(context, {"value": parameter}))
        self.assertGreater(proposals[1].loss, proposals[0].loss)
        self.assertGreater(proposals[2].loss, proposals[0].loss)
        self.assertNotEqual(float(proposals[1].gradients["value"]), float(proposals[0].gradients["value"]))
        self.assertNotEqual(float(proposals[2].gradients["value"]), float(proposals[0].gradients["value"]))

    def test_smart_weights_change_loss_and_gradient(self):
        module = nn.Module()
        module.register_parameter("value", nn.Parameter(torch.tensor(0.7)))
        context = loss_context(module.value)
        parameters = {"smart.value": module.value}
        proposals = [
            SmaRT(module, structure_weight=structure, consistency_weight=consistency).propose(context, parameters)
            for structure, consistency in ((0.0, 0.0), (0.4, 0.0), (0.0, 0.5))
        ]
        structure_loss, _ = two_lung_structure_loss(context.smart_logits)
        self.assertAlmostEqual(proposals[1].loss - proposals[0].loss, 0.4 * float(structure_loss.detach()), places=6)
        self.assertGreater(proposals[2].loss, proposals[0].loss)
        self.assertNotEqual(float(proposals[1].gradients["smart.value"]), float(proposals[0].gradients["smart.value"]))
        self.assertNotEqual(float(proposals[2].gradients["smart.value"]), float(proposals[0].gradients["smart.value"]))

    def test_grata_noise_setting_changes_augmented_forward(self):
        image = torch.linspace(-1, 1, 256).reshape(1, 16, 16)
        quiet = make_engine(configured(methods=("GraTa",), grata_noise_std=0.0))
        noisy = make_engine(configured(methods=("GraTa",), grata_noise_std=0.5))
        quiet_context = quiet._make_context(image, "quiet")
        noisy_context = noisy._make_context(image, "noisy")
        self.assertTrue(torch.equal(quiet_context.student_logits, quiet_context.augmented_logits))
        self.assertFalse(torch.equal(noisy_context.student_logits, noisy_context.augmented_logits))

    def test_engine_wires_settings_and_preserves_them_after_reset(self):
        config = configured(image_size=16)
        engine = make_engine(config)
        image = torch.linspace(-1, 1, 256).reshape(1, 16, 16)
        for index in range(4):
            engine.step(image, f"case-{index}")
        self.assertEqual(len(engine.vptta.memory), 2)
        self.assertEqual(len(engine.dltta.memory), 3)
        engine.reset()
        self.assertEqual(engine.step_index, 0)
        self.assertEqual(len(engine.vptta.memory), 0)
        self.assertEqual(len(engine.dltta.memory), 0)
        self.assertEqual(engine.vptta.memory_size, config.vptta_memory_size)
        self.assertEqual(engine.vptta.prompt_size, config.vptta_prompt_size)
        self.assertEqual(engine.vptta.prompt_strength, config.vptta_prompt_strength)
        self.assertEqual(engine.dltta.memory_size, config.dltta_memory_size)
        self.assertEqual(engine.testfit.confidence, config.testfit_confidence)
        self.assertEqual(engine.testfit.feature_weight, config.testfit_feature_weight)
        self.assertEqual(engine.grata.consistency_weight, config.grata_consistency_weight)
        self.assertEqual(engine.grata.feature_weight, config.grata_feature_weight)
        self.assertEqual(engine.smart.structure_weight, config.smart_structure_weight)
        self.assertEqual(engine.smart.consistency_weight, config.smart_consistency_weight)
        for name, method in engine._method_objects.items():
            self.assertEqual(method.lr if name != "DLTTA" else method.base_lr, config.lrs[name])
        prediction, _, _ = engine.step(image, "after-reset")
        self.assertEqual(prediction.shape, (1, 16, 16))

    def test_ema_decay_controls_teacher_update(self):
        for adaptation_steps in (1, 3, 5):
            with self.subTest(adaptation_steps=adaptation_steps):
                engine = make_engine(configured(
                    methods=("DLTTA", "SmaRT"), smart_ema_decay=0.25,
                    adaptation_steps=adaptation_steps,
                ))
                expected_ema = {name: value.detach().clone() for name, value in engine.bundle.ema.named_parameters()}
                with torch.no_grad():
                    engine.bundle.student.conv.weight.add_(0.1)
                commit = engine._commit

                def check_teacher_after_commit(*args, **kwargs):
                    record = commit(*args, **kwargs)
                    student = dict(engine.bundle.student.named_parameters())
                    for name, parameter in engine.bundle.ema.named_parameters():
                        expected_ema[name] = expected_ema[name].mul(0.25).add(student[name], alpha=0.75)
                        torch.testing.assert_close(parameter, expected_ema[name], rtol=0.0, atol=1e-7)
                    return record

                with patch.object(engine, "_commit", side_effect=check_teacher_after_commit) as commits:
                    engine.step(torch.linspace(-1, 1, 256).reshape(1, 16, 16), "ema-probe")
                self.assertEqual(commits.call_count, adaptation_steps)


if __name__ == "__main__":
    unittest.main()
