"""CPU regressions for identity initialization and style-encoder learning."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

TEST_V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_V1))

from Paradigm.tta_composition.model_bridge import SmartAdapter, build_model_bundle


class CheckpointModel(nn.Module):
    """Small state-only checkpoint fixture; no training or segmentation labels."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)


class StyleEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            self.adapter = SmartAdapter()
        ramp = torch.linspace(-1.0, 1.0, 16).reshape(1, 16).expand(16, 16)
        stripes = ((torch.arange(16) % 2).float() * 2 - 1).reshape(1, 16).expand(16, 16) + 0.5
        self.images = torch.stack((ramp, stripes)).unsqueeze(1)
        # Synthetic style-vector supervision probes trainability, not TTA quality.
        self.style_targets = torch.tensor([[-0.3, 0.2], [0.4, -0.1]])
        self.logits = torch.cat((self.images, -0.3 * self.images + 0.7), dim=1)

    def _train_encoder(self, steps: int) -> list[dict[str, float]]:
        optimizer = torch.optim.SGD(self.adapter.style_encoder.parameters(), lr=0.25)
        gradient_history: list[dict[str, float]] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(self.adapter.style_encoder(self.images), self.style_targets)
            loss.backward()
            gradients = {}
            for name, parameter in self.adapter.style_encoder.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                assert parameter.grad is not None
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                gradients[name] = float(parameter.grad.norm())
            gradient_history.append(gradients)
            optimizer.step()
        return gradient_history

    def _assert_identity(self) -> None:
        with torch.no_grad():
            self.assertTrue(torch.equal(self.adapter.transform(self.images), self.images))
            self.assertTrue(torch.equal(self.adapter.augment(self.images, seed=17), self.images))
            self.assertTrue(torch.equal(self.adapter.correct(self.logits), self.logits))

    def test_initial_transform_augmentation_and_correction_are_exact_identity(self) -> None:
        self._assert_identity()
        style = self.adapter.style_encoder(self.images)
        self.assertTrue(torch.equal(style, torch.zeros_like(style)))

    def test_hidden_features_are_nonzero_and_depend_on_input(self) -> None:
        hidden = self.adapter.style_encoder[:-1](self.images)
        conv = self.adapter.style_encoder[0]
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertGreater(int(torch.count_nonzero(conv.weight)), 0)
        self.assertGreater(int(torch.count_nonzero(hidden)), 0)
        self.assertTrue(bool(torch.isfinite(hidden).all()))
        self.assertFalse(torch.equal(hidden[0], hidden[1]))

    def test_first_update_learns_head_before_hidden_layers(self) -> None:
        gradients = self._train_encoder(1)[0]
        self.assertGreater(gradients["4.weight"], 0)
        self.assertGreater(gradients["4.bias"], 0)
        self.assertEqual(gradients["0.weight"], 0)
        self.assertEqual(gradients["0.bias"], 0)
        head = self.adapter.style_encoder[-1]
        self.assertGreater(int(torch.count_nonzero(head.weight)), 0)
        self.assertGreater(int(torch.count_nonzero(head.bias)), 0)

    def test_subsequent_updates_learn_hidden_weights_and_condition_style_on_image(self) -> None:
        initial = deepcopy(self.adapter.style_encoder.state_dict())
        history = self._train_encoder(20)
        self.assertGreater(history[1]["0.weight"], 0)
        self.assertGreater(history[1]["0.bias"], 0)
        self.assertGreater(history[-1]["4.weight"], 0)
        self.assertGreater(history[-1]["0.weight"], 0)
        for name, parameter in self.adapter.style_encoder.named_parameters():
            self.assertFalse(torch.equal(parameter, initial[name]), name)
        styles = self.adapter.style_encoder(self.images).detach()
        self.assertGreater(float((styles[0] - styles[1]).abs().max()), 1e-4)

    def test_zero_output_residual_receives_gradient_and_learns(self) -> None:
        initial = deepcopy(self.adapter.output_correction.state_dict())
        optimizer = torch.optim.SGD(self.adapter.output_correction.parameters(), lr=0.1)
        offset = torch.tensor([0.1, -0.1]).reshape(1, 2, 1, 1)
        loss = F.mse_loss(self.adapter.correct(self.logits), self.logits * 1.05 + offset)
        loss.backward()
        for name, parameter in self.adapter.output_correction.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertGreater(float(parameter.grad.norm()), 0, name)
        optimizer.step()
        for name, parameter in self.adapter.output_correction.named_parameters():
            self.assertFalse(torch.equal(parameter, initial[name]), name)

    def test_loading_initial_snapshot_restores_exact_identity_and_hidden_weights(self) -> None:
        initial = deepcopy(self.adapter.state_dict())
        self._train_encoder(5)
        with torch.no_grad():
            self.adapter.style_scale.fill_(1.2)
            self.adapter.style_shift.fill_(0.2)
            self.adapter.augmentation_strength.fill_(0.3)
            self.adapter.output_correction.weight.fill_(0.1)
        self.adapter.load_state_dict(initial)
        for name, value in self.adapter.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name]), name)
        self._assert_identity()

    def test_model_bridge_preserves_caller_rng_with_smart_enabled_or_disabled(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            checkpoint = deepcopy(CheckpointModel().state_dict())
            torch.manual_seed(123)
            CheckpointModel()
            expected_rng = torch.get_rng_state().clone()
            for enabled in (False, True):
                torch.manual_seed(123)
                with patch(
                    "Paradigm.tta_composition.model_bridge._load_pilot_modules",
                    return_value=(CheckpointModel, None, lambda model: None),
                ), patch(
                    "Paradigm.tta_composition.model_bridge.torch.load",
                    return_value={"model_state_dict": checkpoint},
                ):
                    bundle = build_model_bundle(Path("unused-checkpoint.pth"), torch.device("cpu"), enable_smart=enabled)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                for name, value in checkpoint.items():
                    self.assertTrue(torch.equal(bundle.reference.state_dict()[name], value))
                    self.assertTrue(torch.equal(bundle.student.state_dict()[name], value))
                self.assertEqual(bundle.ema is not None, enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
