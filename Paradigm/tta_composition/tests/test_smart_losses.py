"""Gradient, batch and resolution checks for the local SmaRT loss terms."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

TEST_V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_V1))

from Paradigm.tta_composition.contracts import StepContext
from Paradigm.tta_composition.methods.smart import SmaRT, count_foreground_components, pixelwise_ema_kl, two_lung_structure_loss
from Paradigm.tta_composition.methods.structure import excess_component_suppression


def region_mask(count: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    regions = []
    for row_slice, col_slice in (
        (slice(1, 4), slice(1, 4)),  # 9 pixels
        (slice(6, 8), slice(1, 4)),  # 6 pixels
        (slice(1, 3), slice(7, 9)),  # 4 pixels, found before the second largest
        (slice(8, 9), slice(8, 9)),  # 1 pixel
    )[:count]:
        region = torch.zeros(12, 12, dtype=torch.bool)
        region[row_slice, col_slice] = True
        regions.append(region)
    mask = torch.zeros(12, 12, dtype=torch.bool)
    for region in regions:
        mask |= region
    return mask, regions


def region_logits(mask: torch.Tensor) -> torch.Tensor:
    foreground = torch.where(mask, 0.8, 0.1).to(torch.float64)
    log_odds = (foreground / (1.0 - foreground)).log()
    return torch.stack((torch.zeros_like(log_odds), log_odds)).unsqueeze(0).detach().requires_grad_(True)


def neighborhood_loss(logits: torch.Tensor) -> torch.Tensor:
    foreground = logits.softmax(dim=1)[:, 1:2]
    return ((foreground[..., :, 1:] - foreground[..., :, :-1]).abs().mean()
            + (foreground[..., 1:, :] - foreground[..., :-1, :]).abs().mean())


class SmartLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_zero_one_two_regions_have_no_extra_penalty_and_can_backpropagate(self):
        for count in range(3):
            with self.subTest(count=count):
                mask, _ = region_mask(count)
                logits = region_logits(mask)
                loss, metrics = excess_component_suppression(logits.softmax(dim=1)[:, 1:2])
                self.assertEqual(metrics["foreground_components"], count)
                self.assertEqual(metrics["extra_component_loss"], 0.0)
                self.assertEqual(float(loss.detach()), 0.0)
                loss.backward()
                self.assertIsNotNone(logits.grad)
                self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_three_four_regions_penalize_only_extras_with_positive_foreground_gradient(self):
        for count in (3, 4):
            with self.subTest(count=count):
                mask, regions = region_mask(count)
                logits = region_logits(mask)
                loss, metrics = excess_component_suppression(logits.softmax(dim=1)[:, 1:2])
                self.assertEqual(metrics["foreground_components"], count)
                self.assertAlmostEqual(float(loss.detach()), 0.01 * 0.8 * (count - 2), places=9)
                gradients = torch.autograd.grad(loss, logits)[0]
                expected = torch.zeros(12, 12, dtype=torch.float64)
                for region in regions[2:]:
                    expected[region] = 0.01 * 0.8 * 0.2 / int(region.sum())
                torch.testing.assert_close(gradients[0, 1], expected, rtol=1e-6, atol=1e-10)
                torch.testing.assert_close(gradients[0, 0], -expected, rtol=1e-6, atol=1e-10)
                extra_mask = torch.stack(regions[2:]).any(dim=0)
                self.assertTrue(bool((gradients[0, 1][extra_mask] > 0).all()))
                self.assertTrue(bool((gradients[0, 1][~extra_mask] == 0).all()))

    def test_structure_gradient_contains_extra_signal_in_addition_to_tv(self):
        mask, regions = region_mask(4)
        logits = region_logits(mask)
        structure, metrics = two_lung_structure_loss(logits)
        total_gradient = torch.autograd.grad(structure, logits)[0]
        tv_gradient = torch.autograd.grad(neighborhood_loss(logits), logits)[0]
        expected = torch.zeros_like(logits)
        for region in regions[2:]:
            expected[0, 1][region] = 0.01 * 0.8 * 0.2 / int(region.sum())
            expected[0, 0][region] = -0.01 * 0.8 * 0.2 / int(region.sum())
        torch.testing.assert_close(total_gradient - tv_gradient, expected, rtol=1e-6, atol=1e-10)
        self.assertGreater(metrics["extra_component_loss"], 0.0)
        self.assertGreater(metrics["neighborhood"], 0.0)

    def test_equal_area_ties_keep_first_two_regions_in_row_major_order(self):
        mask = torch.zeros(6, 6, dtype=torch.bool)
        mask[1, 1] = mask[1, 4] = mask[4, 1] = True
        logits = region_logits(mask)
        loss, _ = excess_component_suppression(logits.softmax(dim=1)[:, 1:2])
        gradient = torch.autograd.grad(loss, logits)[0][0, 1]
        self.assertEqual(float(gradient[1, 1]), 0.0)
        self.assertEqual(float(gradient[1, 4]), 0.0)
        self.assertGreater(float(gradient[4, 1]), 0.0)

    def test_region_mean_penalty_is_not_diluted_by_background_resolution(self):
        mask, _ = region_mask(3)
        losses = []
        for size in (32, 256):
            expanded = torch.zeros(size, size, dtype=torch.bool)
            expanded[:12, :12] = mask
            logits = region_logits(expanded)
            penalty, _ = excess_component_suppression(logits.softmax(dim=1)[:, 1:2])
            losses.append(float(penalty.detach()))
        self.assertAlmostEqual(losses[0], 0.008, places=9)
        self.assertAlmostEqual(losses[0], losses[1], places=12)

    def test_many_single_pixel_islands_have_finite_region_mean_gradients(self):
        mask = torch.arange(32).view(-1, 1) % 2 == torch.arange(32).view(1, -1) % 2
        foreground = torch.full((1, 1, 32, 32), 0.1, dtype=torch.float64)
        foreground[0, 0][mask] = 0.8
        foreground.requires_grad_(True)
        loss, metrics = excess_component_suppression(foreground)
        self.assertEqual(metrics["foreground_components"], 512)
        self.assertAlmostEqual(float(loss.detach()), 510 * 0.008, places=12)
        loss.backward()
        expected = mask.to(torch.float64) * 0.01
        expected[0, 0] = expected[0, 2] = 0.0
        torch.testing.assert_close(foreground.grad[0, 0], expected, rtol=0.0, atol=1e-12)

    def test_batch_loss_metrics_and_gradients_equal_independent_case_average(self):
        single_losses, single_gradients, masks = [], [], []
        for count in range(5):
            mask, _ = region_mask(count)
            masks.append(mask)
            logits = region_logits(mask)
            loss, _ = two_lung_structure_loss(logits)
            single_losses.append(float(loss.detach()))
            single_gradients.append(torch.autograd.grad(loss, logits)[0])
        logits = torch.cat([region_logits(mask) for mask in masks]).detach().requires_grad_(True)
        batch_loss, metrics = two_lung_structure_loss(logits)
        gradient = torch.autograd.grad(batch_loss, logits)[0]
        self.assertAlmostEqual(float(batch_loss.detach()), sum(single_losses) / 5, places=12)
        self.assertEqual(metrics["foreground_components"], 2.0)
        self.assertAlmostEqual(metrics["extra_component_loss"], (0.008 + 0.016) / 5, places=9)
        for index, single_gradient in enumerate(single_gradients):
            torch.testing.assert_close(gradient[index], single_gradient[0] / 5, rtol=1e-9, atol=1e-12)

    def test_single_pixel_spatial_dimensions_have_finite_losses_and_gradients(self):
        for shape in ((1, 1), (1, 8), (8, 1)):
            with self.subTest(shape=shape):
                mask = torch.arange(shape[0] * shape[1]).reshape(shape) % 2 == 0
                logits = region_logits(mask)
                loss, metrics = two_lung_structure_loss(logits)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(bool(torch.isfinite(logits.grad).all()))
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_component_count_public_function_keeps_single_sample_shapes(self):
        mask, _ = region_mask(4)
        for sample in (mask, mask.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0)):
            self.assertEqual(count_foreground_components(sample), 4)

    def test_kl_is_pixel_mean_and_has_same_shared_logit_gradient_at_32_and_256(self):
        outcomes = []
        for size in (32, 256):
            student = nn.Parameter(torch.tensor([0.4, -0.4], dtype=torch.float64))
            teacher = nn.Parameter(torch.tensor([-0.2, 0.2], dtype=torch.float64))
            probabilities = student.view(1, 2, 1, 1).expand(1, 2, size, size).softmax(dim=1)
            teacher_logits = teacher.view(1, 2, 1, 1).expand_as(probabilities)
            loss = pixelwise_ema_kl(probabilities, teacher_logits)
            loss.backward()
            expected_gradient = student.detach().softmax(dim=0) - teacher.detach().softmax(dim=0)
            torch.testing.assert_close(student.grad, expected_gradient, rtol=1e-10, atol=1e-12)
            self.assertIsNone(teacher.grad)
            outcomes.append((float(loss.detach()), student.grad.clone()))
        self.assertAlmostEqual(outcomes[0][0], outcomes[1][0], places=12)
        torch.testing.assert_close(outcomes[0][1], outcomes[1][1], rtol=1e-10, atol=1e-12)

    def test_smart_proposal_uses_normalized_kl_and_preserves_default_coefficient(self):
        proposals = []
        for size in (32, 256):
            module = nn.Module()
            module.register_parameter("value", nn.Parameter(torch.tensor([0.4, -0.4], dtype=torch.float64)))
            teacher = torch.tensor([-0.2, 0.2], dtype=torch.float64, requires_grad=True)
            logits = module.value.view(1, 2, 1, 1).expand(1, 2, size, size)
            teacher_logits = teacher.view(1, 2, 1, 1).expand_as(logits)
            image = torch.zeros(1, 1, size, size, dtype=torch.float64)
            context = StepContext(
                step=1, sample_id="resolution-probe", image=image, transformed_image=image,
                reference_logits=teacher_logits, reference_features=image, student_logits=logits,
                student_features=image, smart_logits=logits, ema_logits=teacher_logits,
            )
            adapter = SmaRT(module)
            proposal = adapter.propose(context, {"smart.value": module.value})
            probabilities = module.value.softmax(dim=0)
            target = teacher.detach().softmax(dim=0)
            expected_kl = (target * (target.log() - probabilities.log())).sum()
            expected_loss = -(probabilities * probabilities.log()).sum() + adapter.consistency_weight * expected_kl
            self.assertAlmostEqual(proposal.metrics["ema_consistency"], float(expected_kl.detach()), places=12)
            self.assertAlmostEqual(proposal.loss, float(expected_loss.detach()), places=12)
            self.assertEqual(proposal.metrics["extra_component_loss"], 0.0)
            expected_gradient = torch.autograd.grad(expected_loss, module.value)[0]
            torch.testing.assert_close(proposal.gradients["smart.value"], expected_gradient, rtol=1e-10, atol=1e-12)
            self.assertIsNone(teacher.grad)
            proposals.append(proposal)
        self.assertAlmostEqual(proposals[0].loss, proposals[1].loss, places=12)
        torch.testing.assert_close(proposals[0].gradients["smart.value"], proposals[1].gradients["smart.value"], rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
