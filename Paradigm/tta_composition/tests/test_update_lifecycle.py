"""Regressions for finite update audits and image-specific prompt memory timing."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

TEST_V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_V1))

from Paradigm.tta_composition.audit import record_to_dict
from Paradigm.tta_composition.config import CompositionConfig
from Paradigm.tta_composition.contracts import UpdateProposal
from Paradigm.tta_composition.tests.test_hyperparameters import make_engine


class NumericUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def image(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(17)
        return torch.randn((1, 1, 16, 16), generator=generator)

    def test_real_grata_large_finite_gradients_keep_prediction_and_audit_serializable(self) -> None:
        engine = make_engine(CompositionConfig(
            methods=("GraTa",), image_size=16,
            lrs={"GraTa": 1e-30}, grata_feature_weight=1e30,
        ))
        prediction, record, metrics = engine.step(self.image(), "large-finite-feature-loss")
        self.assertTrue(torch.isfinite(prediction.float()).all())
        self.assertTrue(all(math.isfinite(value) for value in record.losses.values()))
        self.assertTrue(any(value > 1e20 for value in record.gradient_norms.values()))
        self.assertTrue(all(math.isfinite(value) for value in record.gradient_norms.values()))
        serialized = json.dumps(
            record_to_dict(record, sample_id="large-finite-feature-loss", method_metrics=metrics),
            allow_nan=False,
        )
        self.assertEqual(json.loads(serialized)["image_id"], "large-finite-feature-loss")

    def test_finite_float32_gradient_norm_uses_enough_precision_for_large_values(self) -> None:
        engine = make_engine(CompositionConfig(methods=("GraTa",), image_size=16, lrs={"GraTa": 1e-30}))
        names = engine.registry.names_for("GraTa")
        gradients = {name: torch.full_like(engine.registry.parameter(name), 1e24) for name in names}
        proposal = UpdateProposal("GraTa", gradients, names, lr_hint=1e-30, loss=0.0)
        proposal.validate()
        with patch.object(engine.grata, "propose", return_value=proposal):
            prediction, record, metrics = engine.step(self.image(), "large-controlled-gradients")
        self.assertTrue(torch.isfinite(prediction.float()).all())
        for name, gradient in gradients.items():
            expected = math.sqrt(gradient.numel()) * float(gradient.flatten()[0])
            self.assertTrue(math.isfinite(record.gradient_norms[name]))
            self.assertTrue(math.isclose(record.gradient_norms[name], expected, rel_tol=1e-12))
        json.dumps(record_to_dict(record, sample_id="large-controlled-gradients", method_metrics=metrics), allow_nan=False)

    def test_nonfinite_norm_is_rejected_before_any_parameter_mutation(self) -> None:
        engine = make_engine(CompositionConfig(methods=("GraTa",), image_size=16))
        names = engine.registry.names_for("GraTa")
        gradients = {
            name: torch.full(engine.registry.parameter(name).shape, 1e308, dtype=torch.float64)
            for name in names
        }
        proposal = UpdateProposal("GraTa", gradients, names, lr_hint=1e-4, loss=0.0)
        proposal.validate()
        before = {name: parameter.detach().clone() for name, parameter in engine.registry.parameters.items()}
        with patch.object(engine.grata, "propose", return_value=proposal), self.assertRaises(FloatingPointError):
            engine.step(self.image(), "unrepresentable-gradient-norm")
        self.assertEqual(engine.step_index, 0)
        self.assertTrue(all(torch.equal(before[name], parameter.detach()) for name, parameter in engine.registry.parameters.items()))


class PromptPredictionLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        self.engine = make_engine(CompositionConfig(
            methods=("VPTTA",), image_size=16, vptta_memory_size=2,
            vptta_prompt_strength=0.1,
        ))
        self.image = torch.linspace(-1.0, 1.0, 256).reshape(1, 1, 16, 16)

    def test_adaptation_and_prediction_read_the_same_prior_prompt_memory(self) -> None:
        for adaptation_steps in (1, 3, 5):
            with self.subTest(adaptation_steps=adaptation_steps):
                engine = make_engine(CompositionConfig(
                    methods=("VPTTA",), image_size=16, vptta_memory_size=2,
                    vptta_prompt_strength=0.1, adaptation_steps=adaptation_steps,
                ))
                captures: list[list[torch.Tensor]] = []
                prepare = engine.vptta.prepare

                def capture(image, seed):
                    captures.append([prompt.detach().clone() for prompt in engine.vptta.memory])
                    return prepare(image, seed)

                with patch.object(engine.vptta, "prepare", side_effect=capture), \
                        patch.object(engine.vptta, "after_prediction", wraps=engine.vptta.after_prediction) as publications:
                    engine.step(self.image, "first-image")
                    engine.step(self.image, "second-image")
                calls_per_image = adaptation_steps + 1
                self.assertEqual([len(memory) for memory in captures], [0] * calls_per_image + [1] * calls_per_image)
                previous_prompt = captures[calls_per_image][0]
                self.assertTrue(all(torch.equal(previous_prompt, memory[0]) for memory in captures[calls_per_image:]))
                self.assertEqual(publications.call_count, 2)
                self.assertEqual(len(engine.vptta.memory), 2)

    def test_successful_prediction_stores_a_clone_of_the_updated_prompt(self) -> None:
        before = self.engine.vptta.prompt.detach().clone()
        prediction, record, _metrics = self.engine.step(self.image, "successful-image")
        self.assertTrue(torch.isfinite(prediction.float()).all())
        self.assertEqual(record.updated_parameters, 1)
        updated = self.engine.vptta.prompt.detach().clone()
        self.assertFalse(torch.equal(before, updated))
        self.assertEqual(len(self.engine.vptta.memory), 1)
        stored = self.engine.vptta.memory[-1]
        self.assertTrue(torch.equal(stored, updated))
        self.assertNotEqual(stored.data_ptr(), self.engine.vptta.prompt.data_ptr())
        with torch.no_grad():
            self.engine.vptta.prompt.add_(1.0)
        self.assertTrue(torch.equal(stored, updated))

    def test_failed_prediction_does_not_publish_the_current_prompt_to_memory(self) -> None:
        for adaptation_steps in (1, 3, 5):
            with self.subTest(adaptation_steps=adaptation_steps):
                engine = make_engine(CompositionConfig(
                    methods=("VPTTA",), image_size=16, adaptation_steps=adaptation_steps,
                ))
                engine.step(self.image, "successful-prior-image")
                previous = [prompt.detach().clone() for prompt in engine.vptta.memory]
                with patch.object(engine, "_predict", side_effect=FloatingPointError("controlled invalid prediction")), \
                        patch.object(engine.vptta, "after_prediction", wraps=engine.vptta.after_prediction) as publications, \
                        self.assertRaisesRegex(FloatingPointError, "controlled invalid prediction"):
                    engine.step(self.image, "failed-current-image")
                publications.assert_not_called()
                self.assertEqual(len(engine.vptta.memory), len(previous))
                self.assertTrue(all(torch.equal(old, current) for old, current in zip(previous, engine.vptta.memory)))

    def test_memory_stays_bounded_and_reset_reproduces_the_prediction_stream(self) -> None:
        first = []
        for index in range(5):
            prediction, record, _metrics = self.engine.step(self.image, str(index))
            first.append((prediction.clone(), dict(record.losses), self.engine.vptta.prompt.detach().clone()))
            self.assertEqual(len(self.engine.vptta.memory), min(index + 1, 2))
            self.assertTrue(torch.equal(self.engine.vptta.memory[-1], self.engine.vptta.prompt.detach()))
        self.engine.reset()
        self.assertEqual(self.engine.step_index, 0)
        self.assertEqual(len(self.engine.vptta.memory), 0)
        self.assertTrue(torch.equal(self.engine.vptta.prompt.detach(), torch.zeros_like(self.engine.vptta.prompt)))
        for index, (expected_prediction, expected_losses, expected_prompt) in enumerate(first):
            prediction, record, _metrics = self.engine.step(self.image, str(index))
            self.assertTrue(torch.equal(prediction, expected_prediction))
            self.assertEqual(dict(record.losses), expected_losses)
            self.assertTrue(torch.equal(self.engine.vptta.prompt.detach(), expected_prompt))
            self.assertLessEqual(len(self.engine.vptta.memory), 2)


class IterationLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_dltta_memory_is_committed_exactly_once_per_iteration(self) -> None:
        image = torch.linspace(-1.0, 1.0, 256).reshape(1, 1, 16, 16)
        for adaptation_steps in (1, 3, 5):
            with self.subTest(adaptation_steps=adaptation_steps):
                engine = make_engine(CompositionConfig(
                    methods=("DLTTA",), image_size=16, adaptation_steps=adaptation_steps,
                    dltta_memory_size=2,
                ))
                features: list[torch.Tensor] = []
                after_commit = engine.dltta.after_commit

                def publish_feature(context):
                    features.append(context.student_features.detach().mean(dim=(0, 2, 3)).clone())
                    after_commit(context)
                    expected_memory = features[-engine.dltta.memory_size:]
                    self.assertEqual(len(engine.dltta.memory), len(expected_memory))
                    self.assertTrue(all(torch.equal(expected, stored) for expected, stored in zip(expected_memory, engine.dltta.memory)))

                with patch.object(engine.dltta, "after_commit", side_effect=publish_feature) as publications:
                    engine.step(image, "first-image")
                    self.assertEqual(publications.call_count, adaptation_steps)
                    engine.step(image, "second-image")
                    self.assertEqual(publications.call_count, 2 * adaptation_steps)
                self.assertEqual(engine.step_index, 2 * adaptation_steps)

    def test_iteration_audits_preserve_every_commit_and_last_step_compatibility(self) -> None:
        image = torch.linspace(-1.0, 1.0, 256).reshape(1, 1, 16, 16)
        for adaptation_steps in (1, 3, 5):
            with self.subTest(adaptation_steps=adaptation_steps):
                engine = make_engine(CompositionConfig(
                    methods=("DLTTA",), image_size=16, adaptation_steps=adaptation_steps,
                ))
                commits = []
                commit = engine._commit

                def capture_commit(*args, **kwargs):
                    record = commit(*args, **kwargs)
                    commits.append(record)
                    return record

                with patch.object(engine, "_commit", side_effect=capture_commit):
                    for image_index in range(2):
                        prediction, record, metrics = engine.step(image, str(image_index))
                        expected_commits = commits[-adaptation_steps:]
                        self.assertEqual(record.iterations, tuple(expected_commits))
                        first_step = image_index * adaptation_steps + 1
                        self.assertEqual([update.step for update in record.iterations], list(range(first_step, first_step + adaptation_steps)))
                        self.assertEqual(record.step, expected_commits[-1].step)
                        self.assertEqual(record.updated_parameters, expected_commits[-1].updated_parameters)
                        self.assertEqual(sum(update.updated_parameters for update in record.iterations),
                                         adaptation_steps * len(engine.registry.names_for("DLTTA")))
                        self.assertGreaterEqual(record.elapsed_seconds, sum(update.elapsed_seconds for update in record.iterations))
                        self.assertEqual(metrics["adaptation_iterations"], float(adaptation_steps))
                        self.assertTrue(torch.isfinite(prediction.float()).all())
                        serialized = record_to_dict(record, sample_id=str(image_index), method_metrics=metrics)
                        self.assertEqual(len(serialized["iterations"]), adaptation_steps)
                        self.assertEqual([update["step"] for update in serialized["iterations"]], [update.step for update in expected_commits])
                        for saved, update in zip(serialized["iterations"], expected_commits):
                            self.assertEqual(saved["losses"], dict(update.losses))
                            self.assertEqual(saved["lr_hints"], dict(update.lr_hints))
                            self.assertEqual(saved["final_learning_rates"], dict(update.final_learning_rates))
                            self.assertEqual(saved["gradient_norms"], dict(update.gradient_norms))
                            self.assertEqual(saved["updated_parameters"], update.updated_parameters)
                            self.assertEqual(saved["elapsed_seconds"], update.elapsed_seconds)
                        json.dumps(serialized, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
