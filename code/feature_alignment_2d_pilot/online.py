from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from .data import Record, ImageOnlyDataset, SegmentationDataset, load_mask
from .metrics import binary_metrics, save_prediction
from .model import UNet2D, configure_bn_adaptation, entropy_loss, freeze_bn_running_stats


def _prediction(model: nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(image.to(device)).argmax(dim=1).cpu()


def _write_prediction(prediction: torch.Tensor, record: Record, output_dir: Path):
    save_prediction(prediction[0].numpy(), output_dir / f"{record.sample_id}.png")


def run_source_only(base_model: UNet2D, records: list[Record], image_size: int, device: torch.device, output_dir: Path, logger, normalization: str = "percentile"):
    model = deepcopy(base_model).to(device)
    model.eval()
    dataset = ImageOnlyDataset(records, image_size, normalization)
    rows = []
    for step in range(len(dataset)):
        image, sample_id = dataset[step]
        record = records[step]
        start = time.perf_counter()
        prediction = _prediction(model, image.unsqueeze(0), device)
        elapsed = time.perf_counter() - start
        _write_prediction(prediction, record, output_dir)
        rows.append({"method": "Source-only", "step": step + 1, "image_id": sample_id, "loss": None, "entropy": None, "seconds": elapsed})
    logger(f"online method=Source-only count={len(rows)}")
    return rows


def run_tent(base_model: UNet2D, records: list[Record], image_size: int, device: torch.device, output_dir: Path, logger, lr: float = 1e-4, normalization: str = "percentile"):
    model = deepcopy(base_model).to(device)
    params = configure_bn_adaptation(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    dataset = ImageOnlyDataset(records, image_size, normalization)
    rows = []
    for step in range(len(dataset)):
        image, sample_id = dataset[step]
        record = records[step]
        image = image.unsqueeze(0).to(device)
        start = time.perf_counter()
        model.train()
        freeze_bn_running_stats(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = entropy_loss(logits)
        loss.backward()
        optimizer.step()
        prediction = _prediction(model, image, device)
        elapsed = time.perf_counter() - start
        _write_prediction(prediction, record, output_dir)
        rows.append({"method": "TENT", "step": step + 1, "image_id": sample_id, "loss": float(loss.detach().cpu()), "entropy": float(loss.detach().cpu()), "seconds": elapsed})
    logger(f"online method=TENT count={len(rows)}")
    return rows


def run_testfit(base_model: UNet2D, records: list[Record], image_size: int, device: torch.device, output_dir: Path, logger, lr: float = 1e-5, normalization: str = "percentile"):
    student = deepcopy(base_model).to(device)
    teacher = deepcopy(base_model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    dataset = ImageOnlyDataset(records, image_size, normalization)
    rows = []
    for step in range(len(dataset)):
        image, sample_id = dataset[step]
        record = records[step]
        image = image.unsqueeze(0).to(device)
        start = time.perf_counter()
        teacher.eval()
        with torch.no_grad():
            teacher_logits, teacher_features = teacher(image, return_features=True)
            confidence, pseudo = teacher_logits.softmax(dim=1).max(dim=1)
            valid = confidence > 0.80
        student.train()
        freeze_bn_running_stats(student)
        optimizer.zero_grad(set_to_none=True)
        student_logits, student_features = student(image, return_features=True)
        pixel_loss = F.cross_entropy(student_logits, pseudo, reduction="none")
        pseudo_loss = pixel_loss[valid].mean() if valid.any() else pixel_loss.mean() * 0.0
        feature_loss = F.mse_loss(student_features.mean(dim=(2, 3)), teacher_features.mean(dim=(2, 3)))
        loss = pseudo_loss + 0.10 * feature_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
                teacher_parameter.mul_(0.99).add_(student_parameter, alpha=0.01)
        prediction = _prediction(student, image, device)
        elapsed = time.perf_counter() - start
        _write_prediction(prediction, record, output_dir)
        rows.append({"method": "TestFit", "step": step + 1, "image_id": sample_id, "loss": float(loss.detach().cpu()), "entropy": float(entropy_loss(student_logits).detach().cpu()), "seconds": elapsed})
    logger(f"online method=TestFit count={len(rows)}")
    return rows


def run_grata(base_model: UNet2D, records: list[Record], image_size: int, device: torch.device, output_dir: Path, logger, lr: float = 1e-4, normalization: str = "percentile"):
    model = deepcopy(base_model).to(device)
    params = configure_bn_adaptation(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    dataset = ImageOnlyDataset(records, image_size, normalization)
    rows = []
    for step in range(len(dataset)):
        image, sample_id = dataset[step]
        record = records[step]
        image = image.unsqueeze(0).to(device)
        start = time.perf_counter()
        model.train()
        freeze_bn_running_stats(model)
        noisy = image + 0.03 * torch.randn_like(image)
        if normalization == "percentile":
            noisy = noisy.clamp(0, 1)
        optimizer.zero_grad(set_to_none=True)
        logits, features = model(image, return_features=True)
        augmented_logits, augmented_features = model(noisy, return_features=True)
        consistency = F.mse_loss(logits.softmax(dim=1), augmented_logits.softmax(dim=1))
        feature_consistency = F.mse_loss(features.mean(dim=(2, 3)), augmented_features.mean(dim=(2, 3)))
        entropy = entropy_loss(logits)
        loss = entropy + consistency + 0.10 * feature_consistency
        loss.backward()
        optimizer.step()
        prediction = _prediction(model, image, device)
        elapsed = time.perf_counter() - start
        _write_prediction(prediction, record, output_dir)
        rows.append({"method": "GraTa", "step": step + 1, "image_id": sample_id, "loss": float(loss.detach().cpu()), "entropy": float(entropy.detach().cpu()), "seconds": elapsed})
    logger(f"online method=GraTa count={len(rows)}")
    return rows


def evaluate_saved_predictions(records: list[Record], prediction_root: Path, methods: list[str], image_size: int):
    """Evaluate only after all online adaptation streams have completed."""
    eval_dataset = SegmentationDataset(records, image_size)
    target_by_id = {record.sample_id: record for record in records}
    rows = []
    for method in methods:
        method_dir = prediction_root / method
        for record in records:
            prediction_path = method_dir / f"{record.sample_id}.png"
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            from PIL import Image
            import numpy as np
            prediction = np.asarray(Image.open(prediction_path).convert("L"), dtype=np.uint8) > 127
            target_record = target_by_id[record.sample_id]
            target = load_mask(target_record.mask_path, image_size).numpy().astype(bool)
            values = binary_metrics(prediction, target)
            rows.append({"method": method, "image_id": record.sample_id, **values})
    return rows
