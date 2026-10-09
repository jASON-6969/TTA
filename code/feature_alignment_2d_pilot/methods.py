from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .data import ImageOnlyDataset, SegmentationDataset
from .metrics import aggregate, binary_metrics, save_prediction
from .model import UNet2D, entropy_loss, gradient_reverse, segmentation_loss, configure_bn_adaptation


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def train_source(model, train_loader, val_loader, device, epochs, patience, lr, checkpoint_path, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_dice = validate_dice(model, val_loader, device)
        logger(f"source epoch={epoch} train_loss={sum(losses)/max(len(losses),1):.5f} val_dice={val_dice:.5f}")
        if val_dice > best:
            best = val_dice
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_dice": best}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return {"best_val_dice": float(best), "best_epoch": best_epoch, "epochs_run": epoch}


def validate_dice(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for images, masks, _ in loader:
            logits = model(images.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            target = masks.numpy()
            for p, t in zip(pred, target):
                values.append(binary_metrics(p, t)["Dice"])
    return sum(values) / max(len(values), 1)


def evaluate(model: UNet2D, loader: DataLoader, device, method: str, prediction_dir: Path, logger):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, masks, sample_ids in loader:
            images = images.to(device)
            start = time.perf_counter()
            logits = model(images)
            elapsed = time.perf_counter() - start
            pred = logits.argmax(dim=1).cpu().numpy()
            targets = masks.numpy()
            for p, t, sample_id in zip(pred, targets, sample_ids):
                values = binary_metrics(p, t)
                row = {"method": method, "image_id": sample_id, **values, "inference_seconds": elapsed / len(sample_ids)}
                rows.append(row)
                save_prediction(p, prediction_dir / method / f"{sample_id}.png")
    logger(f"evaluated method={method} count={len(rows)} dice={aggregate(rows).get('Dice_mean'):.5f}")
    return rows


def adapt_tent(base_model, target_loader, device, lr, logger):
    model = deepcopy(base_model).to(device)
    params = configure_bn_adaptation(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = []
    for step, (images, _) in enumerate(target_loader, 1):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = entropy_loss(logits)
        loss.backward()
        optimizer.step()
        logs.append({"method": "TENT", "step": step, "loss": float(loss.detach().cpu()), "entropy": float(loss.detach().cpu())})
    logger(f"adapted method=TENT steps={len(logs)}")
    return model, logs


def adapt_testfit(base_model, target_loader, device, lr, logger):
    student = deepcopy(base_model).to(device)
    teacher = deepcopy(base_model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    logs = []
    for step, (images, _) in enumerate(target_loader, 1):
        images = images.to(device)
        with torch.no_grad():
            teacher_logits, teacher_features = teacher(images, return_features=True)
            confidence, pseudo = teacher_logits.softmax(dim=1).max(dim=1)
            valid = confidence > 0.80
        optimizer.zero_grad(set_to_none=True)
        student_logits, student_features = student(images, return_features=True)
        pixel_loss = F.cross_entropy(student_logits, pseudo, reduction="none")
        pseudo_loss = pixel_loss[valid].mean() if valid.any() else pixel_loss.mean() * 0.0
        feature_loss = F.mse_loss(student_features.mean(dim=(2, 3)), teacher_features.mean(dim=(2, 3)))
        loss = pseudo_loss + 0.10 * feature_loss
        loss.backward()
        optimizer.step()
        logs.append({"method": "TestFit", "step": step, "loss": float(loss.detach().cpu()), "entropy": float(entropy_loss(student_logits).detach().cpu())})
    logger(f"adapted method=TestFit steps={len(logs)}")
    return student, logs


def adapt_grata(base_model, target_loader, device, lr, logger):
    model = deepcopy(base_model).to(device)
    params = configure_bn_adaptation(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = []
    for step, (images, _) in enumerate(target_loader, 1):
        images = images.to(device)
        noisy = (images + 0.03 * torch.randn_like(images)).clamp(0, 1)
        optimizer.zero_grad(set_to_none=True)
        logits, features = model(images, return_features=True)
        aug_logits, aug_features = model(noisy, return_features=True)
        consistency = F.mse_loss(logits.softmax(dim=1), aug_logits.softmax(dim=1))
        feature_consistency = F.mse_loss(features.mean(dim=(2, 3)), aug_features.mean(dim=(2, 3)))
        loss = entropy_loss(logits) + consistency + 0.1 * feature_consistency
        loss.backward()
        optimizer.step()
        logs.append({"method": "GraTa", "step": step, "loss": float(loss.detach().cpu()), "entropy": float(entropy_loss(logits).detach().cpu())})
    logger(f"adapted method=GraTa steps={len(logs)}")
    return model, logs


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_channels, 128), nn.LeakyReLU(), nn.Linear(128, 1))

    def forward(self, features):
        return self.net(features.mean(dim=(2, 3)))


def adapt_dann(base_model, source_loader, target_loader, device, lr, logger, epochs=2):
    model = deepcopy(base_model).to(device)
    discriminator = DomainDiscriminator().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    logs = []
    for epoch in range(epochs):
        source_iter = iter(source_loader)
        for step, (target_images, _) in enumerate(target_loader, 1):
            try:
                source_images, source_masks, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_images, source_masks, _ = next(source_iter)
            source_images, source_masks = source_images.to(device), source_masks.to(device)
            target_images = target_images.to(device)
            with torch.no_grad():
                _, source_features = model(source_images, return_features=True)
                _, target_features = model(target_images, return_features=True)
            disc_optimizer.zero_grad(set_to_none=True)
            domain_logits = torch.cat([discriminator(source_features.detach()), discriminator(target_features.detach())])
            domain_labels = torch.cat([torch.zeros(source_images.size(0), 1, device=device), torch.ones(target_images.size(0), 1, device=device)])
            disc_loss = bce(domain_logits, domain_labels)
            disc_loss.backward()
            disc_optimizer.step()

            model_optimizer.zero_grad(set_to_none=True)
            source_logits, source_features = model(source_images, return_features=True)
            _, target_features = model(target_images, return_features=True)
            domain_logits = torch.cat([discriminator(gradient_reverse(source_features, 1.0)), discriminator(gradient_reverse(target_features, 1.0))])
            seg_loss = segmentation_loss(source_logits, source_masks)
            domain_loss = bce(domain_logits, domain_labels)
            total = seg_loss + 0.10 * domain_loss
            total.backward()
            model_optimizer.step()
            logs.append({"method": "DANN", "step": epoch * len(target_loader) + step, "loss": float(total.detach().cpu()), "entropy": float(domain_loss.detach().cpu())})
    logger(f"adapted method=DANN steps={len(logs)} source_assisted=true")
    return model, logs
