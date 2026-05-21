#!/usr/bin/env python3
"""Fast fixed8 QAT: warm-start from fixed16 → train → eval → export → optional csim hint."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hls_export import export_fixed_variant, export_shared, latest_checkpoint
from quant_models import QUANTIZED8_MNISTCNN

ROOT = Path(__file__).resolve().parent
MILESTONE_DIR = ROOT / "milestones_fixed8"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 5
    learning_rate: float = 5e-4
    warm_start: bool = True


def dataloaders(batch_size: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(ROOT / "data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tfm)
    total_set = torch.utils.data.ConcatDataset([train_set, test_set])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
        DataLoader(total_set, batch_size=batch_size, shuffle=False),
    )


def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def save_checkpoint(model: torch.nn.Module, cfg: TrainConfig) -> Path:
    MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = MILESTONE_DIR / f"QUANTIZED8_MNISTCNN-{stamp}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": "QUANTIZED8_MNISTCNN",
            "param": {
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "learning_rate": cfg.learning_rate,
            },
        },
        path,
    )
    return path


def maybe_warm_start(model: torch.nn.Module) -> None:
    try:
        ckpt = latest_checkpoint("fixed16")
    except FileNotFoundError:
        print("[fixed8] no fixed16 checkpoint — training from scratch")
        return
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model", sd.get("model_state_dict", sd))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[fixed8] warm-start from {ckpt.name} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[fixed8] warm-start from {ckpt.name}")


def train_fixed8(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fixed8] device={device}")

    train_loader, test_loader, total_loader = dataloaders(cfg.batch_size)
    model = QUANTIZED8_MNISTCNN().to(device)
    if cfg.warm_start:
        maybe_warm_start(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.num_epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        test_acc = accuracy(model, test_loader, device)
        print(f"epoch {epoch + 1}/{cfg.num_epochs}  loss={running / len(train_loader):.4f}  test={test_acc:.2f}%")

    ckpt_path = save_checkpoint(model, cfg)
    total_acc = accuracy(model, total_loader, device)
    print(f"[fixed8] saved {ckpt_path.name}")
    print(f"[fixed8] total (70k) accuracy: {total_acc:.2f}%")

    if total_acc < 93.0:
        print("[fixed8] WARNING: below 93% — try num_epochs=10 or lower lr")
    return ckpt_path


def main() -> int:
    cfg = TrainConfig()
    if "--scratch" in sys.argv:
        cfg = TrainConfig(warm_start=False)
    if "--epochs" in sys.argv:
        i = sys.argv.index("--epochs")
        cfg = TrainConfig(
            batch_size=cfg.batch_size,
            num_epochs=int(sys.argv[i + 1]),
            learning_rate=cfg.learning_rate,
            warm_start=cfg.warm_start,
        )

    ckpt = train_fixed8(cfg)

    if not (ROOT / "artifacts" / "hls" / "shared" / "images.h").exists():
        print("[export] shared images.h missing — running export_shared()")
        export_shared()

    export_fixed_variant("fixed8", ckpt_path=ckpt)
    print("\n[done] next: cd hls/fixed8 && bash run_csim.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
