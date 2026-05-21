"""Generate EEE426_QUANT.ipynb — three quantization sections + HLS export."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "EEE426_QUANT.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        "# EEE429 — Quantization Candidates & HLS Export\n\n"
        "| Section | Format | Model class | Milestone dir | HLS output |\n"
        "|---------|--------|-------------|---------------|------------|\n"
        "| **Shared** | — | — | — | `artifacts/hls/shared/` |\n"
        "| **1. Fixed 16/7** | ap_fixed⟨16,7⟩ | `QUANTIZED16_MNISTCNN` | `milestones_fixed16/` | `artifacts/hls/fixed16/weights.h` |\n"
        "| **2. Fixed 8/4** | ap_fixed⟨8,4⟩ | `QUANTIZED8_MNISTCNN` | `milestones_fixed8/` | `artifacts/hls/fixed8/weights.h` |\n"
        "| **3. BNN** | 1-bit ±1 + BN + FC | `BINARY_MNISTCNN` | `milestones_bnn/` | `artifacts/hls/bnn/weights.h` |\n\n"
        "Run **Shared setup** first, then train/eval/export per section."
    ),
    code(
        """import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass, InitVar, field, asdict
from pathlib import Path

from quant_models import VARIANTS, QUANTIZED16_MNISTCNN, QUANTIZED8_MNISTCNN, BINARY_MNISTCNN, calibrate_bn
from hls_export import export_variant, export_shared, export_all_trained, latest_checkpoint

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('device:', device)"""
    ),
    code(
        """@dataclass(frozen=True)
class HyperParameter:
    batch_size: int
    num_epochs: int
    learning_rate: float

@dataclass
class MNIST:
    root: InitVar[str]
    param: InitVar[HyperParameter]
    train: DataLoader = field(init=False)
    test:  DataLoader = field(init=False)
    total: DataLoader = field(init=False)

    def __post_init__(self, root, param):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        tr = datasets.MNIST(root=root, train=True,  download=True, transform=tf)
        te = datasets.MNIST(root=root, train=False, download=True, transform=tf)
        self.train = DataLoader(tr, batch_size=param.batch_size, shuffle=True)
        self.test  = DataLoader(te, batch_size=param.batch_size, shuffle=False)
        self.total = DataLoader(tr + te, batch_size=param.batch_size, shuffle=False)

@dataclass
class Routine:
    device: object
    criterion: object
    optimizer: object

    def _routine(self, model, loader):
        hits, loss_sum = 0, 0.0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            out = model(images)
            pred = out.argmax(1)
            loss = self.criterion(out, labels)
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            hits += (pred == labels).sum().item()
            loss_sum += loss.item()
        n = len(loader.dataset)
        return hits / n, loss_sum / len(loader)

    def train(self, model, loader):
        model.train()
        return self._routine(model, loader)

    def test(self, model, loader):
        model.eval()
        with torch.no_grad():
            return self._routine(model, loader)

@dataclass
class Milestone:
    parent: str | Path
    model_name: str
    ext: str
    now: str = field(init=False)

    def __post_init__(self):
        from datetime import datetime
        self.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.parent = Path(self.parent)
        self.parent.mkdir(parents=True, exist_ok=True)

    def __iter__(self):
        files = self.parent.glob(f'{self.model_name}-*.{self.ext}')
        return iter(sorted(files, key=lambda f: f.name, reverse=True))

    def save(self, **kwargs):
        import torch
        torch.save(kwargs, self.parent / f'{self.model_name}-{self.now}.{self.ext}')

def run_training(variant_key: str, param: HyperParameter):
    \"\"\"Train one variant; save best test checkpoint.\"\"\"
    cfg = VARIANTS[variant_key]
    mnist = MNIST('./data', param)
    model = cfg['model_cls']().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param.learning_rate)
    routine = Routine(device, criterion, optimizer)
    ms = Milestone(f\"./{cfg['milestone_dir']}\", cfg['model_name'], 'pth')

    print(f'=== {variant_key} ===  model={cfg[\"model_name\"]}')
    print(f'| {\"EPOCH\":^5} | {\"TRAIN ACC\":^12} | {\"TEST ACC\":^12} |')
    best = None
    for epoch in range(param.num_epochs):
        tr_acc, _ = routine.train(model, mnist.train)
        if variant_key == 'bnn':
            calibrate_bn(model, mnist.train, device)
        te_acc, _ = routine.test(model, mnist.test)
        mark = ''
        if best is None or te_acc > best['accuracy']:
            best = dict(epoch=epoch, accuracy=te_acc,
                        param=asdict(param), model=model.state_dict(),
                        optim=optimizer.state_dict())
            ms.save(**best)
            mark = ' *'
        print(f'| {epoch:^5} | {tr_acc:^12.2%} | {te_acc:^12.2%} |{mark}')
    return ms, mnist, routine

def eval_checkpoints(variant_key: str, mnist, routine):
    cfg = VARIANTS[variant_key]
    ms = Milestone(f\"./{cfg['milestone_dir']}\", cfg['model_name'], 'pth')
    model = cfg['model_cls']()
    model.eval()
    with torch.no_grad():
        for f in ms:
            ck = torch.load(f, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            model.to(device)
            acc, _ = routine.test(model, mnist.total)
            print(f'{f.stem}: {acc:.2%}')"""
    ),
    md("---\n## Shared — test vectors for all HLS variants\n\nExports `images.h` + `MNIST_DATASET_*.npy` → `artifacts/hls/shared/`"),
    code("export_shared()"),
    md(
        "---\n## Section 1 — Fixed 16/7 (ap_fixed⟨16,7⟩)\n\n"
        "- Range ≈ [−64, 64), step 2⁻⁹\n"
        "- Graph: Input→Q→Conv→Q→ReLU (×3)→Pool→Q→FC\n"
        "- Export: `export_variant('fixed16')` → `artifacts/hls/fixed16/weights.h`"
    ),
    code(
        """param16 = HyperParameter(batch_size=256, num_epochs=5, learning_rate=1e-3)
ms16, mnist16, routine16 = run_training('fixed16', param16)"""
    ),
    code("eval_checkpoints('fixed16', mnist16, routine16)"),
    code("export_variant('fixed16')"),
    md(
        "---\n## Section 2 — Fixed 8/4 (ap_fixed⟨8,4⟩)\n\n"
        "- Range ≈ [−8, 8), step 2⁻⁴\n"
        "- Same graph as 16-bit; narrower dynamic range\n"
        "- Export: `export_variant('fixed8')` → `artifacts/hls/fixed8/weights.h`"
    ),
    code(
        """param8 = HyperParameter(batch_size=256, num_epochs=10, learning_rate=1e-3)
ms8, mnist8, routine8 = run_training('fixed8', param8)"""
    ),
    code("eval_checkpoints('fixed8', mnist8, routine8)"),
    code("export_variant('fixed8')"),
    md(
        "---\n## Section 3 — BNN (Binary Neural Network)\n\n"
        "- Conv weights & activations → {−1, +1}\n"
        "- BatchNorm (float) before each binarization\n"
        "- FC layer stays float32\n"
        "- Export: `export_variant('bnn')` → `artifacts/hls/bnn/weights.h` (int8 conv + float BN/FC)"
    ),
    code(
        """param_bnn = HyperParameter(batch_size=256, num_epochs=20, learning_rate=1e-3)
ms_bnn, mnist_bnn, routine_bnn = run_training('bnn', param_bnn)"""
    ),
    code("eval_checkpoints('bnn', mnist_bnn, routine_bnn)"),
    code("export_variant('bnn')"),
    md("---\n## Export all (skip variants without checkpoint)"),
    code("export_all_trained()"),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}")
