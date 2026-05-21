"""Three quantization candidates: fixed16, fixed8, BNN — same CNN topology."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── fixed-point helpers ──────────────────────────────────────────────────────

def make_fixed_round(fixed_w: int, fixed_i: int):
    fixed_f = fixed_w - fixed_i
    scale = 2 ** fixed_f
    vmin = -(2 ** (fixed_i - 1))
    vmax = (2 ** (fixed_i - 1)) - (1 / scale)

    def fixed_point_round(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, vmin, vmax)
        return torch.round(x * scale) / scale

    return fixed_point_round, fixed_w, fixed_i


def make_fixed_quantize(fixed_w: int, fixed_i: int):
    round_fn, w, i = make_fixed_round(fixed_w, fixed_i)

    class FixedPointQuantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return round_fn(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return FixedPointQuantize, round_fn, w, i


def build_fixed_cnn(fixed_w: int, fixed_i: int, class_name: str) -> type[nn.Module]:
    FPQ, _, _, _ = make_fixed_quantize(fixed_w, fixed_i)

    class QuantizedConv2d(nn.Conv2d):
        def forward(self, x):
            qw = FPQ.apply(self.weight)
            qb = FPQ.apply(self.bias) if self.bias is not None else None
            return self._conv_forward(x, qw, qb)

    class QuantizedLinear(nn.Linear):
        def forward(self, x):
            qw = FPQ.apply(self.weight)
            qb = FPQ.apply(self.bias) if self.bias is not None else None
            return F.linear(x, qw, qb)

    class FixedMNISTCNN(nn.Module):
        def forward(self, x):
            x = FPQ.apply(x)
            x = self.conv1(x)
            x = FPQ.apply(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = FPQ.apply(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = FPQ.apply(x)
            x = F.relu(x)
            x = self.pool(x)
            x = FPQ.apply(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    FixedMNISTCNN.__name__ = class_name
    FixedMNISTCNN.__qualname__ = class_name

    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = QuantizedConv2d(1, 16, 3, 1, bias=True)
        self.conv2 = QuantizedConv2d(16, 32, 3, 1, bias=True)
        self.conv3 = QuantizedConv2d(32, 32, 3, 1, bias=True)
        self.pool = nn.MaxPool2d(2)
        self.fc = QuantizedLinear(3872, 10, bias=True)

    FixedMNISTCNN.__init__ = __init__
    return FixedMNISTCNN


QUANTIZED16_MNISTCNN = build_fixed_cnn(16, 7, "QUANTIZED16_MNISTCNN")
QUANTIZED8_MNISTCNN = build_fixed_cnn(8, 4, "QUANTIZED8_MNISTCNN")


# ── BNN ──────────────────────────────────────────────────────────────────────

def binary_sign(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).float() * 2 - 1


class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return binary_sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).float()


class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        w_b = BinaryQuantize.apply(self.weight)
        return self._conv_forward(x, w_b, self.bias)


class BINARY_MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BinaryConv2d(1, 16, 3, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.1)
        self.conv2 = BinaryConv2d(16, 32, 3, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv3 = BinaryConv2d(32, 32, 3, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32, momentum=0.1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(3872, 10, bias=True)

    def forward(self, x):
        x = BinaryQuantize.apply(x)
        x = self.bn1(self.conv1(x))
        x = BinaryQuantize.apply(x)
        x = self.bn2(self.conv2(x))
        x = BinaryQuantize.apply(x)
        x = self.bn3(self.conv3(x))
        x = BinaryQuantize.apply(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


def calibrate_bn(model: nn.Module, loader, device, passes: int = 1) -> None:
    """Refresh BatchNorm running stats before eval (fixes BNN test-acc swings)."""
    was_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            for images, _ in loader:
                model(images.to(device))
    model.train(was_training)


# ── variant registry ─────────────────────────────────────────────────────────

VARIANTS = {
    "fixed16": {
        "model_cls": QUANTIZED16_MNISTCNN,
        "model_name": "QUANTIZED16_MNISTCNN",
        "milestone_dir": "milestones_fixed16",
        "fixed_w": 16,
        "fixed_i": 7,
        "kind": "fixed",
    },
    "fixed8": {
        "model_cls": QUANTIZED8_MNISTCNN,
        "model_name": "QUANTIZED8_MNISTCNN",
        "milestone_dir": "milestones_fixed8",
        "fixed_w": 8,
        "fixed_i": 4,
        "kind": "fixed",
    },
    "bnn": {
        "model_cls": BINARY_MNISTCNN,
        "model_name": "BINARY_MNISTCNN",
        "milestone_dir": "milestones_bnn",
        "fixed_w": None,
        "fixed_i": None,
        "kind": "bnn",
    },
}
