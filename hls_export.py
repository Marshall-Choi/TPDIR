"""
HLS artifact export for fixed16 / fixed8 / BNN variants.

Output layout:
  artifacts/hls/shared/          images.h, MNIST_DATASET_*.npy
  artifacts/hls/fixed16/         weights.h, npy/, README.txt
  artifacts/hls/fixed8/          weights.h, npy/, README.txt
  artifacts/hls/bnn/             weights.h, npy/, README.txt
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from quant_models import VARIANTS, binary_sign, make_fixed_round

ROOT = Path(__file__).resolve().parent
HLS_ROOT = ROOT / "artifacts" / "hls"


def hls_dir(variant: str) -> Path:
    return HLS_ROOT / variant


# legacy checkpoint names (EEE426_FINAL.ipynb used QUANTIZED_MNISTCNN for 16-bit)
_CHECKPOINT_ALIASES: dict[str, list[str]] = {
    "fixed16": ["QUANTIZED16_MNISTCNN", "QUANTIZED_MNISTCNN"],
    "fixed8": ["QUANTIZED8_MNISTCNN"],
    "bnn": ["BINARY_MNISTCNN"],
}

_SEARCH_DIRS: dict[str, list[str]] = {
    "fixed16": ["milestones_fixed16", "milestones", "milestones_qat"],
    "fixed8": ["milestones_fixed8"],
    "bnn": ["milestones_bnn"],
}


def latest_checkpoint(variant: str) -> Path:
    prefixes = _CHECKPOINT_ALIASES.get(variant, [VARIANTS[variant]["model_name"]])
    dirs = _SEARCH_DIRS.get(variant, [VARIANTS[variant]["milestone_dir"]])
    candidates: list[Path] = []
    for dname in dirs:
        d = ROOT / dname
        if not d.is_dir():
            continue
        for prefix in prefixes:
            candidates.extend(d.glob(f"{prefix}-*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for {variant}. Searched {dirs} with prefixes {prefixes}."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _make_c_name(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    return name.upper()


def _nested_format(obj, fmt, indent=0):
    sp = "\t" * indent
    if isinstance(obj, list):
        if not obj:
            return "{}"
        if not isinstance(obj[0], list):
            return "{" + ", ".join(fmt(x) for x in obj) + "}"
        inner = tuple(_nested_format(x, fmt, indent + 1) for x in obj)
        return "{\n" + ",\n".join("\t" * (indent + 1) + s for s in inner) + "\n" + sp + "}"
    return fmt(obj)


def _load_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    raise KeyError(f"No model weights in {ckpt_path.name} (keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)})")


def _tensor_to_int_grid(t: torch.Tensor, fixed_w: int, fixed_i: int) -> np.ndarray:
    round_fn, _, _ = make_fixed_round(fixed_w, fixed_i)
    q = round_fn(t.detach().cpu().float())
    scale = 2 ** (fixed_w - fixed_i)
    qi = torch.round(q * scale).to(torch.int32)
    bits = fixed_w
    lo, hi = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    qi = torch.clamp(qi, lo, hi)
    dtype = np.int16 if bits > 8 else np.int8
    return qi.numpy().astype(dtype)


# ── shared (images + labels) ─────────────────────────────────────────────────

def export_shared(data_root: Path | None = None, image_h_count: int = 280) -> Path:
    out = hls_dir("shared")
    out.mkdir(parents=True, exist_ok=True)
    data_root = data_root or ROOT / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_d = datasets.MNIST(str(data_root), train=True, download=True, transform=transform)
    test_d = datasets.MNIST(str(data_root), train=False, download=True, transform=transform)

    buckets: list[list] = [[] for _ in range(10)]
    for img, lab in train_d:
        buckets[int(lab)].append(img)
    for img, lab in test_d:
        buckets[int(lab)].append(img)

    # images.h — 28 per digit
    nested = [buckets[i % 10][i // 10].tolist() for i in range(image_h_count)]
    lines = [
        "#pragma once",
        "#include <cstddef>",
        "",
        f"// IMAGES shape=({image_h_count}, 1, 28, 28), dtype=float, Normalize(0.5,0.5)",
        f"#define IMAGE_NUM {image_h_count}",
        f"static const float IMAGES[{image_h_count}][1][28][28] = "
        f"{_nested_format(nested, lambda x: f'{repr(float(x))}f', 0)};",
    ]
    (out / "images.h").write_text("\n".join(lines), encoding="utf-8")

    # full MNIST npy — digit-sorted
    images, labels = [], []
    for digit, items in enumerate(buckets):
        for t in items:
            images.append(t.numpy())
            labels.append(digit)
    np.save(out / "MNIST_DATASET_IMAGE.npy", np.stack(images, axis=0))
    np.save(out / "MNIST_DATASET_LABEL.npy", np.array(labels, dtype=np.int64))

    readme = "\n".join([
        "Shared HLS test vectors (all variants use the same input preprocessing).",
        "",
        "images.h              — 280 float samples [N][1][28][28]",
        "MNIST_DATASET_IMAGE.npy — (70000, 1, 28, 28) float32, digit-sorted",
        "MNIST_DATASET_LABEL.npy — (70000,) int64",
        "Preprocessing: ToTensor + Normalize(mean=0.5, std=0.5) → [-1, +1]",
    ])
    (out / "README.txt").write_text(readme + "\n", encoding="utf-8")
    print(f"[shared] → {out.resolve()}")
    return out


# ── fixed-point (16 / 8 bit) ─────────────────────────────────────────────────

def export_fixed_variant(
    variant: str,
    ckpt_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    if variant not in ("fixed16", "fixed8"):
        raise ValueError("variant must be fixed16 or fixed8")
    cfg = VARIANTS[variant]
    fw, fi = cfg["fixed_w"], cfg["fixed_i"]
    ckpt_path = ckpt_path or latest_checkpoint(variant)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = out_dir or hls_dir(variant)
    npy_dir = out_dir / "npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(exist_ok=True)

    round_fn, _, _ = make_fixed_round(fw, fi)
    sd = _load_state_dict(ckpt_path)
    type_name = "fixed_t"
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        '#include "ap_fixed.h"',
        "",
        f"// Source: {ckpt_path.name}",
        f"// ap_fixed<{fw},{fi}>  F={fw-fi}  scale=2^{fw-fi}",
        f"using {type_name} = ap_fixed<{fw}, {fi}, AP_RND, AP_SAT>;",
        "",
    ]

    def fmt(x):
        return f"{repr(float(x))}f"

    readme_rows = [f"Variant: {variant}  ap_fixed<{fw},{fi}>", f"Checkpoint: {ckpt_path.name}", ""]

    for key, tsr in sd.items():
        cname = _make_c_name(key)
        tsr = round_fn(tsr.detach().cpu().float().contiguous())
        shape = tuple(tsr.shape)
        dims = "".join(f"[{d}]" for d in shape)
        nested = tsr.tolist()

        lines.extend([
            f"// {cname} shape={shape}",
            *[f"#define {cname}_DIM{i} {d}" for i, d in enumerate(shape)],
            f"static const {type_name} {cname}{dims} = {_nested_format(nested, fmt, 0)};",
            "",
        ])

        safe = key.replace(".", "_")
        np.save(npy_dir / f"{safe}_float32.npy", tsr.numpy().astype(np.float32))
        np.save(npy_dir / f"{safe}_int.npy", _tensor_to_int_grid(tsr, fw, fi))
        readme_rows.append(f"  {key:20s} {shape}")

    (out_dir / "weights.h").write_text("\n".join(lines), encoding="utf-8")
    readme_rows.extend(["", "weights.h — include in HLS project", "npy/ — co-simulation"])
    (out_dir / "README.txt").write_text("\n".join(readme_rows) + "\n", encoding="utf-8")

    print(f"[{variant}] checkpoint: {ckpt_path.name}")
    print(f"[{variant}] → {out_dir.resolve()}/weights.h")
    return out_dir


# ── BNN ──────────────────────────────────────────────────────────────────────

def export_bnn_variant(
    ckpt_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    ckpt_path = ckpt_path or latest_checkpoint("bnn")
    out_dir = out_dir or hls_dir("bnn")
    npy_dir = out_dir / "npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(exist_ok=True)

    sd = _load_state_dict(ckpt_path)
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "",
        f"// Source: {ckpt_path.name}",
        "// BNN: conv weights int8 {-1,+1}, BatchNorm + FC float32",
        "",
    ]
    readme_rows = [f"Variant: bnn", f"Checkpoint: {ckpt_path.name}", ""]

    for key, tsr in sd.items():
        cname = _make_c_name(key)
        t = tsr.detach().cpu().float().contiguous()
        shape = tuple(t.shape)
        dims = "".join(f"[{d}]" for d in shape)
        safe = key.replace(".", "_")
        is_bin_w = "conv" in key and "weight" in key

        if is_bin_w:
            t_bin = binary_sign(t).to(torch.int8)
            nested = t_bin.tolist()
            lines.extend([
                f"// {cname} shape={shape} dtype=int8 binary {{-1,+1}}",
                *[f"#define {cname}_DIM{i} {d}" for i, d in enumerate(shape)],
                f"static const int8_t {cname}{dims} = {_nested_format(nested, str, 0)};",
                "",
            ])
            np.save(npy_dir / f"{safe}_int8.npy", t_bin.numpy())
            readme_rows.append(f"  {key:20s} {shape}  int8 binary")
        else:
            nested = t.tolist()
            lines.extend([
                f"// {cname} shape={shape} dtype=float32",
                *[f"#define {cname}_DIM{i} {d}" for i, d in enumerate(shape)],
                f"static const float {cname}{dims} = {_nested_format(nested, lambda x: f'{repr(float(x))}f', 0)};",
                "",
            ])
            np.save(npy_dir / f"{safe}_float32.npy", t.numpy())
            tag = "BN" if "bn" in key else "FC"
            readme_rows.append(f"  {key:20s} {shape}  float32 [{tag}]")

    (out_dir / "weights.h").write_text("\n".join(lines), encoding="utf-8")
    readme_rows.extend(["", "weights.h — include in HLS project", "npy/ — co-simulation"])
    (out_dir / "README.txt").write_text("\n".join(readme_rows) + "\n", encoding="utf-8")

    print(f"[bnn] checkpoint: {ckpt_path.name}")
    print(f"[bnn] → {out_dir.resolve()}/weights.h")
    return out_dir


def export_variant(variant: str, ckpt_path: Path | None = None) -> Path:
    if variant == "shared":
        return export_shared()
    if variant in ("fixed16", "fixed8"):
        return export_fixed_variant(variant, ckpt_path)
    if variant == "bnn":
        return export_bnn_variant(ckpt_path)
    raise ValueError(f"Unknown variant: {variant}. Use fixed16|fixed8|bnn|shared")


def export_all_trained() -> dict[str, Path]:
    results = {"shared": export_shared()}
    for v in ("fixed16", "fixed8", "bnn"):
        try:
            results[v] = export_variant(v)
        except FileNotFoundError as e:
            print(f"[{v}] skip — {e}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export HLS artifacts")
    ap.add_argument(
        "variant",
        nargs="?",
        choices=["fixed16", "fixed8", "bnn", "shared", "all"],
        default="all",
    )
    args = ap.parse_args()
    if args.variant == "all":
        export_all_trained()
    else:
        export_variant(args.variant)
