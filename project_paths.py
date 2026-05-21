"""EEE429 Final 프로젝트 루트 경로 (노트북이 reference/lab6 아래 있어도 동작)."""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("requirements-train.txt", "FINAL_POL26.ipynb")


def project_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if not (p / "data" / "MNIST").is_dir():
            continue
        if any((p / m).is_file() for m in _MARKERS):
            return p
    return cwd


def data_dir() -> Path:
    return project_root() / "data"


def artifacts_lab6() -> Path:
    p = project_root() / "artifacts" / "lab6"
    p.mkdir(parents=True, exist_ok=True)
    return p


def milestones_qat_dir() -> Path:
    p = project_root() / "milestones_qat"
    p.mkdir(parents=True, exist_ok=True)
    return p
