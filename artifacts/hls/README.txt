HLS artifact layout (EEE429 Final)
==================================

Run from project root (Final/).

Notebooks
---------
  EEE426_FINAL.ipynb     — main: Fixed16 + Fixed8 + BNN + ImageEmbed + export
  EEE426_QUANT.ipynb     — (duplicate layout; use FINAL instead)

CLI
---
  python hls_export.py shared
  python hls_export.py fixed16
  python hls_export.py fixed8
  python hls_export.py bnn
  python hls_export.py all

Directories
-----------
  artifacts/hls/shared/
    images.h
    MNIST_DATASET_IMAGE.npy
    MNIST_DATASET_LABEL.npy

  artifacts/hls/fixed16/
    weights.h              ap_fixed<16,7>
    npy/                   per-layer float32 + int16 grid
    README.txt

  artifacts/hls/fixed8/
    weights.h              ap_fixed<8,4>
    npy/                   per-layer float32 + int8 grid
    README.txt

  artifacts/hls/bnn/
    weights.h              int8 conv weights + float BN/FC
    npy/
    README.txt

Checkpoints (training)
----------------------
  milestones_fixed16/  or milestones/  (QUANTIZED16_MNISTCNN / QUANTIZED_MNISTCNN)
  milestones_fixed8/  QUANTIZED8_MNISTCNN
  milestones_bnn/       BINARY_MNISTCNN
