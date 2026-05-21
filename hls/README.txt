================================================================================
HLS Starter — MNIST CNN
================================================================================

  hls/fixed8/    ap_fixed<8,4> QAT   → top: mnist_cnn   ← 메인 (빠른 사이클)
  hls/fixed16/   ap_fixed<16,7> QAT  → top: mnist_cnn
  hls/bnn/       ±1 conv + BN + FC   → top: mnist_bnn   (실험용)

한 사이클 (교수님: end-to-end 먼저)
------------------------------------
  python train_fixed8.py
  cd hls/fixed8 && bash run_csim.sh
  # EE: vitis_hls -f run_vitis_hls.tcl

공통 준비
---------
  python hls_export.py shared
  python hls_export.py fixed8

Fixed8 (메인)
-------------
  cd hls/fixed8 && bash setup_links.sh
  bash run_csim.sh
  vitis_hls -f run_vitis_hls.tcl    # Windows

Fixed16 (비교/백업)
-------------------
  cd hls/fixed16 && bash setup_links.sh
  bash run_csim.sh

Part: xc7z020clg400-1 (Pynq-Z2)

Graph (fixed8 / fixed16)
------------------------
  Normalize input [-1,1]
  → Q → Conv → Q → ReLU → … → MaxPool → Q → FC

================================================================================
