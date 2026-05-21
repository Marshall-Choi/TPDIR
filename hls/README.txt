================================================================================
HLS Starter — MNIST CNN
================================================================================

  hls/fixed16/   ap_fixed<16,7> QAT  → top: mnist_cnn
  hls/bnn/       ±1 conv + BN + FC   → top: mnist_bnn   ← 프로젝트 메인

공통 준비
---------
  python hls_export.py shared
  python hls_export.py bnn          # 또는 fixed16

BNN (메인)
----------
  cd hls/bnn && bash setup_links.sh
  bash run_csim.sh                  # Mac C-sim (~98%)
  vitis_hls -f run_vitis_hls.tcl    # Windows: csim + synthesis

Fixed16 (비교/연습)
-------------------
  cd hls/fixed16 && bash setup_links.sh
  bash run_csim.sh
  vitis_hls -f run_vitis_hls.tcl

Part: xc7z020clg400-1 (Pynq-Z2)

BNN 그래프
----------
  Normalize input [-1,1]
  → sign(±1) → bin_conv → BN(float) → sign → … → MaxPool → FC(float)

================================================================================
