================================================================================
HLS Starter — MNIST CNN (fixed16)
================================================================================

SW export (weights.h, images.h) 를 HLS C++ 추론 코드에 연결하는 스타터 프로젝트.

디렉토리
--------
  hls/common/           공통 차원·ap_fixed stub·quantize
  hls/fixed16/          fixed16 추론 + testbench + Vitis TCL

준비 (최초 1회)
---------------
  1. SW 쪽 export (프로젝트 루트):
       python hls_export.py shared
       python hls_export.py fixed16

  2. symlink:
       cd hls/fixed16 && bash setup_links.sh

Mac / PC — C simulation (Vivado 없이)
--------------------------------------
  cd hls/fixed16
  bash run_csim.sh

  → images.h 280장으로 정확도 출력 (90% 이상이면 PASS)

Vitis HLS (합성·IP export)
--------------------------
  cd hls/fixed16
  vitis_hls -f run_vitis_hls.tcl

  Pynq-Z2 part: xc7z020clg400-1 (run_vitis_hls.tcl 에 설정됨)

파일 설명
---------
  cnn_forward.cpp   QAT 그래프: Input→Q→Conv→Q→ReLU ×3 → Pool → Q → FC
  top_kernel.cpp    HLS top `mnist_cnn()` — AXI interface
  tb_csim.cpp       images.h testbench
  weights.h         symlink → artifacts/hls/fixed16/weights.h
  images.h          symlink → artifacts/hls/shared/images.h

다음 단계 (최적화)
------------------
  - conv/FC loop tiling, DATAFLOW 파이프라인
  - fixed8 / BNN variant 별도 subdir 추가
  - cosim_design 로 RTL 검증

================================================================================
