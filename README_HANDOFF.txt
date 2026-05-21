================================================================================
EEE429 Final — SW → Team Code Handoff
================================================================================

이 폴더를 팀 repo / zip으로 공유하면 됩니다.
노트북 실행 → 학습(선택) → export → artifacts/hls/ 에 weights.h 생성.

--------------------------------------------------------------------------------
1. 넘길 파일 (zip 구성)
--------------------------------------------------------------------------------

필수 (코드)
  EEE426_FINAL.ipynb       ← 메인: 3 variant 학습 + export
  hls_export.py            ← weights.h / images.h 생성 CLI
  quant_models.py          ← hls_export.py가 import (모델 정의)
  requirements-train.txt   ← PC 학습 환경
  README_HANDOFF.txt       ← 이 파일
  hls/                     HLS C++ starter (fixed16 csim + Vitis TCL)
  hls/README.txt

선택 (이미 학습된 checkpoint — 재학습 생략 가능)
  milestones/              ← fixed16 (QUANTIZED_MNISTCNN-*.pth)
  milestones_bnn/          ← BNN
  milestones_fixed8/       ← 8-bit (학습 후 생성)
  artifacts/hls/           ← 이미 export한 weights.h

넣지 말 것
  .venv/                   ← 팀원이 각자 생성
  data/                    ← MNIST 자동 다운로드 (용량 큼)
  weights_*.h, images.h    ← 루트에 떨어진 구버전 export
  bnn_export_*/            ← 구버전 수동 export
  build_midterm_*.py, EEE426_QUANT.ipynb, FINAL_POL26.ipynb

--------------------------------------------------------------------------------
2. 환경 세팅 (Mac / PC)
--------------------------------------------------------------------------------

  cd Final/
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements-train.txt
  pip install numpy   # hls_export용
  jupyter notebook EEE426_FINAL.ipynb

--------------------------------------------------------------------------------
3. 사용법 (노트북)
--------------------------------------------------------------------------------

  Section 0  Setup          — import, MNIST loader, device
  Section 1  Shared         — export_shared() → 테스트 이미지
  Section 2  Fixed 16/7     — QAT 학습 / eval  (이미 됐으면 skip)
  Section 3  Fixed 8/4      — 8-bit QAT 학습 / eval
  Section 4  BNN            — BNN 학습 / eval
  Section 5  Export         — export_all_trained() 또는 variant별 export

터미널 only:
  python hls_export.py all

--------------------------------------------------------------------------------
4. Export 산출물 (EE/HLS 팀용)
--------------------------------------------------------------------------------

  artifacts/hls/shared/
    images.h, MNIST_DATASET_IMAGE.npy, MNIST_DATASET_LABEL.npy

  artifacts/hls/fixed16/weights.h    ap_fixed<16,7>
  artifacts/hls/fixed8/weights.h     ap_fixed<8,4>   (Section 3 학습 후)
  artifacts/hls/bnn/weights.h        int8 conv ±1 + float BN/FC

각 variant 폴더의 npy/ — Python 코시뮬 디버깅용 (HLS 빌드엔 weights.h만 필요)

--------------------------------------------------------------------------------
5. HW 스펙 (EE 팀 필독)
--------------------------------------------------------------------------------

전처리: ToTensor + Normalize(0.5, 0.5) → float [-1, +1]

Fixed 16/7 & 8/4:
  Input → Q(ap_fixed) → Conv → Q → ReLU → … → MaxPool → Q → FC

BNN:
  Input → sign(±1) → Conv(±1) → BN → sign → … → MaxPool → FC(float32)

Conv weight layout: [out_ch, in_ch, kH, kW]
FC layout: [10, 3872]

최소 정확도: 93% (현재 fixed16 ~99%, BNN ~98%)

================================================================================
