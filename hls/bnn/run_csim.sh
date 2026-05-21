#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

bash setup_links.sh

CXX="${CXX:-g++}"
FLAGS=(-std=c++17 -O2 -DCSIM_LOCAL -I. -I../common)

echo "=== build BNN csim ==="
"$CXX" "${FLAGS[@]}" cnn_forward.cpp tb_csim.cpp -o tb_csim

echo "=== run BNN csim ==="
./tb_csim
