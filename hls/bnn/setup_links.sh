#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ART="$ROOT/../../artifacts/hls"

ln -sf "$ART/bnn/weights.h" "$ROOT/weights.h"
ln -sf "$ART/shared/images.h" "$ROOT/images.h"

echo "linked:"
ls -la "$ROOT/weights.h" "$ROOT/images.h"
