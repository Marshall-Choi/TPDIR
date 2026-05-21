#!/usr/bin/env bash
# Symlink exported SW artifacts into this HLS folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ART="$ROOT/../../artifacts/hls"

ln -sf "$ART/fixed16/weights.h" "$ROOT/weights.h"
ln -sf "$ART/shared/images.h" "$ROOT/images.h"

echo "linked:"
ls -la "$ROOT/weights.h" "$ROOT/images.h"
