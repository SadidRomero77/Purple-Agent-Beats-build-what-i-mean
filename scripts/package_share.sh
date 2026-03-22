#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_NAME="agente_purpura_v2_shareable.zip"

cd "$ROOT_DIR/.."
rm -f "$OUT_NAME"
zip -r "$OUT_NAME" "$(basename "$ROOT_DIR")" -x "*/__pycache__/*" "*.pyc" >/dev/null

echo "Zip generado: $(pwd)/$OUT_NAME"
