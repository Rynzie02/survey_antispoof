#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
  PYTHON_BIN=(uv run python)
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN=("$SCRIPT_DIR/.venv/bin/python")
else
  PYTHON_BIN=(python)
fi

AUDIO_BASE_DIR="/mnt/data/wht/antispoof/audio_diffattack_20260420_234725"
RESULT_JSON="$SCRIPT_DIR/results/logs/diffattack_attack_results_20260420_234725.json"

echo "[run_repeated_purification_study] workdir=$SCRIPT_DIR"
echo "[run_repeated_purification_study] python=${PYTHON_BIN[*]}"
echo "[run_repeated_purification_study] audio_base_dir=$AUDIO_BASE_DIR"
echo "[run_repeated_purification_study] result_json=$RESULT_JSON"

"${PYTHON_BIN[@]}" analyze_repeated_purification.py \
  --audio-base-dir "$AUDIO_BASE_DIR" \
  --result-json "$RESULT_JSON"
