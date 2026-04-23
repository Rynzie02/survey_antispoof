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

echo "[run_diffattack] workdir=$SCRIPT_DIR"
echo "[run_diffattack] python=${PYTHON_BIN[*]}"
echo "[run_diffattack] using defaults from config.py"

"${PYTHON_BIN[@]}" - <<'PY'
from config import config
from main import run_experiment

config.use_targeted = False

print("[run_diffattack] config:")
print(f"  device={config.device}")
print(f"  data_root={config.data_root}")
print(f"  num_samples={config.num_samples}")
print(f"  batch_size={config.batch_size}")
print(f"  epsilon={config.epsilon}")
print(f"  num_iterations={config.num_iterations}")
print(f"  step_size={config.step_size}")
print(f"  alpha={config.alpha}")
print(f"  beta={config.beta}")
print(f"  purification_reverse_timestep={config.purification_reverse_timestep}")
print(f"  purification_step_stride={config.purification_step_stride}")
print(f"  diffattack_t_interval={getattr(config, 'diffattack_t_interval', 1)}")
print(f"  log_dir={config.log_dir}")

metrics, success = run_experiment(config, attack_type="diffattack")
print("[run_diffattack] finished")
print(f"[run_diffattack] success={success}")
print(f"[run_diffattack] metrics={metrics}")
PY
