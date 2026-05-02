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
import os

from config import config
from main import run_experiment

config.use_targeted = False

if "BATCH_SIZE" not in os.environ and config.batch_size > 1:
    config.batch_size = 1
    print("[run_diffattack] defaulting batch_size=1 for fulltrace GPU memory safety")

env_overrides = {
    "DATA_ROOT": ("data_root", str),
    "NUM_SAMPLES": ("num_samples", int),
    "BATCH_SIZE": ("batch_size", int),
    "NUM_ITERATIONS": ("num_iterations", int),
    "AUDIO_LENGTH": ("audio_length", float),
    "STAGED_DIRECT_RATIO": ("staged_direct_ratio", float),
    "PURIFICATION_REVERSE_TIMESTEP": ("purification_reverse_timestep", int),
    "PURIFICATION_STEP_STRIDE": ("purification_step_stride", int),
    "SAVE_AUDIO": ("save_audio", lambda v: v.lower() in ("1", "true", "yes", "on")),
    "AUDIO_OUTPUT_DIR": ("audio_output_dir", str),
}
updates = {}
for env_name, (attr, caster) in env_overrides.items():
    raw = os.environ.get(env_name)
    if raw is not None and raw != "":
        updates[attr] = caster(raw)
if updates:
    config.update(**updates)
    print("[run_diffattack] env overrides:")
    for key, value in updates.items():
        print(f"  {key}={value}")

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
print(f"  save_audio={config.save_audio}")
print(f"  audio_output_dir={config.audio_output_dir}")

metrics, success = run_experiment(config, attack_type=config.attack_type)
print("[run_diffattack] finished")
print(f"[run_diffattack] success={success}")
print(f"[run_diffattack] metrics={metrics}")
PY
