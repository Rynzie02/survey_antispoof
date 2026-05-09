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

DATA_ROOT="${DATA_ROOT:-/mnt/wht/exp/test_900}"
DATA_ROOTS="${DATA_ROOTS:-}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/wht/exp/dual_attack_outputs/test_900_protected}"
if [[ -z "${ATTACK_TYPE:-}" ]]; then
  ATTACK_TYPE="$("${PYTHON_BIN[@]}" - <<'PY'
from config import config
print(config.attack_type)
PY
)"
fi
RUN_NAME="${RUN_NAME:-${ATTACK_TYPE}_${RUN_ID}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
SHARD_MODE="${SHARD_MODE:-round_robin}"
SHARD_DIR="${SHARD_DIR:-$RUN_OUTPUT_DIR/shards/$RUN_ID}"
LOG_DIR_BASE="${LOG_DIR:-$RUN_OUTPUT_DIR/logs}"
AUDIO_OUTPUT_DIR_BASE="${AUDIO_OUTPUT_DIR:-$RUN_OUTPUT_DIR/protected}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-$RUN_OUTPUT_DIR/worker_logs/$RUN_ID}"
SAVE_AUDIO="${SAVE_AUDIO:-1}"
AUDIO_OUTPUT_INCLUDE_RUN_TS="${AUDIO_OUTPUT_INCLUDE_RUN_TS:-0}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -eq 0 || -z "${GPU_LIST[0]}" ]]; then
  echo "[run_diffattack] ERROR: no GPUs specified. Example: GPUS=0,1,2 $0" >&2
  exit 1
fi

mkdir -p "$SHARD_DIR" "$LOG_DIR_BASE" "$AUDIO_OUTPUT_DIR_BASE" "$WORKER_LOG_DIR"

echo "[run_diffattack] workdir=$SCRIPT_DIR"
echo "[run_diffattack] python=${PYTHON_BIN[*]}"
echo "[run_diffattack] data_root=$DATA_ROOT"
if [[ -n "$DATA_ROOTS" ]]; then
  echo "[run_diffattack] data_roots=$DATA_ROOTS"
fi
echo "[run_diffattack] gpus=$GPUS"
echo "[run_diffattack] attack_type=$ATTACK_TYPE"
echo "[run_diffattack] run_id=$RUN_ID"
echo "[run_diffattack] run_output_dir=$RUN_OUTPUT_DIR"
echo "[run_diffattack] audio_output_dir=$AUDIO_OUTPUT_DIR_BASE"
echo "[run_diffattack] shard_mode=$SHARD_MODE"

export DATA_ROOT DATA_ROOTS GPUS NUM_SAMPLES SHARD_DIR SHARD_MODE
"${PYTHON_BIN[@]}" - <<'PY'
import math
import os
from pathlib import Path

audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
data_root = os.environ["DATA_ROOT"]
data_roots = [
    item.strip()
    for item in os.environ.get("DATA_ROOTS", "").split(",")
    if item.strip()
]
gpus = [gpu.strip() for gpu in os.environ["GPUS"].split(",") if gpu.strip()]
num_samples = int(os.environ.get("NUM_SAMPLES") or 0)
shard_dir = Path(os.environ["SHARD_DIR"])
shard_mode = os.environ.get("SHARD_MODE", "round_robin")

if data_roots and len(data_roots) != len(gpus):
    raise SystemExit(
        "[run_diffattack] ERROR: DATA_ROOTS must have the same number of entries as GPUS"
    )


def collect_audio(root_value):
    root = Path(root_value).expanduser()
    paths = []
    if root.is_file():
        with root.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw_path = line.split("|", 1)[0].strip()
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = (root.parent / path).resolve()
                if path.suffix.lower() in audio_exts:
                    paths.append(str(path))
    elif root.is_dir():
        paths = [
            str(path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in audio_exts
        ]
    else:
        raise SystemExit(f"[run_diffattack] ERROR: data root does not exist: {root}")
    return sorted(paths)

shard_dir.mkdir(parents=True, exist_ok=True)
for old in shard_dir.glob("shard_*.txt"):
    old.unlink()

workers = len(gpus)
shards = [[] for _ in range(workers)]
if data_roots:
    for idx, root_value in enumerate(data_roots):
        shard_paths = collect_audio(root_value)
        if num_samples > 0:
            shard_paths = shard_paths[:num_samples]
        shards[idx] = shard_paths
    total = sum(len(shard) for shard in shards)
else:
    paths = collect_audio(data_root)
    if num_samples > 0:
        paths = paths[:num_samples]
    if not paths:
        raise SystemExit(f"[run_diffattack] ERROR: no audio files found under: {data_root}")
    if shard_mode == "contiguous":
        chunk = math.ceil(len(paths) / workers)
        for idx in range(workers):
            shards[idx] = paths[idx * chunk : (idx + 1) * chunk]
    elif shard_mode == "speaker":
        speaker_groups = {}
        for path in paths:
            speaker = Path(path).parent.name
            speaker_groups.setdefault(speaker, []).append(path)
        for group_idx, speaker in enumerate(sorted(speaker_groups)):
            shards[group_idx % workers].extend(speaker_groups[speaker])
    else:
        for idx, path in enumerate(paths):
            shards[idx % workers].append(path)
    total = len(paths)

if total == 0:
    raise SystemExit("[run_diffattack] ERROR: no audio files found")

print(f"[run_diffattack] discovered {total} audio files")
for idx, shard in enumerate(shards):
    shard_file = shard_dir / f"shard_{idx}.txt"
    with shard_file.open("w", encoding="utf-8") as handle:
        for path in shard:
            handle.write(f"{path}\n")
    print(f"[run_diffattack] shard {idx}: gpu={gpus[idx]} files={len(shard)} file={shard_file}")
PY

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  echo "[run_diffattack] dry run complete; not starting workers"
  exit 0
fi

pids=()
for idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$idx]}"
  shard_file="$SHARD_DIR/shard_${idx}.txt"
  worker_log="$WORKER_LOG_DIR/gpu${gpu}_worker${idx}.log"

  if [[ ! -s "$shard_file" ]]; then
    echo "[run_diffattack] skip worker=$idx gpu=$gpu because shard is empty"
    continue
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export DATA_ROOT="$shard_file"
    export NUM_SAMPLES=0
    export WORKER_INDEX="$idx"
    export PHYSICAL_GPU_ID="$gpu"
    export LOG_DIR="$LOG_DIR_BASE/gpu${gpu}_worker${idx}"
    export AUDIO_OUTPUT_DIR="$AUDIO_OUTPUT_DIR_BASE"
    export SAVE_AUDIO="$SAVE_AUDIO"
    export AUDIO_OUTPUT_INCLUDE_RUN_TS="$AUDIO_OUTPUT_INCLUDE_RUN_TS"
    export ATTACK_TYPE="$ATTACK_TYPE"
    "${PYTHON_BIN[@]}" - <<'PY'
import os

from config import config
from main import run_experiment

config.use_targeted = False
config.gpu_id = 0
config.device = __import__("torch").device("cuda:0" if __import__("torch").cuda.is_available() else "cpu")

env_overrides = {
    "DATA_ROOT": ("data_root", str),
    "NUM_SAMPLES": ("num_samples", int),
    "BATCH_SIZE": ("batch_size", int),
    "NUM_ITERATIONS": ("num_iterations", int),
    "AUDIO_LENGTH": ("audio_length", float),
    "SPEAKER_MODEL_TYPE": ("speaker_model_type", str),
    "CAMPP_MODEL_PATH": ("campp_model_path", str),
    "STAGED_DIRECT_RATIO": ("staged_direct_ratio", float),
    "PURIFICATION_REVERSE_TIMESTEP": ("purification_reverse_timestep", int),
    "PURIFICATION_STEP_STRIDE": ("purification_step_stride", int),
    "ATTACK_TYPE": ("attack_type", str),
    "SAVE_AUDIO": ("save_audio", lambda v: v.lower() in ("1", "true", "yes", "on")),
    "AUDIO_OUTPUT_DIR": ("audio_output_dir", str),
    "AUDIO_OUTPUT_INCLUDE_RUN_TS": ("audio_output_include_run_ts", lambda v: v.lower() in ("1", "true", "yes", "on")),
    "LOG_DIR": ("log_dir", str),
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

print("[run_diffattack] worker:")
print(f"  worker_index={os.environ.get('WORKER_INDEX')}")
print(f"  physical_gpu_id={os.environ.get('PHYSICAL_GPU_ID')}")
print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("[run_diffattack] config:")
print(f"  device={config.device}")
print(f"  data_root={config.data_root}")
print(f"  num_samples={config.num_samples}")
print(f"  batch_size={config.batch_size}")
print(f"  epsilon={config.epsilon}")
print(f"  num_iterations={config.num_iterations}")
print(f"  step_size={config.step_size}")
print(f"  speaker_model_type={config.speaker_model_type}")
print(f"  campp_model_path={getattr(config, 'campp_model_path', None)}")
print(f"  alpha={config.alpha}")
print(f"  beta={config.beta}")
print(f"  purification_reverse_timestep={config.purification_reverse_timestep}")
print(f"  purification_step_stride={config.purification_step_stride}")
print(f"  diffattack_t_interval={getattr(config, 'diffattack_t_interval', 1)}")
print(f"  log_dir={config.log_dir}")
print(f"  save_audio={config.save_audio}")
print(f"  audio_output_dir={config.audio_output_dir}")
print(f"  audio_output_include_run_ts={config.audio_output_include_run_ts}")

metrics, success = run_experiment(config, attack_type=config.attack_type)
print("[run_diffattack] finished")
print(f"[run_diffattack] success={success}")
print(f"[run_diffattack] metrics={metrics}")
PY
  ) >"$worker_log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "[run_diffattack] started worker=$idx gpu=$gpu pid=$pid log=$worker_log"
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "[run_diffattack] ERROR: no workers were started" >&2
  exit 1
fi

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[run_diffattack] one or more workers failed. Logs are in $WORKER_LOG_DIR" >&2
  exit 1
fi

echo "[run_diffattack] all workers finished"
echo "[run_diffattack] worker logs: $WORKER_LOG_DIR"
echo "[run_diffattack] result logs: $LOG_DIR_BASE"
echo "[run_diffattack] audio outputs: $AUDIO_OUTPUT_DIR_BASE"
