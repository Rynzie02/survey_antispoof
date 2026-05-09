#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Keep CUDA ordinals consistent with nvidia-smi / PCI bus order across all
# child projects. Without this, CUDA may use FASTEST_FIRST and make cuda:0 map
# to a different physical GPU, such as a faster Pro 6000 at nvidia-smi index 4.
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_DEVICE_ORDER

usage() {
  cat <<'EOF'
Usage:
  ./full.sh

Runs run_exp.sh first, then purifies the generated adv/ audio with AudioPure,
DualPure, WavePurifier, and De-AntiFake PhonePuRe.
Then it runs Chatterbox, F5-TTS, CosyVoice, and Qwen3-TTS voice cloning for
adv and each purified output.

Common environment overrides:
  GPUS=0,1,2,3                         GPUs for run_exp.sh
  DEFAULT_PURIFY_GPU=0                  Default single GPU for all purifiers
  CUDA_DEVICE_ORDER=PCI_BUS_ID          Keep CUDA GPU indexes aligned to PCI/nvidia-smi order
  STRICT_COUNT_CHECK=1                  Require every generated stage count to match adv
  RUN_ID=20260508_025528                Stable run id
  RUN_OUTPUT_DIR=/path/to/run           Override the run output directory
  SKIP_RUN=1 RUN_OUTPUT_DIR=/path/to/run
                                         Purify an existing run directory

AudioPure overrides:
  PURIFY_GPUS=0                         GPU for AudioPure purification
  PURIFY_OUTPUT_DIR=/path/to/output     Default: <RUN_OUTPUT_DIR>/purify/audiopure
  PURIFY_EXPECTED_AUDIO_COUNT=900       Set to auto to use the discovered count
  PURIFY_OVERWRITE=1                    Recompute existing purified files
  PURIFY_DRY_RUN=1                      Print purification jobs without running them
  SKIP_AUDIOPURE=1                      Skip AudioPure purification

DualPure overrides:
  DUALPURE_GPUS=0                       GPU for DualPure purification
  DUALPURE_OUTPUT_DIR=/path/to/output   Default: <RUN_OUTPUT_DIR>/purify/DualPure
  DUALPURE_EXPECTED_COUNT=900           Set to auto, or -1 to disable the count check
  DUALPURE_OVERWRITE=1                  Recompute existing purified files
  DUALPURE_DRY_RUN=1                    Print DualPure jobs without running them

WavePurifier overrides:
  WAVEPURIFIER_GPUS=0                   GPU for WavePurifier purification
  WAVEPURIFIER_OUTPUT_DIR=/path/to/out  Default: <RUN_OUTPUT_DIR>/purify/wavepurifier
  WAVEPURIFIER_EXPECTED_COUNT=900       Set to auto, or -1 to disable the count check
  WAVEPURIFIER_OVERWRITE=1              Recompute existing purified files
  WAVEPURIFIER_DRY_RUN=1                Print WavePurifier jobs without running them

De-AntiFake PhonePuRe overrides:
  PHONEPURE_GPUS=0                      GPU for PhonePuRe purification
  PHONEPURE_OUTPUT_DIR=/path/to/out     Default: <RUN_OUTPUT_DIR>/purify/phonepure
  PHONEPURE_EXPECTED_COUNT=900          Set to auto, or -1 to disable the count check
  PHONEPURE_SKIP_EXISTING=1             Skip completed speaker groups
  PHONEPURE_DRY_RUN=1                   Print PhonePuRe commands without running them

Chatterbox clone overrides:
  CHATTERBOX_GPU=0                      GPU for Chatterbox cloning
  CHATTERBOX_OUTPUT_ROOT=/path/to/out   Default: <RUN_OUTPUT_DIR>/clone/chatterbox
  CHATTERBOX_LIMIT=1000000000           Max refs per input directory
  CHATTERBOX_RECURSIVE=1                Recurse into clone input dirs
  CHATTERBOX_DRY_RUN=1                  Print clone jobs without running them

F5-TTS clone overrides:
  F5TTS_GPU=0                           GPU for F5-TTS cloning
  F5TTS_OUTPUT_ROOT=/path/to/out        Default: <RUN_OUTPUT_DIR>/clone/f5-tts
  F5TTS_TEXT_DIR=/path/to/texts         Ref transcripts; default auto-uses test_900_text if present
  F5TTS_LIMIT=0                         Max refs per input directory; 0 means all
  F5TTS_RECURSIVE=1                     Recurse into clone input dirs
  F5TTS_DRY_RUN=1                       Print clone jobs without running them

CosyVoice clone overrides:
  COSYVOICE_GPU=0                       GPU for CosyVoice cloning
  COSYVOICE_OUTPUT_ROOT=/path/to/out    Default: <RUN_OUTPUT_DIR>/clone/cosyvoice
  COSYVOICE_LIMIT=10                    Optional max refs per input directory
  COSYVOICE_EXPECTED_COUNT=900          Set to auto, or -1 to disable count check
  COSYVOICE_CHECK_ONLY=1                Check inputs and planned outputs only
  COSYVOICE_DRY_RUN=1                   Print clone jobs without running them

Qwen3-TTS clone overrides:
  QWEN3TTS_GPU=0                        GPU for Qwen3-TTS cloning
  QWEN3TTS_OUTPUT_ROOT=/path/to/out     Default: <RUN_OUTPUT_DIR>/clone/qwen3-tts
  QWEN3TTS_LIMIT=10                     Optional max refs per input directory
  QWEN3TTS_EXPECTED_COUNT=900           Set to auto, or -1 to disable count check
  QWEN3TTS_DEVICE=cuda:0                Logical device after CUDA_VISIBLE_DEVICES
  QWEN3TTS_DRY_RUN=1                    Print clone jobs without running them

Clone speaker similarity overrides:
  SIMILARITY_GPU=0                      Single GPU for speaker similarity
  SIMILARITY_CLEAN_DIR=/mnt/wht/exp/test_900
  SIMILARITY_OUTPUT_DIR=/path/to/out    Default: <repo>/eval/<run directory name>
  SIMILARITY_MODELS="ecapa xvector dvector"
  SIMILARITY_EXPECTED_COUNT=900         Set to auto, or -1 to disable count check
  SIMILARITY_DRY_RUN=1                  Match files only; do not load models
  SKIP_SIMILARITY=1                     Skip clone speaker similarity
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

split_csv_or_space() {
  local raw item
  for raw in "$@"; do
    raw="${raw//,/ }"
    for item in $raw; do
      [[ -n "$item" ]] && printf '%s\n' "$item"
    done
  done
}

first_csv_or_space_item() {
  local item
  while IFS= read -r item; do
    printf '%s\n' "$item"
    return 0
  done < <(split_csv_or_space "$@")
  printf '0\n'
}

join_csv_from_values() {
  local first=1 item
  for item in "$@"; do
    if [[ "$first" -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf ',%s' "$item"
    fi
  done
}

count_audio_files() {
  local root="$1"
  shift || true

  local extensions=()
  mapfile -t extensions < <(split_csv_or_space "$@")
  if [[ "${#extensions[@]}" -eq 0 ]]; then
    extensions=(".wav")
  fi

  local count=0 path lower ext normalized_ext
  while IFS= read -r -d '' path; do
    lower="${path,,}"
    for ext in "${extensions[@]}"; do
      normalized_ext="${ext#.}"
      normalized_ext="${normalized_ext,,}"
      if [[ "$lower" == *".${normalized_ext}" ]]; then
        ((count+=1))
        break
      fi
    done
  done < <(find "$root" -type f -print0)

  printf '%s\n' "$count"
}

count_audio_files_direct() {
  local root="$1"
  shift || true

  local extensions=()
  mapfile -t extensions < <(split_csv_or_space "$@")
  if [[ "${#extensions[@]}" -eq 0 ]]; then
    extensions=(".wav")
  fi

  local count=0 path lower ext normalized_ext
  while IFS= read -r -d '' path; do
    lower="${path,,}"
    for ext in "${extensions[@]}"; do
      normalized_ext="${ext#.}"
      normalized_ext="${normalized_ext,,}"
      if [[ "$lower" == *".${normalized_ext}" ]]; then
        ((count+=1))
        break
      fi
    done
  done < <(find "$root" -maxdepth 1 -type f -print0)

  printf '%s\n' "$count"
}

count_clone_audio_files() {
  local root="$1"
  local recursive="$2"
  shift 2 || true

  if is_truthy "$recursive"; then
    count_audio_files "$root" "$@"
  else
    count_audio_files_direct "$root" "$@"
  fi
}

assert_audio_count_matches_base() {
  local label="$1"
  local root="$2"
  local recursive="$3"
  shift 3 || true

  if ! is_truthy "${STRICT_COUNT_CHECK:-1}"; then
    return 0
  fi
  if [[ ! -d "$root" ]]; then
    echo "[full] ERROR: $label output directory does not exist: $root" >&2
    return 1
  fi

  local count
  count="$(count_clone_audio_files "$root" "$recursive" "$@")"
  echo "[full] ${label}_output_count=$count"
  if [[ "$count" -ne "$BASE_AUDIO_COUNT" ]]; then
    echo "[full] ERROR: $label output audio count mismatch: expected initial adv count $BASE_AUDIO_COUNT, found $count under $root." >&2
    return 1
  fi
}

stage_direct_audio_links() {
  local source_dir="$1"
  local stage_dir="$2"
  local stage_label="$3"
  shift 3 || true

  if [[ -z "$stage_dir" || "$stage_dir" == "/" ]]; then
    echo "[full] ERROR: invalid $stage_label staging dir: $stage_dir" >&2
    return 1
  fi
  if [[ -z "$stage_label" || ! "$stage_label" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[full] ERROR: invalid staging label: $stage_label" >&2
    return 1
  fi

  local extensions=()
  mapfile -t extensions < <(split_csv_or_space "$@")
  if [[ "${#extensions[@]}" -eq 0 ]]; then
    extensions=(".wav")
  fi

  local marker="$stage_dir/.full_sh_${stage_label}_stage"
  if [[ -d "$stage_dir" ]]; then
    if [[ ! -f "$marker" ]]; then
      if find "$stage_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "[full] ERROR: refusing to clear unmarked $stage_label staging dir: $stage_dir" >&2
        echo "[full] Set ${stage_label}_INPUT_STAGING_ROOT to an empty/generated directory." >&2
        return 1
      fi
      : >"$marker"
    fi
    find "$stage_dir" -mindepth 1 -maxdepth 1 ! -name "$(basename "$marker")" -exec rm -rf -- {} +
  else
    mkdir -p "$stage_dir"
    : >"$marker"
  fi

  local path lower ext normalized_ext
  while IFS= read -r -d '' path; do
    lower="${path,,}"
    for ext in "${extensions[@]}"; do
      normalized_ext="${ext#.}"
      normalized_ext="${normalized_ext,,}"
      if [[ "$lower" == *".${normalized_ext}" ]]; then
        ln -s "$path" "$stage_dir/$(basename "$path")"
        break
      fi
    done
  done < <(find "$source_dir" -maxdepth 1 -type f -print0)
}

infer_attack_type() {
  local python_bin=()
  if command -v uv >/dev/null 2>&1; then
    python_bin=(uv run python)
  elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    python_bin=("$SCRIPT_DIR/.venv/bin/python")
  else
    python_bin=(python)
  fi

  "${python_bin[@]}" - <<'PY'
from config import config
print(config.attack_type)
PY
}

resolve_adv_parent() {
  local preferred_parent="$1"
  local adv_subdir="$2"

  if [[ -d "$preferred_parent/$adv_subdir" ]]; then
    printf '%s\n' "$preferred_parent"
    return 0
  fi

  if [[ -d "$RUN_OUTPUT_DIR/$adv_subdir" ]]; then
    printf '%s\n' "$RUN_OUTPUT_DIR"
    return 0
  fi

  if [[ -d "$RUN_OUTPUT_DIR/protected/$adv_subdir" ]]; then
    printf '%s\n' "$RUN_OUTPUT_DIR/protected"
    return 0
  fi

  local matches=()
  mapfile -t matches < <(find "$RUN_OUTPUT_DIR" -type d -name "$adv_subdir" -print 2>/dev/null || true)
  if [[ "${#matches[@]}" -eq 1 ]]; then
    dirname "${matches[0]}"
    return 0
  fi

  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "[full] ERROR: found multiple '$adv_subdir' directories under $RUN_OUTPUT_DIR:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    echo "[full] Set PURIFY_PROTECTED_DIR to the directory that directly contains '$adv_subdir'." >&2
  else
    echo "[full] ERROR: could not find '$adv_subdir' under $preferred_parent or $RUN_OUTPUT_DIR." >&2
  fi
  return 1
}

RUN_EXP_SCRIPT="${RUN_EXP_SCRIPT:-$SCRIPT_DIR/run_exp.sh}"
AUDIOPURE_ROOT="${AUDIOPURE_ROOT:-/workspace/wht/pyproj/tts_related/audiopure_reproduce}"
AUDIOPURE_VENV="${AUDIOPURE_VENV:-$AUDIOPURE_ROOT/.venv}"
AUDIOPURE_SCRIPT="${AUDIOPURE_SCRIPT:-script/purify_purguard_audiopure.py}"
DUALPURE_ROOT="${DUALPURE_ROOT:-/workspace/wht/pyproj/tts_related/DualPure}"
DUALPURE_VENV="${DUALPURE_VENV:-$DUALPURE_ROOT/.venv-cu130}"
DUALPURE_SCRIPT="${DUALPURE_SCRIPT:-$DUALPURE_ROOT/script/purguard_purify.sh}"
WAVEPURIFIER_ROOT="${WAVEPURIFIER_ROOT:-/workspace/wht/pyproj/tts_related/wavepurifier.github.io}"
WAVEPURIFIER_VENV="${WAVEPURIFIER_VENV:-$WAVEPURIFIER_ROOT/.venv}"
WAVEPURIFIER_SCRIPT="${WAVEPURIFIER_SCRIPT:-$WAVEPURIFIER_ROOT/scripts/PurGuard_purify.sh}"
DEANTIFAKE_ROOT="${DEANTIFAKE_ROOT:-/workspace/wht/pyproj/tts_related/De-AntiFake}"
DEANTIFAKE_VENV="${DEANTIFAKE_VENV:-$DEANTIFAKE_ROOT/.venv}"
DEANTIFAKE_SCRIPT="${DEANTIFAKE_SCRIPT:-$DEANTIFAKE_ROOT/script/PurGuard_purify.sh}"
CHATTERBOX_ROOT="${CHATTERBOX_ROOT:-/workspace/wht/pyproj/tts_related/tts/chatterbox}"
CHATTERBOX_VENV="${CHATTERBOX_VENV:-/workspace/wht/shared-venv/tts}"
CHATTERBOX_SCRIPT="${CHATTERBOX_SCRIPT:-$CHATTERBOX_ROOT/scripts/purguard_clone_chatterbox.sh}"
F5TTS_ROOT="${F5TTS_ROOT:-/workspace/wht/pyproj/tts_related/tts/f5tts}"
F5TTS_VENV="${F5TTS_VENV:-/workspace/wht/shared-venv/tts}"
F5TTS_SCRIPT="${F5TTS_SCRIPT:-$F5TTS_ROOT/scripts/purguard_clone.py}"
COSYVOICE_ROOT="${COSYVOICE_ROOT:-/workspace/wht/pyproj/tts_related/tts/CosyVoice}"
COSYVOICE_VENV="${COSYVOICE_VENV:-$COSYVOICE_ROOT/.venv}"
COSYVOICE_SCRIPT="${COSYVOICE_SCRIPT:-$COSYVOICE_ROOT/scripts/purguard_clone_cosyvoice.py}"
QWEN3TTS_ROOT="${QWEN3TTS_ROOT:-/workspace/wht/pyproj/tts_related/tts/qwen3-tts}"
QWEN3TTS_VENV="${QWEN3TTS_VENV:-$QWEN3TTS_ROOT/.venv}"
QWEN3TTS_SCRIPT="${QWEN3TTS_SCRIPT:-$QWEN3TTS_ROOT/scripts/clone_purguard_qwen3_tts.py}"
SIMILARITY_VENV="${SIMILARITY_VENV:-$SCRIPT_DIR/.venv}"
SIMILARITY_SCRIPT="${SIMILARITY_SCRIPT:-$PROJECT_ROOT/scripts/eval_protected_speaker_similarity.py}"

if [[ ! -x "$RUN_EXP_SCRIPT" ]]; then
  echo "[full] ERROR: run_exp.sh is not executable: $RUN_EXP_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$AUDIOPURE_ROOT" ]]; then
  echo "[full] ERROR: AudioPure repo does not exist: $AUDIOPURE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$AUDIOPURE_ROOT/$AUDIOPURE_SCRIPT" ]]; then
  echo "[full] ERROR: AudioPure script does not exist: $AUDIOPURE_ROOT/$AUDIOPURE_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$DUALPURE_ROOT" ]]; then
  echo "[full] ERROR: DualPure repo does not exist: $DUALPURE_ROOT" >&2
  exit 1
fi
if [[ ! -x "$DUALPURE_SCRIPT" ]]; then
  echo "[full] ERROR: DualPure script is not executable: $DUALPURE_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$DUALPURE_VENV/bin/python" ]]; then
  echo "[full] ERROR: DualPure Python environment does not exist: $DUALPURE_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -d "$WAVEPURIFIER_ROOT" ]]; then
  echo "[full] ERROR: WavePurifier repo does not exist: $WAVEPURIFIER_ROOT" >&2
  exit 1
fi
if [[ ! -x "$WAVEPURIFIER_SCRIPT" ]]; then
  echo "[full] ERROR: WavePurifier script is not executable: $WAVEPURIFIER_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$WAVEPURIFIER_VENV/bin/python" ]]; then
  echo "[full] ERROR: WavePurifier Python environment does not exist: $WAVEPURIFIER_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -d "$DEANTIFAKE_ROOT" ]]; then
  echo "[full] ERROR: De-AntiFake repo does not exist: $DEANTIFAKE_ROOT" >&2
  exit 1
fi
if [[ ! -x "$DEANTIFAKE_SCRIPT" ]]; then
  echo "[full] ERROR: De-AntiFake script is not executable: $DEANTIFAKE_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$DEANTIFAKE_VENV/bin/python" ]]; then
  echo "[full] ERROR: De-AntiFake Python environment does not exist: $DEANTIFAKE_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -d "$CHATTERBOX_ROOT" ]]; then
  echo "[full] ERROR: Chatterbox repo does not exist: $CHATTERBOX_ROOT" >&2
  exit 1
fi
if [[ ! -x "$CHATTERBOX_SCRIPT" ]]; then
  echo "[full] ERROR: Chatterbox script is not executable: $CHATTERBOX_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$CHATTERBOX_VENV/bin/activate" ]]; then
  echo "[full] ERROR: Chatterbox uv virtualenv activate script does not exist: $CHATTERBOX_VENV/bin/activate" >&2
  exit 1
fi
if [[ ! -x "$CHATTERBOX_VENV/bin/python" ]]; then
  echo "[full] ERROR: Chatterbox Python environment does not exist: $CHATTERBOX_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -d "$F5TTS_ROOT" ]]; then
  echo "[full] ERROR: F5-TTS repo does not exist: $F5TTS_ROOT" >&2
  exit 1
fi
if [[ ! -f "$F5TTS_SCRIPT" ]]; then
  echo "[full] ERROR: F5-TTS script does not exist: $F5TTS_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$F5TTS_VENV/bin/python" ]]; then
  echo "[full] ERROR: F5-TTS Python environment does not exist: $F5TTS_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -d "$COSYVOICE_ROOT" ]]; then
  echo "[full] ERROR: CosyVoice repo does not exist: $COSYVOICE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$COSYVOICE_SCRIPT" ]]; then
  echo "[full] ERROR: CosyVoice script does not exist: $COSYVOICE_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$COSYVOICE_VENV/bin/python" ]]; then
  echo "[full] ERROR: CosyVoice Python environment does not exist: $COSYVOICE_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -d "$QWEN3TTS_ROOT" ]]; then
  echo "[full] ERROR: Qwen3-TTS repo does not exist: $QWEN3TTS_ROOT" >&2
  exit 1
fi
if [[ ! -f "$QWEN3TTS_SCRIPT" ]]; then
  echo "[full] ERROR: Qwen3-TTS script does not exist: $QWEN3TTS_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$QWEN3TTS_VENV/bin/python" ]]; then
  echo "[full] ERROR: Qwen3-TTS Python environment does not exist: $QWEN3TTS_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -f "$SIMILARITY_SCRIPT" ]]; then
  echo "[full] ERROR: clone speaker similarity script does not exist: $SIMILARITY_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$SIMILARITY_VENV/bin/python" ]]; then
  echo "[full] ERROR: clone speaker similarity Python environment does not exist: $SIMILARITY_VENV/bin/python" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/wht/exp/dual_attack_outputs/test_900_protected}"
ATTACK_TYPE="${ATTACK_TYPE:-$(infer_attack_type)}"
RUN_NAME="${RUN_NAME:-${ATTACK_TYPE}_${RUN_ID}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-$RUN_OUTPUT_DIR/protected}"
AUDIO_OUTPUT_INCLUDE_RUN_TS="${AUDIO_OUTPUT_INCLUDE_RUN_TS:-0}"
SAVE_AUDIO="${SAVE_AUDIO:-1}"
DEFAULT_PURIFY_GPU="${DEFAULT_PURIFY_GPU:-$(first_csv_or_space_item "${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}")}"

export RUN_ID OUTPUT_ROOT ATTACK_TYPE RUN_NAME RUN_OUTPUT_DIR
export AUDIO_OUTPUT_DIR AUDIO_OUTPUT_INCLUDE_RUN_TS SAVE_AUDIO

echo "[full] run_id=$RUN_ID"
echo "[full] attack_type=$ATTACK_TYPE"
echo "[full] run_output_dir=$RUN_OUTPUT_DIR"
echo "[full] attack_audio_output_dir=$AUDIO_OUTPUT_DIR"
echo "[full] cuda_device_order=$CUDA_DEVICE_ORDER"
echo "[full] default_purify_gpu=$DEFAULT_PURIFY_GPU"
echo "[full] strict_count_check=${STRICT_COUNT_CHECK:-1}"

if is_truthy "${SKIP_RUN:-0}"; then
  echo "[full] SKIP_RUN=1; using existing run output."
else
  echo "[full] running $RUN_EXP_SCRIPT"
  "$RUN_EXP_SCRIPT"
fi

if is_truthy "${DRY_RUN:-0}"; then
  echo "[full] DRY_RUN=1; run_exp.sh did not create attack audio, so purification is skipped."
  exit 0
fi

if ! is_truthy "$SAVE_AUDIO"; then
  echo "[full] ERROR: SAVE_AUDIO=$SAVE_AUDIO, so run_exp.sh will not create adv audio to purify." >&2
  exit 1
fi

PURIFY_ADV_SUBDIR="${PURIFY_ADV_SUBDIR:-adv}"
PURIFY_PROTECTED_DIR="${PURIFY_PROTECTED_DIR:-$AUDIO_OUTPUT_DIR}"
PURIFY_PROTECTED_DIR="$(resolve_adv_parent "$PURIFY_PROTECTED_DIR" "$PURIFY_ADV_SUBDIR")"
PURIFY_ADV_DIR="$PURIFY_PROTECTED_DIR/$PURIFY_ADV_SUBDIR"
PURIFY_OUTPUT_DIR="${PURIFY_OUTPUT_DIR:-$RUN_OUTPUT_DIR/purify/audiopure}"
PURIFY_DEVICE="${PURIFY_DEVICE:-auto}"
PURIFY_GPUS="${PURIFY_GPUS:-$DEFAULT_PURIFY_GPU}"
PURIFY_EXTENSIONS="${PURIFY_EXTENSIONS:-.wav}"
PURIFY_T="${PURIFY_T:-3}"
PURIFY_SEGMENT_LENGTH="${PURIFY_SEGMENT_LENGTH:-16000}"
PURIFY_SEGMENT_BATCH_SIZE="${PURIFY_SEGMENT_BATCH_SIZE:-1}"
PURIFY_CPU_WORKERS="${PURIFY_CPU_WORKERS:-1}"

adv_count="$(count_audio_files "$PURIFY_ADV_DIR" "$PURIFY_EXTENSIONS")"
BASE_AUDIO_COUNT="$adv_count"
PURIFY_EXPECTED_AUDIO_COUNT="${PURIFY_EXPECTED_AUDIO_COUNT:-900}"
if [[ "$PURIFY_EXPECTED_AUDIO_COUNT" == "auto" ]]; then
  PURIFY_EXPECTED_AUDIO_COUNT="$adv_count"
fi

if ! [[ "$BASE_AUDIO_COUNT" =~ ^[0-9]+$ ]] || [[ "$BASE_AUDIO_COUNT" -lt 1 ]]; then
  echo "[full] ERROR: initial adv audio count must be positive, found $BASE_AUDIO_COUNT under $PURIFY_ADV_DIR." >&2
  exit 1
fi
if ! [[ "$PURIFY_EXPECTED_AUDIO_COUNT" =~ ^[0-9]+$ ]] || [[ "$PURIFY_EXPECTED_AUDIO_COUNT" -lt 1 ]]; then
  echo "[full] ERROR: PURIFY_EXPECTED_AUDIO_COUNT must be a positive integer or auto." >&2
  exit 1
fi
if is_truthy "${STRICT_COUNT_CHECK:-1}" && [[ "$PURIFY_EXPECTED_AUDIO_COUNT" -ne "$BASE_AUDIO_COUNT" ]]; then
  echo "[full] ERROR: PURIFY_EXPECTED_AUDIO_COUNT=$PURIFY_EXPECTED_AUDIO_COUNT does not match initial adv count $BASE_AUDIO_COUNT. Set PURIFY_EXPECTED_AUDIO_COUNT=auto or fix the input." >&2
  exit 1
fi

mkdir -p "$PURIFY_OUTPUT_DIR"

echo "[full] purify_input=$PURIFY_ADV_DIR"
echo "[full] purify_input_count=$adv_count"
echo "[full] base_audio_count=$BASE_AUDIO_COUNT"
echo "[full] purify_expected_count=$PURIFY_EXPECTED_AUDIO_COUNT"
echo "[full] purify_output_dir=$PURIFY_OUTPUT_DIR"

purify_gpus=()
mapfile -t purify_gpus < <(split_csv_or_space "$PURIFY_GPUS")
if [[ "${#purify_gpus[@]}" -eq 0 ]]; then
  purify_gpus=("0")
fi

purify_extensions=()
mapfile -t purify_extensions < <(split_csv_or_space "$PURIFY_EXTENSIONS")
if [[ "${#purify_extensions[@]}" -eq 0 ]]; then
  purify_extensions=(".wav")
fi

purify_args=(
  "$AUDIOPURE_SCRIPT"
  "$PURIFY_PROTECTED_DIR"
  --adv-subdir "$PURIFY_ADV_SUBDIR"
  --output-dir "$PURIFY_OUTPUT_DIR"
  --expected-audio-count "$PURIFY_EXPECTED_AUDIO_COUNT"
  --extensions "${purify_extensions[@]}"
  --device "$PURIFY_DEVICE"
  --gpus "${purify_gpus[@]}"
  --t "$PURIFY_T"
  --segment-length "$PURIFY_SEGMENT_LENGTH"
  --segment-batch-size "$PURIFY_SEGMENT_BATCH_SIZE"
  --cpu-workers "$PURIFY_CPU_WORKERS"
)

if [[ -n "${PURIFY_DDPM_CONFIG:-}" ]]; then
  purify_args+=(--ddpm-config "$PURIFY_DDPM_CONFIG")
fi
if [[ -n "${PURIFY_DDPM_PATH:-}" ]]; then
  purify_args+=(--ddpm-path "$PURIFY_DDPM_PATH")
fi
if [[ -n "${PURIFY_LIMIT:-}" ]]; then
  purify_args+=(--limit "$PURIFY_LIMIT")
fi
if [[ -n "${PURIFY_MANIFEST:-}" ]]; then
  purify_args+=(--manifest "$PURIFY_MANIFEST")
fi
if is_truthy "${PURIFY_OVERWRITE:-0}"; then
  purify_args+=(--overwrite)
fi
if is_truthy "${PURIFY_RECORD_SKIPS:-0}"; then
  purify_args+=(--record-skips)
fi
if is_truthy "${PURIFY_FAIL_FAST:-0}"; then
  purify_args+=(--fail-fast)
fi
if is_truthy "${PURIFY_DRY_RUN:-0}"; then
  purify_args+=(--dry-run)
fi

if is_truthy "${SKIP_AUDIOPURE:-0}"; then
  echo "[full] SKIP_AUDIOPURE=1; skipping AudioPure purification."
elif command -v uv >/dev/null 2>&1; then
  echo "[full] running AudioPure with uv in $AUDIOPURE_ROOT"
  (
    cd "$AUDIOPURE_ROOT"
    UV_PROJECT_ENVIRONMENT="$AUDIOPURE_VENV" uv run python "${purify_args[@]}"
  )
elif [[ -x "$AUDIOPURE_VENV/bin/python" ]]; then
  echo "[full] uv not found; using $AUDIOPURE_VENV/bin/python"
  (
    cd "$AUDIOPURE_ROOT"
    "$AUDIOPURE_VENV/bin/python" "${purify_args[@]}"
  )
else
  echo "[full] ERROR: could not find uv or $AUDIOPURE_VENV/bin/python." >&2
  exit 1
fi
if ! is_truthy "${SKIP_AUDIOPURE:-0}" && ! is_truthy "${PURIFY_DRY_RUN:-0}"; then
  assert_audio_count_matches_base "audiopure" "$PURIFY_OUTPUT_DIR" "${PURIFY_OUTPUT_RECURSIVE:-0}" "$PURIFY_EXTENSIONS"
fi

DUALPURE_OUTPUT_DIR="${DUALPURE_OUTPUT_DIR:-$RUN_OUTPUT_DIR/purify/DualPure}"
DUALPURE_GPUS="${DUALPURE_GPUS:-$DEFAULT_PURIFY_GPU}"
DUALPURE_EXTENSIONS="${DUALPURE_EXTENSIONS:-$PURIFY_EXTENSIONS}"
DUALPURE_EXPECTED_COUNT="${DUALPURE_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
if [[ "$DUALPURE_EXPECTED_COUNT" == "auto" ]]; then
  DUALPURE_EXPECTED_COUNT="$adv_count"
fi
DUALPURE_T="${DUALPURE_T:-2}"
DUALPURE_SAMPLE_STEP="${DUALPURE_SAMPLE_STEP:-1}"
DUALPURE_BATCH_SIZE="${DUALPURE_BATCH_SIZE:-1}"
DUALPURE_SHARD_BY="${DUALPURE_SHARD_BY:-speaker}"
DUALPURE_LOG_DIR="${DUALPURE_LOG_DIR:-$DUALPURE_OUTPUT_DIR/logs}"

if ! [[ "$DUALPURE_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$DUALPURE_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: DUALPURE_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi

dualpure_gpus=()
mapfile -t dualpure_gpus < <(split_csv_or_space "$DUALPURE_GPUS")
if [[ "${#dualpure_gpus[@]}" -eq 0 ]]; then
  dualpure_gpus=("0")
fi
DUALPURE_GPUS_CSV="$(join_csv_from_values "${dualpure_gpus[@]}")"

dualpure_extensions=()
mapfile -t dualpure_extensions < <(split_csv_or_space "$DUALPURE_EXTENSIONS")
if [[ "${#dualpure_extensions[@]}" -eq 0 ]]; then
  dualpure_extensions=(".wav")
fi
DUALPURE_EXTENSIONS_CSV="$(join_csv_from_values "${dualpure_extensions[@]}")"

dualpure_args=(
  "$RUN_OUTPUT_DIR"
  --expected-count "$DUALPURE_EXPECTED_COUNT"
  --output-root "$DUALPURE_OUTPUT_DIR"
  --extensions "$DUALPURE_EXTENSIONS_CSV"
  --gpus "$DUALPURE_GPUS_CSV"
  --shard-by "$DUALPURE_SHARD_BY"
  --log-dir "$DUALPURE_LOG_DIR"
  --t "$DUALPURE_T"
  --sample-step "$DUALPURE_SAMPLE_STEP"
  --batch-size "$DUALPURE_BATCH_SIZE"
)

if is_truthy "${DUALPURE_CPU:-0}"; then
  dualpure_args+=(--cpu)
fi
if [[ -n "${DUALPURE_LIMIT:-}" ]]; then
  dualpure_args+=(--limit "$DUALPURE_LIMIT")
fi
if [[ -n "${DUALPURE_DIFFWAV_PATH:-}" ]]; then
  dualpure_args+=(--diffwav-path "$DUALPURE_DIFFWAV_PATH")
fi
if [[ -n "${DUALPURE_DIFFSPEC_PATH:-}" ]]; then
  dualpure_args+=(--diffspec-path "$DUALPURE_DIFFSPEC_PATH")
fi
if [[ -n "${DUALPURE_DDPM_CONFIG:-}" ]]; then
  dualpure_args+=(--ddpm-config "$DUALPURE_DDPM_CONFIG")
fi
if [[ -n "${DUALPURE_DIFFUSION_TYPE:-}" ]]; then
  dualpure_args+=(--diffusion-type "$DUALPURE_DIFFUSION_TYPE")
fi
if [[ -n "${DUALPURE_GRIFFINLIM_ITER:-}" ]]; then
  dualpure_args+=(--griffinlim-iter "$DUALPURE_GRIFFINLIM_ITER")
fi
if [[ -n "${DUALPURE_SEED:-}" ]]; then
  dualpure_args+=(--seed "$DUALPURE_SEED")
fi
if is_truthy "${DUALPURE_OVERWRITE:-0}"; then
  dualpure_args+=(--overwrite)
fi
if is_truthy "${DUALPURE_INCLUDE_HIDDEN:-0}"; then
  dualpure_args+=(--include-hidden)
fi
if is_truthy "${DUALPURE_NO_CUDA_PREFLIGHT:-0}"; then
  dualpure_args+=(--no-cuda-preflight)
fi
if is_truthy "${DUALPURE_DRY_RUN:-0}"; then
  dualpure_args+=(--dry-run)
fi

echo "[full] dualpure_input=$PURIFY_ADV_DIR"
echo "[full] dualpure_expected_count=$DUALPURE_EXPECTED_COUNT"
echo "[full] dualpure_output_dir=$DUALPURE_OUTPUT_DIR"

if is_truthy "${SKIP_DUALPURE:-0}"; then
  echo "[full] SKIP_DUALPURE=1; skipping DualPure purification."
else
  echo "[full] running DualPure with $DUALPURE_SCRIPT"
  "$DUALPURE_SCRIPT" "${dualpure_args[@]}"
fi
if ! is_truthy "${SKIP_DUALPURE:-0}" && ! is_truthy "${DUALPURE_DRY_RUN:-0}"; then
  assert_audio_count_matches_base "dualpure" "$DUALPURE_OUTPUT_DIR" "${DUALPURE_OUTPUT_RECURSIVE:-0}" "$DUALPURE_EXTENSIONS"
fi

WAVEPURIFIER_OUTPUT_DIR="${WAVEPURIFIER_OUTPUT_DIR:-$RUN_OUTPUT_DIR/purify/wavepurifier}"
WAVEPURIFIER_GPUS="${WAVEPURIFIER_GPUS:-$DEFAULT_PURIFY_GPU}"
WAVEPURIFIER_EXTENSIONS="${WAVEPURIFIER_EXTENSIONS:-$PURIFY_EXTENSIONS}"
WAVEPURIFIER_EXPECTED_COUNT="${WAVEPURIFIER_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
WAVEPURIFIER_CONFIG="${WAVEPURIFIER_CONFIG:-_256.yml}"
WAVEPURIFIER_T_S="${WAVEPURIFIER_T_S:-56}"
WAVEPURIFIER_T_M="${WAVEPURIFIER_T_M:-40}"
WAVEPURIFIER_T_L="${WAVEPURIFIER_T_L:-40}"
WAVEPURIFIER_SAMPLE_STEP="${WAVEPURIFIER_SAMPLE_STEP:-1}"
WAVEPURIFIER_LOG_DIR="${WAVEPURIFIER_LOG_DIR:-$WAVEPURIFIER_OUTPUT_DIR/logs}"
WAVEPURIFIER_TMP_DIR="${WAVEPURIFIER_TMP_DIR:-$WAVEPURIFIER_OUTPUT_DIR/tmp}"

wavepurifier_extensions=()
mapfile -t wavepurifier_extensions < <(split_csv_or_space "$WAVEPURIFIER_EXTENSIONS")
if [[ "${#wavepurifier_extensions[@]}" -eq 0 ]]; then
  wavepurifier_extensions=(".wav")
fi

wavepurifier_count="$(count_audio_files "$PURIFY_ADV_DIR" "$WAVEPURIFIER_EXTENSIONS")"
if [[ "$WAVEPURIFIER_EXPECTED_COUNT" == "auto" ]]; then
  WAVEPURIFIER_EXPECTED_COUNT="$wavepurifier_count"
fi
if ! [[ "$WAVEPURIFIER_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$WAVEPURIFIER_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: WAVEPURIFIER_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi
if [[ "$WAVEPURIFIER_EXPECTED_COUNT" -ge 0 && "$wavepurifier_count" -ne "$WAVEPURIFIER_EXPECTED_COUNT" ]]; then
  echo "[full] ERROR: expected $WAVEPURIFIER_EXPECTED_COUNT WavePurifier input audio files under $PURIFY_ADV_DIR, found $wavepurifier_count." >&2
  exit 1
fi

wavepurifier_gpus=()
mapfile -t wavepurifier_gpus < <(split_csv_or_space "$WAVEPURIFIER_GPUS")
if [[ "${#wavepurifier_gpus[@]}" -eq 0 ]]; then
  wavepurifier_gpus=("0")
fi

wavepurifier_args=(
  --input "$PURIFY_ADV_DIR"
  --output "$WAVEPURIFIER_OUTPUT_DIR"
  --extensions "${wavepurifier_extensions[@]}"
  --gpus "${wavepurifier_gpus[@]}"
  --log-dir "$WAVEPURIFIER_LOG_DIR"
  --tmp-dir "$WAVEPURIFIER_TMP_DIR"
  --config "$WAVEPURIFIER_CONFIG"
  --t_s "$WAVEPURIFIER_T_S"
  --t_m "$WAVEPURIFIER_T_M"
  --t_l "$WAVEPURIFIER_T_L"
  --sample_step "$WAVEPURIFIER_SAMPLE_STEP"
)

if [[ -n "${WAVEPURIFIER_MAX_FILES:-}" ]]; then
  wavepurifier_args+=(--max-files "$WAVEPURIFIER_MAX_FILES")
fi
if [[ -n "${WAVEPURIFIER_SEED:-}" ]]; then
  wavepurifier_args+=(--seed "$WAVEPURIFIER_SEED")
fi
if [[ -n "${WAVEPURIFIER_DATA_SEED:-}" ]]; then
  wavepurifier_args+=(--data_seed "$WAVEPURIFIER_DATA_SEED")
fi
if [[ -n "${WAVEPURIFIER_DIFFUSION_TYPE:-}" ]]; then
  wavepurifier_args+=(--diffusion_type "$WAVEPURIFIER_DIFFUSION_TYPE")
fi
if [[ -n "${WAVEPURIFIER_SCORE_TYPE:-}" ]]; then
  wavepurifier_args+=(--score_type "$WAVEPURIFIER_SCORE_TYPE")
fi
if is_truthy "${WAVEPURIFIER_OVERWRITE:-0}"; then
  wavepurifier_args+=(--overwrite)
fi
if is_truthy "${WAVEPURIFIER_INCLUDE_HIDDEN:-0}"; then
  wavepurifier_args+=(--include-hidden)
fi
if is_truthy "${WAVEPURIFIER_STOP_ON_ERROR:-0}"; then
  wavepurifier_args+=(--stop-on-error)
fi
if is_truthy "${WAVEPURIFIER_CLEAN_LOGS:-0}"; then
  wavepurifier_args+=(--clean-logs)
fi
if is_truthy "${WAVEPURIFIER_DRY_RUN:-0}"; then
  wavepurifier_args+=(--dry-run)
fi

echo "[full] wavepurifier_input=$PURIFY_ADV_DIR"
echo "[full] wavepurifier_input_count=$wavepurifier_count"
echo "[full] wavepurifier_expected_count=$WAVEPURIFIER_EXPECTED_COUNT"
echo "[full] wavepurifier_output_dir=$WAVEPURIFIER_OUTPUT_DIR"

if is_truthy "${SKIP_WAVEPURIFIER:-0}"; then
  echo "[full] SKIP_WAVEPURIFIER=1; skipping WavePurifier purification."
else
  echo "[full] running WavePurifier with $WAVEPURIFIER_SCRIPT"
  "$WAVEPURIFIER_SCRIPT" "${wavepurifier_args[@]}"
fi
if ! is_truthy "${SKIP_WAVEPURIFIER:-0}" && ! is_truthy "${WAVEPURIFIER_DRY_RUN:-0}"; then
  assert_audio_count_matches_base "wavepurifier" "$WAVEPURIFIER_OUTPUT_DIR" "${WAVEPURIFIER_OUTPUT_RECURSIVE:-0}" "$WAVEPURIFIER_EXTENSIONS"
fi

PHONEPURE_OUTPUT_DIR="${PHONEPURE_OUTPUT_DIR:-$RUN_OUTPUT_DIR/purify/phonepure}"
PHONEPURE_GPUS="${PHONEPURE_GPUS:-$DEFAULT_PURIFY_GPU}"
PHONEPURE_EXPECTED_COUNT="${PHONEPURE_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
PHONEPURE_T="${PHONEPURE_T:-3}"
PHONEPURE_SAMPLE_STEP="${PHONEPURE_SAMPLE_STEP:-1}"
PHONEPURE_SCORE_N="${PHONEPURE_SCORE_N:-30}"
PHONEPURE_SNR="${PHONEPURE_SNR:-0.4}"
PHONEPURE_BATCH_SIZE="${PHONEPURE_BATCH_SIZE:-1}"
PHONEPURE_WORKERS="${PHONEPURE_WORKERS:-0}"
PHONEPURE_TEXT_CACHE_DIR="${PHONEPURE_TEXT_CACHE_DIR:-$PHONEPURE_OUTPUT_DIR/text}"
PHONEPURE_WORK_DIR="${PHONEPURE_WORK_DIR:-$PHONEPURE_OUTPUT_DIR/.phonepure_work}"

phonepure_count="$(count_audio_files "$PURIFY_ADV_DIR" ".wav")"
if [[ "$PHONEPURE_EXPECTED_COUNT" == "auto" ]]; then
  PHONEPURE_EXPECTED_COUNT="$phonepure_count"
fi
if ! [[ "$PHONEPURE_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$PHONEPURE_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: PHONEPURE_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi
if [[ "$PHONEPURE_EXPECTED_COUNT" -ge 0 && "$phonepure_count" -ne "$PHONEPURE_EXPECTED_COUNT" ]]; then
  echo "[full] ERROR: expected $PHONEPURE_EXPECTED_COUNT PhonePuRe input wav files under $PURIFY_ADV_DIR, found $phonepure_count." >&2
  exit 1
fi

phonepure_gpus=()
mapfile -t phonepure_gpus < <(split_csv_or_space "$PHONEPURE_GPUS")
if [[ "${#phonepure_gpus[@]}" -eq 0 ]]; then
  phonepure_gpus=("0")
fi
PHONEPURE_GPUS_CSV="$(join_csv_from_values "${phonepure_gpus[@]}")"

phonepure_args=(
  --input "$PURIFY_ADV_DIR"
  --output "$PHONEPURE_OUTPUT_DIR"
  --gpus "$PHONEPURE_GPUS_CSV"
  --text-cache-dir "$PHONEPURE_TEXT_CACHE_DIR"
  --work-dir "$PHONEPURE_WORK_DIR"
  --t "$PHONEPURE_T"
  --sample-step "$PHONEPURE_SAMPLE_STEP"
  --score-n "$PHONEPURE_SCORE_N"
  --snr "$PHONEPURE_SNR"
  --batch-size "$PHONEPURE_BATCH_SIZE"
  --workers "$PHONEPURE_WORKERS"
)

if [[ -n "${PHONEPURE_TEXT_DIR:-}" ]]; then
  phonepure_args+=(--text-dir "$PHONEPURE_TEXT_DIR")
fi
if [[ -n "${PHONEPURE_CMU_ROOT:-}" ]]; then
  phonepure_args+=(--cmu-root "$PHONEPURE_CMU_ROOT")
fi
if [[ -n "${PHONEPURE_SPEAKERS:-}" ]]; then
  phonepure_args+=(--speakers "$PHONEPURE_SPEAKERS")
fi
if [[ -n "${PHONEPURE_DIFFWAV_PATH:-}" ]]; then
  phonepure_args+=(--diffwav-path "$PHONEPURE_DIFFWAV_PATH")
fi
if [[ -n "${PHONEPURE_SCORE_PATH:-}" ]]; then
  phonepure_args+=(--score-path "$PHONEPURE_SCORE_PATH")
fi
if [[ -n "${PHONEPURE_PHONEME_AVG_SPEC:-}" ]]; then
  phonepure_args+=(--phoneme-avg-spec "$PHONEPURE_PHONEME_AVG_SPEC")
fi
if [[ -n "${PHONEPURE_METHOD:-}" ]]; then
  phonepure_args+=(--purification-method "$PHONEPURE_METHOD")
fi
if is_truthy "${PHONEPURE_SKIP_EXISTING:-0}"; then
  phonepure_args+=(--skip-existing)
fi
if is_truthy "${PHONEPURE_SKIP_TEXT_PREP:-0}"; then
  phonepure_args+=(--skip-text-prep)
fi
if is_truthy "${PHONEPURE_ALLOW_MISSING_TEXT:-0}"; then
  phonepure_args+=(--allow-missing-text)
fi
if is_truthy "${PHONEPURE_FORCE_MFA:-0}"; then
  phonepure_args+=(--force-mfa)
fi
if is_truthy "${PHONEPURE_DRY_RUN:-0}"; then
  phonepure_args+=(--dry-run)
fi

echo "[full] phonepure_input=$PURIFY_ADV_DIR"
echo "[full] phonepure_input_count=$phonepure_count"
echo "[full] phonepure_expected_count=$PHONEPURE_EXPECTED_COUNT"
echo "[full] phonepure_output_dir=$PHONEPURE_OUTPUT_DIR"

if is_truthy "${SKIP_PHONEPURE:-0}"; then
  echo "[full] SKIP_PHONEPURE=1; skipping De-AntiFake PhonePuRe purification."
else
  if command -v uv >/dev/null 2>&1; then
    echo "[full] running De-AntiFake PhonePuRe with uv in $DEANTIFAKE_ROOT"
    (
      cd "$DEANTIFAKE_ROOT"
      UV_PROJECT_ENVIRONMENT="$DEANTIFAKE_VENV" \
        PYTHON_BIN="$DEANTIFAKE_VENV/bin/python" \
        uv run "$DEANTIFAKE_SCRIPT" "${phonepure_args[@]}"
    )
  else
    echo "[full] uv not found; running De-AntiFake PhonePuRe with $DEANTIFAKE_VENV/bin/python"
    PYTHON_BIN="$DEANTIFAKE_VENV/bin/python" "$DEANTIFAKE_SCRIPT" "${phonepure_args[@]}"
  fi
fi
if ! is_truthy "${SKIP_PHONEPURE:-0}" && ! is_truthy "${PHONEPURE_DRY_RUN:-0}"; then
  assert_audio_count_matches_base "phonepure" "$PHONEPURE_OUTPUT_DIR" "${PHONEPURE_OUTPUT_RECURSIVE:-0}" ".wav"
fi

CHATTERBOX_OUTPUT_ROOT="${CHATTERBOX_OUTPUT_ROOT:-$RUN_OUTPUT_DIR/clone/chatterbox}"
CHATTERBOX_GPU="${CHATTERBOX_GPU:-$DEFAULT_PURIFY_GPU}"
CHATTERBOX_DEVICE="${CHATTERBOX_DEVICE:-cuda}"
CHATTERBOX_LIMIT="${CHATTERBOX_LIMIT:-1000000000}"
CHATTERBOX_TEXT_MODE="${CHATTERBOX_TEXT_MODE:-cycle}"
CHATTERBOX_CHECKPOINTS="${CHATTERBOX_CHECKPOINTS:-$CHATTERBOX_ROOT/checkpoints}"
CHATTERBOX_EXPECTED_COUNT="${CHATTERBOX_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
CHATTERBOX_RECURSIVE="${CHATTERBOX_RECURSIVE:-0}"

if [[ "$CHATTERBOX_EXPECTED_COUNT" == "auto" ]]; then
  CHATTERBOX_EXPECTED_COUNT="$adv_count"
fi
if ! [[ "$CHATTERBOX_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$CHATTERBOX_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: CHATTERBOX_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi

clone_common_args=(
  --device "$CHATTERBOX_DEVICE"
  --limit "$CHATTERBOX_LIMIT"
  --text-mode "$CHATTERBOX_TEXT_MODE"
  --checkpoints "$CHATTERBOX_CHECKPOINTS"
)
if is_truthy "$CHATTERBOX_RECURSIVE"; then
  clone_common_args+=(--recursive)
else
  clone_common_args+=(--no-recursive)
fi
if [[ -n "${CHATTERBOX_EXAGGERATION:-}" ]]; then
  clone_common_args+=(--exaggeration "$CHATTERBOX_EXAGGERATION")
fi
if [[ -n "${CHATTERBOX_CFG_WEIGHT:-}" ]]; then
  clone_common_args+=(--cfg-weight "$CHATTERBOX_CFG_WEIGHT")
fi
if [[ -n "${CHATTERBOX_TEMPERATURE:-}" ]]; then
  clone_common_args+=(--temperature "$CHATTERBOX_TEMPERATURE")
fi
if [[ -n "${CHATTERBOX_TOP_P:-}" ]]; then
  clone_common_args+=(--top-p "$CHATTERBOX_TOP_P")
fi
if [[ -n "${CHATTERBOX_MIN_P:-}" ]]; then
  clone_common_args+=(--min-p "$CHATTERBOX_MIN_P")
fi
if [[ -n "${CHATTERBOX_REPETITION_PENALTY:-}" ]]; then
  clone_common_args+=(--repetition-penalty "$CHATTERBOX_REPETITION_PENALTY")
fi

clone_inputs=(
  "adv:$PURIFY_ADV_DIR"
  "audiopure:$PURIFY_OUTPUT_DIR"
  "DualPure:$DUALPURE_OUTPUT_DIR"
  "wavepurifier:$WAVEPURIFIER_OUTPUT_DIR"
  "phonepure:$PHONEPURE_OUTPUT_DIR"
)

echo "[full] chatterbox_output_root=$CHATTERBOX_OUTPUT_ROOT"
echo "[full] chatterbox_gpu=$CHATTERBOX_GPU"
echo "[full] chatterbox_recursive=$CHATTERBOX_RECURSIVE"

if is_truthy "${SKIP_CHATTERBOX:-0}"; then
  echo "[full] SKIP_CHATTERBOX=1; skipping Chatterbox cloning."
else
  for clone_item in "${clone_inputs[@]}"; do
    clone_name="${clone_item%%:*}"
    clone_input="${clone_item#*:}"
    clone_output="$CHATTERBOX_OUTPUT_ROOT/$clone_name"

    if [[ ! -d "$clone_input" ]]; then
      echo "[full] ERROR: Chatterbox input for $clone_name does not exist: $clone_input" >&2
      exit 1
    fi

    clone_count="$(count_clone_audio_files "$clone_input" "$CHATTERBOX_RECURSIVE" ".wav,.mp3,.flac,.ogg,.m4a")"
    if [[ "$CHATTERBOX_EXPECTED_COUNT" -ge 0 && "$clone_count" -ne "$CHATTERBOX_EXPECTED_COUNT" ]]; then
      echo "[full] ERROR: expected $CHATTERBOX_EXPECTED_COUNT Chatterbox input audio files for $clone_name under $clone_input, found $clone_count." >&2
      exit 1
    fi

    echo "[full] chatterbox_${clone_name}_input=$clone_input"
    echo "[full] chatterbox_${clone_name}_input_count=$clone_count"
    echo "[full] chatterbox_${clone_name}_output=$clone_output"

    if is_truthy "${CHATTERBOX_DRY_RUN:-0}"; then
      continue
    fi

    (
      if [[ "$CHATTERBOX_DEVICE" != "cpu" ]]; then
        export CUDA_VISIBLE_DEVICES="$CHATTERBOX_GPU"
      fi
      export UV_PROJECT_ENVIRONMENT="$CHATTERBOX_VENV"
      export VENV="$CHATTERBOX_VENV/bin/activate"
      "$CHATTERBOX_SCRIPT" \
        --input "$clone_input" \
        --output "$clone_output" \
        "${clone_common_args[@]}"
    )
    assert_audio_count_matches_base "chatterbox_${clone_name}" "$clone_output" "$CHATTERBOX_RECURSIVE" ".wav"
  done
fi

F5TTS_OUTPUT_ROOT="${F5TTS_OUTPUT_ROOT:-$RUN_OUTPUT_DIR/clone/f5-tts}"
F5TTS_GPU="${F5TTS_GPU:-$DEFAULT_PURIFY_GPU}"
F5TTS_DEVICE="${F5TTS_DEVICE:-cuda}"
F5TTS_LIMIT="${F5TTS_LIMIT:-0}"
F5TTS_EXPECTED_COUNT="${F5TTS_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
F5TTS_RECURSIVE="${F5TTS_RECURSIVE:-0}"
F5TTS_CKPT_FILE="${F5TTS_CKPT_FILE:-$F5TTS_ROOT/checkpoints/F5TTS_v1_Base/model_1250000.safetensors}"
F5TTS_VOCODER_DIR="${F5TTS_VOCODER_DIR:-$F5TTS_ROOT/checkpoints/vocos-mel-24khz}"
F5TTS_DEFAULT_TEXT_DIR="${F5TTS_DEFAULT_TEXT_DIR:-/mnt/wht/exp/dual_attack_outputs/test_900_text}"
if [[ -z "${F5TTS_TEXT_DIR:-}" && -d "$F5TTS_DEFAULT_TEXT_DIR" ]]; then
  F5TTS_TEXT_DIR="$F5TTS_DEFAULT_TEXT_DIR"
fi

if [[ "$F5TTS_EXPECTED_COUNT" == "auto" ]]; then
  F5TTS_EXPECTED_COUNT="$adv_count"
fi
if ! [[ "$F5TTS_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$F5TTS_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: F5TTS_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi

f5tts_common_args=(
  --device "$F5TTS_DEVICE"
  --limit "$F5TTS_LIMIT"
  --ckpt-file "$F5TTS_CKPT_FILE"
  --vocoder-dir "$F5TTS_VOCODER_DIR"
  --offline
)
if is_truthy "$F5TTS_RECURSIVE"; then
  f5tts_common_args+=(--recursive)
fi
if is_truthy "${F5TTS_RESUME:-0}"; then
  f5tts_common_args+=(--resume)
fi
if is_truthy "${F5TTS_REMOVE_SILENCE:-0}"; then
  f5tts_common_args+=(--remove-silence)
fi
if is_truthy "${F5TTS_STRICT_TEXT_COUNT:-0}"; then
  f5tts_common_args+=(--strict-text-count)
fi
if [[ -n "${F5TTS_TEXT_DIR:-}" ]]; then
  f5tts_common_args+=(--text-dir "$F5TTS_TEXT_DIR")
fi
if [[ -n "${F5TTS_NFE_STEP:-}" ]]; then
  f5tts_common_args+=(--nfe-step "$F5TTS_NFE_STEP")
fi
if [[ -n "${F5TTS_SPEED:-}" ]]; then
  f5tts_common_args+=(--speed "$F5TTS_SPEED")
fi
if [[ -n "${F5TTS_CFG_STRENGTH:-}" ]]; then
  f5tts_common_args+=(--cfg-strength "$F5TTS_CFG_STRENGTH")
fi
if [[ -n "${F5TTS_SWAY_SAMPLING_COEF:-}" ]]; then
  f5tts_common_args+=(--sway-sampling-coef "$F5TTS_SWAY_SAMPLING_COEF")
fi
if [[ -n "${F5TTS_TARGET_RMS:-}" ]]; then
  f5tts_common_args+=(--target-rms "$F5TTS_TARGET_RMS")
fi
if [[ -n "${F5TTS_CROSS_FADE_DURATION:-}" ]]; then
  f5tts_common_args+=(--cross-fade-duration "$F5TTS_CROSS_FADE_DURATION")
fi

echo "[full] f5tts_output_root=$F5TTS_OUTPUT_ROOT"
echo "[full] f5tts_gpu=$F5TTS_GPU"
echo "[full] f5tts_recursive=$F5TTS_RECURSIVE"
echo "[full] f5tts_text_dir=${F5TTS_TEXT_DIR:-}"

if is_truthy "${SKIP_F5TTS:-0}"; then
  echo "[full] SKIP_F5TTS=1; skipping F5-TTS cloning."
else
  for clone_item in "${clone_inputs[@]}"; do
    clone_name="${clone_item%%:*}"
    clone_input="${clone_item#*:}"
    clone_output="$F5TTS_OUTPUT_ROOT/$clone_name"

    if [[ ! -d "$clone_input" ]]; then
      echo "[full] ERROR: F5-TTS input for $clone_name does not exist: $clone_input" >&2
      exit 1
    fi

    clone_count="$(count_clone_audio_files "$clone_input" "$F5TTS_RECURSIVE" ".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus,.wma")"
    if [[ "$F5TTS_EXPECTED_COUNT" -ge 0 && "$clone_count" -ne "$F5TTS_EXPECTED_COUNT" ]]; then
      echo "[full] ERROR: expected $F5TTS_EXPECTED_COUNT F5-TTS input audio files for $clone_name under $clone_input, found $clone_count." >&2
      exit 1
    fi

    echo "[full] f5tts_${clone_name}_input=$clone_input"
    echo "[full] f5tts_${clone_name}_input_count=$clone_count"
    echo "[full] f5tts_${clone_name}_output=$clone_output"

    if is_truthy "${F5TTS_DRY_RUN:-0}"; then
      continue
    fi

    (
      cd "$F5TTS_ROOT"
      if [[ "$F5TTS_DEVICE" != "cpu" ]]; then
        export CUDA_VISIBLE_DEVICES="$F5TTS_GPU"
      fi
      export UV_PROJECT_ENVIRONMENT="$F5TTS_VENV"
      "$F5TTS_VENV/bin/python" "$F5TTS_SCRIPT" \
        --input "$clone_input" \
        --output "$clone_output" \
        "${f5tts_common_args[@]}"
    )
    assert_audio_count_matches_base "f5tts_${clone_name}" "$clone_output" "$F5TTS_RECURSIVE" ".wav"
  done
fi

COSYVOICE_OUTPUT_ROOT="${COSYVOICE_OUTPUT_ROOT:-$RUN_OUTPUT_DIR/clone/cosyvoice}"
if [[ -n "${COSYVOICE_PCI_BUS_ID:-}" ]]; then
  COSYVOICE_GPU="${COSYVOICE_GPU:-}"
else
  COSYVOICE_GPU="${COSYVOICE_GPU:-$DEFAULT_PURIFY_GPU}"
fi
COSYVOICE_EXPECTED_COUNT="${COSYVOICE_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
COSYVOICE_EXTENSIONS="${COSYVOICE_EXTENSIONS:-.wav,.flac,.mp3,.m4a,.ogg}"
COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-$COSYVOICE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B}"
COSYVOICE_SPEED="${COSYVOICE_SPEED:-1.0}"
COSYVOICE_PROMPT_PREFIX="${COSYVOICE_PROMPT_PREFIX:-You are a helpful assistant.<|endofprompt|>}"
COSYVOICE_INPUT_STAGING_ROOT="${COSYVOICE_INPUT_STAGING_ROOT:-$RUN_OUTPUT_DIR/clone/.cosyvoice_inputs}"

if [[ "$COSYVOICE_EXPECTED_COUNT" == "auto" ]]; then
  COSYVOICE_EXPECTED_COUNT="$adv_count"
fi
if ! [[ "$COSYVOICE_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$COSYVOICE_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: COSYVOICE_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi
if [[ -n "${COSYVOICE_LIMIT:-}" ]] && ! [[ "$COSYVOICE_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "[full] ERROR: COSYVOICE_LIMIT must be a non-negative integer when set." >&2
  exit 1
fi
if [[ -n "${COSYVOICE_GPU:-}" ]] && ! [[ "$COSYVOICE_GPU" =~ ^[0-9]+$ ]]; then
  echo "[full] ERROR: COSYVOICE_GPU must be a single GPU index." >&2
  exit 1
fi
if [[ -n "${COSYVOICE_GPU:-}" && -n "${COSYVOICE_PCI_BUS_ID:-}" ]]; then
  echo "[full] ERROR: set only one of COSYVOICE_GPU or COSYVOICE_PCI_BUS_ID." >&2
  exit 1
fi

cosyvoice_common_args=(
  --model-dir "$COSYVOICE_MODEL_DIR"
  --speed "$COSYVOICE_SPEED"
  --prompt-prefix "$COSYVOICE_PROMPT_PREFIX"
)
if [[ -n "${COSYVOICE_LIMIT:-}" ]]; then
  cosyvoice_common_args+=(--limit "$COSYVOICE_LIMIT")
fi
if [[ -n "${COSYVOICE_PCI_BUS_ID:-}" ]]; then
  cosyvoice_common_args+=(--pci-bus-id "$COSYVOICE_PCI_BUS_ID")
elif [[ -n "${COSYVOICE_GPU:-}" ]]; then
  cosyvoice_common_args+=(--gpu "$COSYVOICE_GPU")
fi
if is_truthy "${COSYVOICE_CHECK_ONLY:-0}"; then
  cosyvoice_common_args+=(--check-only)
fi
if is_truthy "${COSYVOICE_OVERWRITE:-0}"; then
  cosyvoice_common_args+=(--overwrite)
fi
if is_truthy "${COSYVOICE_STOP_ON_ERROR:-0}"; then
  cosyvoice_common_args+=(--stop-on-error)
fi
if is_truthy "${COSYVOICE_NO_FP16:-0}"; then
  cosyvoice_common_args+=(--no-fp16)
fi

COSYVOICE_GPU_LABEL="$COSYVOICE_GPU"
if [[ -z "${COSYVOICE_GPU_LABEL:-}" && -n "${COSYVOICE_PCI_BUS_ID:-}" ]]; then
  COSYVOICE_GPU_LABEL="pci-bus-id:$COSYVOICE_PCI_BUS_ID"
elif [[ -z "${COSYVOICE_GPU_LABEL:-}" ]]; then
  COSYVOICE_GPU_LABEL="<default>"
fi

echo "[full] cosyvoice_output_root=$COSYVOICE_OUTPUT_ROOT"
echo "[full] cosyvoice_gpu=$COSYVOICE_GPU_LABEL"
echo "[full] cosyvoice_input_staging_root=$COSYVOICE_INPUT_STAGING_ROOT"

if is_truthy "${SKIP_COSYVOICE:-0}"; then
  echo "[full] SKIP_COSYVOICE=1; skipping CosyVoice cloning."
else
  for clone_item in "${clone_inputs[@]}"; do
    clone_name="${clone_item%%:*}"
    clone_input="${clone_item#*:}"
    clone_output="$COSYVOICE_OUTPUT_ROOT/$clone_name"
    cosyvoice_stage_input="$COSYVOICE_INPUT_STAGING_ROOT/$clone_name"

    if [[ ! -d "$clone_input" ]]; then
      echo "[full] ERROR: CosyVoice input for $clone_name does not exist: $clone_input" >&2
      exit 1
    fi

    clone_count="$(count_audio_files_direct "$clone_input" "$COSYVOICE_EXTENSIONS")"
    if [[ "$COSYVOICE_EXPECTED_COUNT" -ge 0 && "$clone_count" -ne "$COSYVOICE_EXPECTED_COUNT" ]]; then
      echo "[full] ERROR: expected $COSYVOICE_EXPECTED_COUNT CosyVoice input audio files for $clone_name under $clone_input, found $clone_count." >&2
      exit 1
    fi

    stage_direct_audio_links "$clone_input" "$cosyvoice_stage_input" "cosyvoice" "$COSYVOICE_EXTENSIONS"

    cosyvoice_args=(
      --input "$cosyvoice_stage_input"
      --output "$clone_output"
      "${cosyvoice_common_args[@]}"
    )
    cosyvoice_expected_for_script="$COSYVOICE_EXPECTED_COUNT"
    if [[ -n "${COSYVOICE_LIMIT:-}" && "$cosyvoice_expected_for_script" -ge 0 && "$COSYVOICE_LIMIT" -lt "$cosyvoice_expected_for_script" ]]; then
      cosyvoice_expected_for_script="$COSYVOICE_LIMIT"
    fi
    if [[ "$cosyvoice_expected_for_script" -ge 0 ]]; then
      cosyvoice_args+=(--expected-count "$cosyvoice_expected_for_script")
    fi

    echo "[full] cosyvoice_${clone_name}_input=$clone_input"
    echo "[full] cosyvoice_${clone_name}_staged_input=$cosyvoice_stage_input"
    echo "[full] cosyvoice_${clone_name}_input_count=$clone_count"
    echo "[full] cosyvoice_${clone_name}_output=$clone_output"

    if is_truthy "${COSYVOICE_DRY_RUN:-0}"; then
      continue
    fi

    (
      cd "$COSYVOICE_ROOT"
      export UV_PROJECT_ENVIRONMENT="$COSYVOICE_VENV"
      if command -v uv >/dev/null 2>&1; then
        uv run python "$COSYVOICE_SCRIPT" "${cosyvoice_args[@]}"
      else
        "$COSYVOICE_VENV/bin/python" "$COSYVOICE_SCRIPT" "${cosyvoice_args[@]}"
      fi
    )
    if ! is_truthy "${COSYVOICE_CHECK_ONLY:-0}"; then
      assert_audio_count_matches_base "cosyvoice_${clone_name}" "$clone_output" 0 ".wav"
    fi
  done
fi

QWEN3TTS_OUTPUT_ROOT="${QWEN3TTS_OUTPUT_ROOT:-$RUN_OUTPUT_DIR/clone/qwen3-tts}"
QWEN3TTS_GPU="${QWEN3TTS_GPU:-$DEFAULT_PURIFY_GPU}"
QWEN3TTS_DEVICE="${QWEN3TTS_DEVICE:-cuda:0}"
QWEN3TTS_EXPECTED_COUNT="${QWEN3TTS_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
QWEN3TTS_EXTENSIONS="${QWEN3TTS_EXTENSIONS:-.wav,.flac,.mp3,.m4a,.ogg,.opus}"
QWEN3TTS_INPUT_STAGING_ROOT="${QWEN3TTS_INPUT_STAGING_ROOT:-$RUN_OUTPUT_DIR/clone/.qwen3tts_inputs}"
QWEN3TTS_SIZE="${QWEN3TTS_SIZE:-0.6b}"
QWEN3TTS_ATTN="${QWEN3TTS_ATTN:-sdpa}"
QWEN3TTS_LANGUAGE="${QWEN3TTS_LANGUAGE:-English}"
QWEN3TTS_MAX_NEW_TOKENS="${QWEN3TTS_MAX_NEW_TOKENS:-512}"
QWEN3TTS_MANIFEST_NAME="${QWEN3TTS_MANIFEST_NAME:-manifest.csv}"

if [[ "$QWEN3TTS_EXPECTED_COUNT" == "auto" ]]; then
  QWEN3TTS_EXPECTED_COUNT="$adv_count"
fi
if ! [[ "$QWEN3TTS_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$QWEN3TTS_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: QWEN3TTS_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi
if [[ -n "${QWEN3TTS_LIMIT:-}" ]] && ! [[ "$QWEN3TTS_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[full] ERROR: QWEN3TTS_LIMIT must be a positive integer when set." >&2
  exit 1
fi
if [[ -n "${QWEN3TTS_GPU:-}" ]] && ! [[ "$QWEN3TTS_GPU" =~ ^[0-9]+$ ]]; then
  echo "[full] ERROR: QWEN3TTS_GPU must be a single GPU index." >&2
  exit 1
fi
if ! [[ "$QWEN3TTS_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[full] ERROR: QWEN3TTS_MAX_NEW_TOKENS must be a positive integer." >&2
  exit 1
fi

qwen3tts_common_args=(
  --device "$QWEN3TTS_DEVICE"
  --size "$QWEN3TTS_SIZE"
  --attn "$QWEN3TTS_ATTN"
  --language "$QWEN3TTS_LANGUAGE"
  --max-new-tokens "$QWEN3TTS_MAX_NEW_TOKENS"
  --manifest-name "$QWEN3TTS_MANIFEST_NAME"
)
if [[ -n "${QWEN3TTS_LIMIT:-}" ]]; then
  qwen3tts_common_args+=(--limit "$QWEN3TTS_LIMIT")
fi
if [[ -n "${QWEN3TTS_MODEL_PATH:-}" ]]; then
  qwen3tts_common_args+=(--model-path "$QWEN3TTS_MODEL_PATH")
fi
if [[ -n "${QWEN3TTS_CACHE_DIR:-}" ]]; then
  qwen3tts_common_args+=(--cache-dir "$QWEN3TTS_CACHE_DIR")
fi
if [[ -n "${QWEN3TTS_NUM_SHARDS:-}" ]]; then
  qwen3tts_common_args+=(--num-shards "$QWEN3TTS_NUM_SHARDS")
fi
if [[ -n "${QWEN3TTS_SHARD_INDEX:-}" ]]; then
  qwen3tts_common_args+=(--shard-index "$QWEN3TTS_SHARD_INDEX")
fi
if is_truthy "${QWEN3TTS_OVERWRITE:-0}"; then
  qwen3tts_common_args+=(--overwrite)
fi
if is_truthy "${QWEN3TTS_CONTINUE_ON_ERROR:-0}"; then
  qwen3tts_common_args+=(--continue-on-error)
fi
if is_truthy "${QWEN3TTS_DRY_RUN:-0}"; then
  qwen3tts_common_args+=(--dry-run)
fi

echo "[full] qwen3tts_output_root=$QWEN3TTS_OUTPUT_ROOT"
echo "[full] qwen3tts_gpu=$QWEN3TTS_GPU"
echo "[full] qwen3tts_device=$QWEN3TTS_DEVICE"
echo "[full] qwen3tts_input_staging_root=$QWEN3TTS_INPUT_STAGING_ROOT"

if is_truthy "${SKIP_QWEN3TTS:-0}"; then
  echo "[full] SKIP_QWEN3TTS=1; skipping Qwen3-TTS cloning."
else
  for clone_item in "${clone_inputs[@]}"; do
    clone_name="${clone_item%%:*}"
    clone_input="${clone_item#*:}"
    clone_output="$QWEN3TTS_OUTPUT_ROOT/$clone_name"
    qwen3tts_stage_input="$QWEN3TTS_INPUT_STAGING_ROOT/$clone_name"

    if [[ ! -d "$clone_input" ]]; then
      echo "[full] ERROR: Qwen3-TTS input for $clone_name does not exist: $clone_input" >&2
      exit 1
    fi

    clone_count="$(count_audio_files_direct "$clone_input" "$QWEN3TTS_EXTENSIONS")"
    if [[ "$QWEN3TTS_EXPECTED_COUNT" -ge 0 && "$clone_count" -ne "$QWEN3TTS_EXPECTED_COUNT" ]]; then
      echo "[full] ERROR: expected $QWEN3TTS_EXPECTED_COUNT Qwen3-TTS input audio files for $clone_name under $clone_input, found $clone_count." >&2
      exit 1
    fi

    stage_direct_audio_links "$clone_input" "$qwen3tts_stage_input" "qwen3tts" "$QWEN3TTS_EXTENSIONS"

    qwen3tts_expected_for_script="$QWEN3TTS_EXPECTED_COUNT"
    if [[ "$qwen3tts_expected_for_script" -lt 0 ]]; then
      qwen3tts_expected_for_script="$clone_count"
    fi

    qwen3tts_args=(
      --input-dir "$qwen3tts_stage_input"
      --output-dir "$clone_output"
      --expected-count "$qwen3tts_expected_for_script"
      "${qwen3tts_common_args[@]}"
    )

    echo "[full] qwen3tts_${clone_name}_input=$clone_input"
    echo "[full] qwen3tts_${clone_name}_staged_input=$qwen3tts_stage_input"
    echo "[full] qwen3tts_${clone_name}_input_count=$clone_count"
    echo "[full] qwen3tts_${clone_name}_output=$clone_output"

    (
      cd "$QWEN3TTS_ROOT"
      if [[ "$QWEN3TTS_DEVICE" != "cpu" ]]; then
        export CUDA_VISIBLE_DEVICES="$QWEN3TTS_GPU"
      fi
      export UV_PROJECT_ENVIRONMENT="$QWEN3TTS_VENV"
      if command -v uv >/dev/null 2>&1; then
        uv run python "$QWEN3TTS_SCRIPT" "${qwen3tts_args[@]}"
      else
        "$QWEN3TTS_VENV/bin/python" "$QWEN3TTS_SCRIPT" "${qwen3tts_args[@]}"
      fi
    )
    if ! is_truthy "${QWEN3TTS_DRY_RUN:-0}"; then
      assert_audio_count_matches_base "qwen3tts_${clone_name}" "$clone_output" 0 ".wav"
    fi
  done
fi

SIMILARITY_CLEAN_DIR="${SIMILARITY_CLEAN_DIR:-/mnt/wht/exp/test_900}"
SIMILARITY_RUN_NAME="${SIMILARITY_RUN_NAME:-$(basename "$RUN_OUTPUT_DIR")}"
SIMILARITY_OUTPUT_DIR="${SIMILARITY_OUTPUT_DIR:-$PROJECT_ROOT/eval/$SIMILARITY_RUN_NAME}"
SIMILARITY_GPU="${SIMILARITY_GPU:-$DEFAULT_PURIFY_GPU}"
SIMILARITY_DEVICE="${SIMILARITY_DEVICE:-cuda:0}"
SIMILARITY_MODELS="${SIMILARITY_MODELS:-ecapa xvector dvector}"
SIMILARITY_EXPECTED_COUNT="${SIMILARITY_EXPECTED_COUNT:-$PURIFY_EXPECTED_AUDIO_COUNT}"
SIMILARITY_EXTENSIONS="${SIMILARITY_EXTENSIONS:-.wav,.flac,.mp3,.ogg,.m4a,.aac,.opus}"
SIMILARITY_SAMPLE_RATE="${SIMILARITY_SAMPLE_RATE:-16000}"
SIMILARITY_MATCH="${SIMILARITY_MATCH:-auto}"
SIMILARITY_RECURSIVE="${SIMILARITY_RECURSIVE:-1}"

if [[ "$SIMILARITY_EXPECTED_COUNT" == "auto" ]]; then
  SIMILARITY_EXPECTED_COUNT="$adv_count"
fi
if ! [[ "$SIMILARITY_EXPECTED_COUNT" =~ ^-?[0-9]+$ ]] || [[ "$SIMILARITY_EXPECTED_COUNT" -lt -1 ]]; then
  echo "[full] ERROR: SIMILARITY_EXPECTED_COUNT must be -1 or a non-negative integer." >&2
  exit 1
fi
if [[ -n "${SIMILARITY_LIMIT:-}" ]] && ! [[ "$SIMILARITY_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[full] ERROR: SIMILARITY_LIMIT must be a positive integer when set." >&2
  exit 1
fi
if [[ -n "${SIMILARITY_GPU:-}" ]] && ! [[ "$SIMILARITY_GPU" =~ ^[0-9]+$ ]]; then
  echo "[full] ERROR: SIMILARITY_GPU must be a single GPU index." >&2
  exit 1
fi

similarity_models=()
mapfile -t similarity_models < <(split_csv_or_space "$SIMILARITY_MODELS")
if [[ "${#similarity_models[@]}" -eq 0 ]]; then
  similarity_models=("ecapa" "xvector" "dvector")
fi

similarity_args=(
  --clean-dir "$SIMILARITY_CLEAN_DIR"
  --protected-root "$RUN_OUTPUT_DIR/clone"
  --methods "__full_sh_no_default__"
  --output-dir "$SIMILARITY_OUTPUT_DIR"
  --models "${similarity_models[@]}"
  --device "$SIMILARITY_DEVICE"
  --sample-rate "$SIMILARITY_SAMPLE_RATE"
  --match "$SIMILARITY_MATCH"
)
if ! is_truthy "$SIMILARITY_RECURSIVE"; then
  similarity_args+=(--no-recursive)
fi
if [[ -n "${SIMILARITY_LIMIT:-}" ]]; then
  similarity_args+=(--limit "$SIMILARITY_LIMIT")
fi
if [[ -n "${SIMILARITY_ECAPA_MODEL_PATH:-}" ]]; then
  similarity_args+=(--ecapa-model-path "$SIMILARITY_ECAPA_MODEL_PATH")
fi
if [[ -n "${SIMILARITY_XVECTOR_MODEL_PATH:-}" ]]; then
  similarity_args+=(--xvector-model-path "$SIMILARITY_XVECTOR_MODEL_PATH")
fi
if [[ -n "${SIMILARITY_ECAPA_THRESHOLD:-}" ]]; then
  similarity_args+=(--ecapa-threshold "$SIMILARITY_ECAPA_THRESHOLD")
fi
if [[ -n "${SIMILARITY_XVECTOR_THRESHOLD:-}" ]]; then
  similarity_args+=(--xvector-threshold "$SIMILARITY_XVECTOR_THRESHOLD")
fi
if [[ -n "${SIMILARITY_DVECTOR_THRESHOLD:-}" ]]; then
  similarity_args+=(--dvector-threshold "$SIMILARITY_DVECTOR_THRESHOLD")
fi
if [[ -n "${SIMILARITY_EXCLUDE_DIRS:-}" ]]; then
  similarity_exclude_dirs=()
  mapfile -t similarity_exclude_dirs < <(split_csv_or_space "$SIMILARITY_EXCLUDE_DIRS")
  for exclude_dir in "${similarity_exclude_dirs[@]}"; do
    similarity_args+=(--exclude-dir "$exclude_dir")
  done
fi
if is_truthy "${SIMILARITY_NO_DEFAULT_EXCLUDES:-0}"; then
  similarity_args+=(--no-default-excludes)
fi
if is_truthy "${SIMILARITY_DRY_RUN:-0}"; then
  similarity_args+=(--dry-run)
fi

clone_tts_outputs=(
  "chatterbox:$CHATTERBOX_OUTPUT_ROOT"
  "f5-tts:$F5TTS_OUTPUT_ROOT"
  "qwen3-tts:$QWEN3TTS_OUTPUT_ROOT"
  "cosyvoice:$COSYVOICE_OUTPUT_ROOT"
)

echo "[full] similarity_clean_dir=$SIMILARITY_CLEAN_DIR"
echo "[full] similarity_output_dir=$SIMILARITY_OUTPUT_DIR"
echo "[full] similarity_gpu=$SIMILARITY_GPU"
echo "[full] similarity_device=$SIMILARITY_DEVICE"
echo "[full] similarity_models=${similarity_models[*]}"

if is_truthy "${SKIP_SIMILARITY:-0}"; then
  echo "[full] SKIP_SIMILARITY=1; skipping clone speaker similarity."
else
  if [[ ! -d "$SIMILARITY_CLEAN_DIR" ]]; then
    echo "[full] ERROR: similarity clean directory does not exist: $SIMILARITY_CLEAN_DIR" >&2
    exit 1
  fi

  for tts_item in "${clone_tts_outputs[@]}"; do
    tts_name="${tts_item%%:*}"
    tts_output_root="${tts_item#*:}"

    for clone_item in "${clone_inputs[@]}"; do
      clone_name="${clone_item%%:*}"
      eval_dir="$tts_output_root/$clone_name"

      if [[ ! -d "$eval_dir" ]]; then
        echo "[full] ERROR: clone similarity input does not exist: $eval_dir" >&2
        exit 1
      fi

      eval_count="$(count_clone_audio_files "$eval_dir" "$SIMILARITY_RECURSIVE" "$SIMILARITY_EXTENSIONS")"
      if [[ "$SIMILARITY_EXPECTED_COUNT" -ge 0 && "$eval_count" -ne "$SIMILARITY_EXPECTED_COUNT" ]]; then
        echo "[full] ERROR: expected $SIMILARITY_EXPECTED_COUNT clone audio files for $tts_name/$clone_name under $eval_dir, found $eval_count." >&2
        exit 1
      fi

      echo "[full] similarity_${tts_name}_${clone_name}_input=$eval_dir"
      echo "[full] similarity_${tts_name}_${clone_name}_input_count=$eval_count"
      similarity_args+=(--eval-dir "$tts_name/$clone_name=$eval_dir")
    done
  done

  (
    cd "$SCRIPT_DIR"
    if [[ "$SIMILARITY_DEVICE" != "cpu" && -n "${SIMILARITY_GPU:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="$SIMILARITY_GPU"
    fi
    export UV_PROJECT_ENVIRONMENT="$SIMILARITY_VENV"
    if command -v uv >/dev/null 2>&1; then
      uv run python "$SIMILARITY_SCRIPT" "${similarity_args[@]}"
    else
      "$SIMILARITY_VENV/bin/python" "$SIMILARITY_SCRIPT" "${similarity_args[@]}"
    fi
  )
fi

echo "[full] done"
echo "[full] attack output: $RUN_OUTPUT_DIR"
echo "[full] audiopure output: $PURIFY_OUTPUT_DIR"
echo "[full] dualpure output: $DUALPURE_OUTPUT_DIR"
echo "[full] wavepurifier output: $WAVEPURIFIER_OUTPUT_DIR"
echo "[full] phonepure output: $PHONEPURE_OUTPUT_DIR"
echo "[full] chatterbox output: $CHATTERBOX_OUTPUT_ROOT"
echo "[full] f5tts output: $F5TTS_OUTPUT_ROOT"
echo "[full] cosyvoice output: $COSYVOICE_OUTPUT_ROOT"
echo "[full] qwen3tts output: $QWEN3TTS_OUTPUT_ROOT"
echo "[full] clone speaker similarity output: $SIMILARITY_OUTPUT_DIR"
