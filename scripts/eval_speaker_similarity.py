#!/usr/bin/env python3
"""Evaluate ECAPA-TDNN, x-vector, and d-vector speaker similarity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUAL_ATTACK_DIR = PROJECT_ROOT / "experiments" / "dual_target_attack"
if str(DUAL_ATTACK_DIR) not in sys.path:
    sys.path.insert(0, str(DUAL_ATTACK_DIR))

DEFAULT_CLEAN_DIR = Path("/mnt/data/wht/voxceleb1/test_800")
DEFAULT_GEN_ROOT = Path("/mnt/data/wht/exp/test_800/antifake")
DEFAULT_OUTPUT_DIR = DEFAULT_GEN_ROOT / "speaker_metrics"
DEFAULT_ECAPA_PATH = DUAL_ATTACK_DIR / "models" / "ecapa_tdnn_pretrained"
DEFAULT_XVECTOR_PATH = DUAL_ATTACK_DIR / "models" / "xvector_tdnn_pretrained"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
DEFAULT_THRESHOLDS = {
    "ecapa": 0.3195,
    "xvector": 0.9472,
    "dvector": 0.7235,
}
torch = None
F = None
torchaudio = None


def require_torch() -> None:
    global torch, F, torchaudio
    if torch is not None:
        return
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
        import torchaudio as torchaudio_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}\n"
            f"Python executable: {sys.executable}\n"
            "Run this script in the dual_target_attack environment, or install:\n"
            "  python -m pip install -U torch torchaudio"
        ) from exc

    torch = torch_module
    F = functional_module
    torchaudio = torchaudio_module


@dataclass(frozen=True)
class Pair:
    eval_set: str
    clean_path: Path
    gen_path: Path
    rel_path: str


class DVectorEncoder:
    """Resemblyzer d-vector wrapper with the same small API as local encoders."""

    def __init__(self, device: str, sample_rate: int) -> None:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: resemblyzer\n"
                f"Python executable: {sys.executable}\n"
                "Install it into this environment with:\n"
                "  python -m pip install -U resemblyzer"
            ) from exc

        self._encoder = VoiceEncoder(device=device, verbose=False)
        self._preprocess_wav = preprocess_wav
        self.sample_rate = sample_rate

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {tuple(x.shape)}")

        embeddings = []
        with torch.no_grad():
            for wav in x.detach().cpu():
                wav_np = wav.numpy().astype(np.float32)
                wav_np = self._preprocess_wav(wav_np, source_sr=self.sample_rate)
                if wav_np.size == 0:
                    raise ValueError("Resemblyzer preprocessing produced empty audio")
                embeddings.append(self._encoder.embed_utterance(wav_np))
        emb = torch.from_numpy(np.stack(embeddings)).float()
        return F.normalize(emb, p=2, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated audio directories against a clean directory using "
            "ECAPA-TDNN, x-vector, and d-vector speaker embeddings."
        )
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help="Clean reference audio directory.",
    )
    parser.add_argument(
        "--gen-root",
        type=Path,
        default=DEFAULT_GEN_ROOT,
        help=(
            "Root containing generated subdirectories such as adv/, chatterbox/, "
            "cosyvoice/. Ignored when --adv-dir is provided."
        ),
    )
    parser.add_argument(
        "--adv-dir",
        type=Path,
        action="append",
        default=[],
        help="Generated audio directory to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-file and summary outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ecapa", "xvector", "dvector"),
        default=("ecapa", "dvector", "xvector"),
        help="Speaker embedding models to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda, cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate used before embedding extraction.",
    )
    parser.add_argument(
        "--match",
        choices=("auto", "relative", "filename", "stem"),
        default="auto",
        help="How generated files are matched to clean files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of matched files per generated directory.",
    )
    parser.add_argument(
        "--ecapa-model-path",
        type=Path,
        default=DEFAULT_ECAPA_PATH,
        help="Local ECAPA SpeechBrain model directory.",
    )
    parser.add_argument(
        "--xvector-model-path",
        type=Path,
        default=Path(os.environ.get("DUAL_ATTACK_XVECTOR_MODEL_PATH", DEFAULT_XVECTOR_PATH)),
        help="Local x-vector SpeechBrain model directory.",
    )
    parser.add_argument(
        "--ecapa-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["ecapa"],
        help="Threshold for ECAPA same-speaker verification on similarity score.",
    )
    parser.add_argument(
        "--xvector-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["xvector"],
        help="Threshold for x-vector same-speaker verification on similarity score.",
    )
    parser.add_argument(
        "--dvector-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["dvector"],
        help="Threshold for d-vector same-speaker verification on similarity score.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly under each directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matched/unmatched files; do not load models.",
    )
    return parser.parse_args()


def iter_audio_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def get_eval_dirs(args: argparse.Namespace, recursive: bool) -> list[Path]:
    if args.adv_dir:
        return [path.resolve() for path in args.adv_dir]

    gen_root = args.gen_root.resolve()
    subdirs = [path for path in sorted(gen_root.iterdir()) if path.is_dir()]
    eval_dirs = [
        path for path in subdirs if any(iter_audio_files(path, recursive=recursive))
    ]
    if eval_dirs:
        return eval_dirs
    if any(iter_audio_files(gen_root, recursive=recursive)):
        return [gen_root]
    return []


def build_clean_indexes(clean_files: list[Path], clean_dir: Path) -> dict[str, dict[str, list[Path]]]:
    indexes: dict[str, dict[str, list[Path]]] = {
        "relative": defaultdict(list),
        "filename": defaultdict(list),
        "stem": defaultdict(list),
        "normalized_stem": defaultdict(list),
    }
    for path in clean_files:
        indexes["relative"][path.relative_to(clean_dir).as_posix()].append(path)
        indexes["filename"][path.name].append(path)
        indexes["stem"][path.stem].append(path)
        indexes["normalized_stem"][normalize_key(path.stem)].append(path)
    return indexes


def unique_lookup(index: dict[str, list[Path]], key: str) -> Path | None:
    matches = index.get(key, [])
    return matches[0] if len(matches) == 1 else None


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalized_containment_lookup(
    index: dict[str, list[Path]], gen_stem: str
) -> Path | None:
    gen_key = normalize_key(gen_stem)
    exact = unique_lookup(index, gen_key)
    if exact is not None:
        return exact

    matches: list[Path] = []
    for clean_key, clean_paths in index.items():
        if len(clean_paths) == 1 and clean_key and clean_key in gen_key:
            matches.append(clean_paths[0])
    unique_matches = set(matches)
    return matches[0] if len(unique_matches) == 1 else None


def match_clean_file(
    gen_path: Path,
    gen_dir: Path,
    clean_dir: Path,
    clean_indexes: dict[str, dict[str, list[Path]]],
    mode: str,
) -> Path | None:
    rel_key = gen_path.relative_to(gen_dir).as_posix()
    attempts = {
        "relative": lambda: unique_lookup(clean_indexes["relative"], rel_key),
        "filename": lambda: unique_lookup(clean_indexes["filename"], gen_path.name),
        "stem": lambda: unique_lookup(clean_indexes["stem"], gen_path.stem),
        "normalized": lambda: normalized_containment_lookup(
            clean_indexes["normalized_stem"], gen_path.stem
        ),
    }
    if mode != "auto":
        return attempts[mode]()

    for key in ("relative", "filename", "stem", "normalized"):
        match = attempts[key]()
        if match is not None:
            return match

    direct = clean_dir / rel_key
    return direct if direct.exists() else None


def collect_pairs(
    clean_dir: Path,
    eval_dirs: list[Path],
    recursive: bool,
    match_mode: str,
    limit: int | None,
) -> tuple[dict[str, list[Pair]], dict[str, int]]:
    clean_files = list(iter_audio_files(clean_dir, recursive=recursive))
    clean_indexes = build_clean_indexes(clean_files, clean_dir)
    pairs_by_set: dict[str, list[Pair]] = {}
    unmatched_by_set: dict[str, int] = {}

    for gen_dir in eval_dirs:
        eval_set = gen_dir.name
        pairs: list[Pair] = []
        unmatched = 0
        for gen_path in iter_audio_files(gen_dir, recursive=recursive):
            clean_path = match_clean_file(
                gen_path=gen_path,
                gen_dir=gen_dir,
                clean_dir=clean_dir,
                clean_indexes=clean_indexes,
                mode=match_mode,
            )
            if clean_path is None:
                unmatched += 1
                continue
            pairs.append(
                Pair(
                    eval_set=eval_set,
                    clean_path=clean_path,
                    gen_path=gen_path,
                    rel_path=gen_path.relative_to(gen_dir).as_posix(),
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
        pairs_by_set[eval_set] = pairs
        unmatched_by_set[eval_set] = unmatched

    return pairs_by_set, unmatched_by_set


def load_audio(path: Path, sample_rate: int, device: torch.device) -> torch.Tensor:
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)
    except Exception:
        try:
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(dim=0, keepdim=True)
        except Exception as torchaudio_exc:
            raise RuntimeError(
                f"Could not load audio file {path} with soundfile or torchaudio"
            ) from torchaudio_exc

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def load_eval_model(model_name: str, args: argparse.Namespace):
    if model_name == "dvector":
        return DVectorEncoder(device=args.device, sample_rate=args.sample_rate)

    try:
        from models.speaker_model import load_speaker_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import experiments/dual_target_attack/models/speaker_model.py"
        ) from exc

    model_path = args.ecapa_model_path if model_name == "ecapa" else args.xvector_model_path
    return load_speaker_model(
        model_path=str(model_path),
        model_type=model_name,
        device=args.device,
        input_sr=args.sample_rate,
    )


def embed_file(model, path: Path, sample_rate: int, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        wav = load_audio(path, sample_rate=sample_rate, device=device)
        emb = model.get_embedding(wav)
    return F.normalize(emb.detach().cpu(), p=2, dim=1).squeeze(0)


def model_similarity_score(model_name: str, cosine: float) -> float:
    if model_name in {"ecapa", "xvector"}:
        return (cosine + 1.0) / 2.0
    return cosine


def summarize_scores(
    rows: list[dict[str, object]],
    eval_set: str,
    model_name: str,
    unmatched: int,
    failed: int,
) -> dict[str, object]:
    scores = [float(row["similarity"]) for row in rows]
    cosines = [float(row["cosine"]) for row in rows]
    decisions = [int(row["same_speaker_pred"]) for row in rows]
    return {
        "eval_set": eval_set,
        "model": model_name,
        "matched": len(rows),
        "unmatched": unmatched,
        "failed": failed,
        "similarity_mean": mean(scores) if scores else None,
        "similarity_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "similarity_median": median(scores) if scores else None,
        "similarity_min": min(scores) if scores else None,
        "similarity_max": max(scores) if scores else None,
        "cosine_mean": mean(cosines) if cosines else None,
        "sva_rate": mean(decisions) if decisions else None,
        "asr_rate": 1.0 - mean(decisions) if decisions else None,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def error_row(
    pair: Pair,
    model_name: str,
    threshold: float,
    error: str,
) -> dict[str, object]:
    return {
        "eval_set": pair.eval_set,
        "model": model_name,
        "rel_path": pair.rel_path,
        "clean_path": str(pair.clean_path),
        "gen_path": str(pair.gen_path),
        "cosine": None,
        "similarity": None,
        "threshold": threshold,
        "same_speaker_pred": None,
        "error": error,
    }


def format_metric(value: object) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    recursive = not args.no_recursive
    clean_dir = args.clean_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not clean_dir.is_dir():
        raise SystemExit(f"Clean directory does not exist: {clean_dir}")

    eval_dirs = get_eval_dirs(args, recursive=recursive)
    if not eval_dirs:
        raise SystemExit(f"No generated audio directories found under: {args.gen_root}")

    pairs_by_set, unmatched_by_set = collect_pairs(
        clean_dir=clean_dir,
        eval_dirs=eval_dirs,
        recursive=recursive,
        match_mode=args.match,
        limit=args.limit,
    )
    total_pairs = sum(len(pairs) for pairs in pairs_by_set.values())
    if total_pairs == 0:
        raise SystemExit("No generated files could be matched to clean files.")

    print(f"Clean dir: {clean_dir}")
    print(f"Eval dirs: {', '.join(path.name for path in eval_dirs)}")
    print(f"Matched pairs: {total_pairs}")
    for eval_set, pairs in pairs_by_set.items():
        print(f"  {eval_set}: matched={len(pairs)} unmatched={unmatched_by_set[eval_set]}")

    if args.dry_run:
        return

    require_torch()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)
    thresholds = {
        "ecapa": args.ecapa_threshold,
        "xvector": args.xvector_threshold,
        "dvector": args.dvector_threshold,
    }
    clean_paths = sorted({pair.clean_path for pairs in pairs_by_set.values() for pair in pairs})

    detail_rows_by_set: dict[str, list[dict[str, object]]] = {
        eval_set: [] for eval_set in pairs_by_set
    }
    summary_rows_by_set: dict[str, list[dict[str, object]]] = {
        eval_set: [] for eval_set in pairs_by_set
    }

    for model_name in args.models:
        print(f"\nLoading {model_name} model on {args.device}")
        model = load_eval_model(model_name, args)

        print(f"Embedding {len(clean_paths)} clean files with {model_name}")
        clean_cache: dict[Path, tuple[torch.Tensor | None, str | None]] = {}
        for path in clean_paths:
            try:
                clean_cache[path] = (
                    embed_file(model, path, args.sample_rate, device),
                    None,
                )
            except Exception as exc:
                clean_cache[path] = (None, f"clean embedding failed: {exc}")

        threshold = thresholds[model_name]
        for eval_set, pairs in pairs_by_set.items():
            print(f"Scoring {eval_set} with {model_name} ({len(pairs)} pairs)")
            rows_for_summary: list[dict[str, object]] = []
            failed = 0
            for pair in pairs:
                clean_emb, clean_error = clean_cache[pair.clean_path]
                if clean_error is not None or clean_emb is None:
                    failed += 1
                    detail_rows_by_set[eval_set].append(
                        error_row(pair, model_name, threshold, clean_error or "clean embedding failed")
                    )
                    continue

                try:
                    gen_emb = embed_file(model, pair.gen_path, args.sample_rate, device)
                except Exception as exc:
                    failed += 1
                    detail_rows_by_set[eval_set].append(
                        error_row(pair, model_name, threshold, f"generated embedding failed: {exc}")
                    )
                    continue

                cosine = float(F.cosine_similarity(clean_emb, gen_emb, dim=0).item())
                similarity = model_similarity_score(model_name, cosine)
                same_speaker_pred = int(similarity >= threshold)
                row = {
                    "eval_set": eval_set,
                    "model": model_name,
                    "rel_path": pair.rel_path,
                    "clean_path": str(pair.clean_path),
                    "gen_path": str(pair.gen_path),
                    "cosine": cosine,
                    "similarity": similarity,
                    "threshold": threshold,
                    "same_speaker_pred": same_speaker_pred,
                    "error": "",
                }
                detail_rows_by_set[eval_set].append(row)
                rows_for_summary.append(row)

            summary_rows_by_set[eval_set].append(
                summarize_scores(
                    rows_for_summary,
                    eval_set=eval_set,
                    model_name=model_name,
                    unmatched=unmatched_by_set[eval_set],
                    failed=failed,
                )
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_fields = [
        "eval_set",
        "model",
        "rel_path",
        "clean_path",
        "gen_path",
        "cosine",
        "similarity",
        "threshold",
        "same_speaker_pred",
        "error",
    ]
    summary_fields = [
        "eval_set",
        "model",
        "matched",
        "unmatched",
        "failed",
        "similarity_mean",
        "similarity_std",
        "similarity_median",
        "similarity_min",
        "similarity_max",
        "cosine_mean",
        "sva_rate",
        "asr_rate",
    ]

    written_paths: list[Path] = []
    for eval_set in pairs_by_set:
        set_output_dir = output_dir / eval_set
        set_output_dir.mkdir(parents=True, exist_ok=True)

        detail_path = set_output_dir / "speaker_similarity_detail.csv"
        summary_csv_path = set_output_dir / "speaker_similarity_summary.csv"
        summary_json_path = set_output_dir / "speaker_similarity_summary.json"

        write_csv(detail_path, detail_rows_by_set[eval_set], detail_fields)
        write_csv(summary_csv_path, summary_rows_by_set[eval_set], summary_fields)
        summary_json_path.write_text(
            json.dumps(summary_rows_by_set[eval_set], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        written_paths.extend([detail_path, summary_csv_path, summary_json_path])

    print("\nSummary:")
    for eval_set in pairs_by_set:
        for row in summary_rows_by_set[eval_set]:
            print(
                f"  {row['eval_set']:12s} {row['model']:8s} "
                f"n={row['matched']} failed={row['failed']} "
                f"sim_mean={format_metric(row['similarity_mean'])} "
                f"sva={format_metric(row['sva_rate'])} "
                f"asr={format_metric(row['asr_rate'])}"
            )

    print("\nWrote per-directory outputs:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
