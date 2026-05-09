#!/usr/bin/env python3
"""Evaluate speaker similarity for /mnt/wht/exp/test_900_tts.

For each TTS/protection/variant directory, compare generated or purified audio
against the original clean audio in /mnt/wht/exp/test_900 using ECAPA-TDNN,
x-vector, and Resemblyzer speaker embeddings.

Each discovered variant directory is checked before matching/model loading and
must contain 900 audio files by default.

Default output layout:
  baseline/CosyVoice/Antifake/adv/speaker_similarity_detail.csv
  baseline/CosyVoice/Antifake/adv/speaker_similarity_summary.csv
  baseline/CosyVoice/Antifake/adv/speaker_similarity_summary.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_speaker_similarity import (  # noqa: E402
    AUDIO_EXTENSIONS,
    DEFAULT_ECAPA_PATH,
    DEFAULT_THRESHOLDS,
    DEFAULT_XVECTOR_PATH,
    Pair,
    build_clean_indexes,
    embed_file,
    format_metric,
    load_eval_model,
    match_clean_file,
    model_similarity_score,
    require_torch,
    write_csv,
)


DEFAULT_CLEAN_DIR = Path("/mnt/wht/exp/test_900")
DEFAULT_TTS_ROOT = Path("/mnt/wht/exp/test_900_tts")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "baseline"
DEFAULT_TTS = ("CosyVoice", "chatterbox", "f5tts", "qwen3-tts")
DEFAULT_PROTECTIONS = ("Antifake", "E2E-VGuard", "SafeSpeech")
DEFAULT_VARIANTS = ("adv", "DualPure", "audiopure", "phonepure", "wavepurifier")
DEFAULT_EXCLUDE_DIRS = (".*", "*-mfa", "*-textgrid", "logs", "text")
DEFAULT_EXPECTED_AUDIO_COUNT = 900
MODEL_ALIASES = {
    "ecapa": "ecapa",
    "ecapa-tdnn": "ecapa",
    "xvector": "xvector",
    "x-vector": "xvector",
    "resemblyzer": "dvector",
    "dvector": "dvector",
    "d-vector": "dvector",
}
MODEL_DISPLAY_NAMES = {
    "ecapa": "ecapa",
    "xvector": "xvector",
    "dvector": "resemblyzer",
}


@dataclass(frozen=True)
class EvalTarget:
    label: str
    tts: str
    protection: str
    variant: str
    path: Path
    audio_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare test_900_tts audio against /mnt/wht/exp/test_900 clean audio "
            "with ECAPA-TDNN, x-vector, and Resemblyzer speaker similarity."
        )
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help="Original clean audio directory. Default: /mnt/wht/exp/test_900",
    )
    parser.add_argument(
        "--tts-root",
        type=Path,
        default=DEFAULT_TTS_ROOT,
        help="Root containing TTS method directories. Default: /mnt/wht/exp/test_900_tts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root. Default: repo baseline/ directory.",
    )
    parser.add_argument(
        "--tts",
        nargs="+",
        default=list(DEFAULT_TTS),
        help="TTS method directories to evaluate.",
    )
    parser.add_argument(
        "--protections",
        nargs="+",
        default=list(DEFAULT_PROTECTIONS),
        help="Protection method directories to evaluate.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=(
            "Variant directories under each protection method. Default evaluates "
            "adv plus DualPure/audiopure/phonepure/wavepurifier."
        ),
    )
    parser.add_argument(
        "--all-tts",
        action="store_true",
        help="Evaluate every direct TTS directory under --tts-root.",
    )
    parser.add_argument(
        "--all-protections",
        action="store_true",
        help="Evaluate every direct protection directory under each selected TTS.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Evaluate every direct variant directory under each selected protection.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("ecapa", "xvector", "resemblyzer"),
        choices=tuple(MODEL_ALIASES),
        help="Speaker embedding models to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda, cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("GPU"),
        help=(
            "Physical GPU index in nvidia-smi PCI-bus order. Can also be set with "
            "GPU=1. When set, CUDA_VISIBLE_DEVICES is restricted to this GPU and "
            "the script uses cuda:0 inside the process."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate used before embedding extraction.",
    )
    parser.add_argument(
        "--match",
        choices=("auto", "relative", "filename", "stem", "normalized"),
        default="auto",
        help="How evaluated files are matched to clean files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of matched files per TTS/protection/variant.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_AUDIO_COUNT,
        help=(
            "Expected number of audio files in each discovered variant directory. "
            "Default: 900. Set to 0 to disable this check."
        ),
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Warn instead of exiting when a variant directory is not --expected-count.",
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
        default=DEFAULT_XVECTOR_PATH,
        help="Local x-vector SpeechBrain model directory.",
    )
    parser.add_argument(
        "--ecapa-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["ecapa"],
        help="Threshold for ECAPA same-speaker verification.",
    )
    parser.add_argument(
        "--xvector-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["xvector"],
        help="Threshold for x-vector same-speaker verification.",
    )
    parser.add_argument(
        "--resemblyzer-threshold",
        "--dvector-threshold",
        dest="dvector_threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["dvector"],
        help="Threshold for Resemblyzer same-speaker verification.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help=(
            "Directory name glob to skip while scanning evaluated audio. Defaults "
            "also skip hidden dirs, *-mfa, *-textgrid, logs, and text."
        ),
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not apply the default evaluated-directory exclude patterns.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly under each variant directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matched/unmatched files; do not load models.",
    )
    parser.add_argument(
        "--write-global-summary",
        action="store_true",
        help=(
            "Also write top-level speaker_similarity_detail/summary files under "
            "--output-dir. By default only per-variant directories are written."
        ),
    )
    return parser.parse_args()


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_child_dir(parent: Path, name: str) -> Path | None:
    exact = parent / name
    if exact.is_dir():
        return exact

    wanted = canonical_key(name)
    for child in sorted(parent.iterdir()) if parent.is_dir() else []:
        if child.is_dir() and canonical_key(child.name) == wanted:
            return child
    return None


def selected_child_dirs(parent: Path, names: Iterable[str], use_all: bool) -> list[Path]:
    if use_all:
        return [path for path in sorted(parent.iterdir()) if path.is_dir()]

    children: list[Path] = []
    for name in names:
        child = resolve_child_dir(parent, name)
        if child is None:
            print(f"Skipping missing directory: {parent / name}")
            continue
        children.append(child)
    return children


def iter_audio_files(
    root: Path,
    recursive: bool = True,
    exclude_dir_patterns: Iterable[str] = (),
) -> Iterable[Path]:
    if not recursive:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                yield path
        return

    patterns = tuple(exclude_dir_patterns)
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts[:-1]
        if any(
            fnmatch.fnmatch(part, pattern)
            for part in rel_parts
            for pattern in patterns
        ):
            continue
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def count_audio_files(
    root: Path,
    recursive: bool,
    exclude_dir_patterns: Iterable[str],
) -> int:
    return sum(
        1
        for _ in iter_audio_files(
            root,
            recursive=recursive,
            exclude_dir_patterns=exclude_dir_patterns,
        )
    )


def validate_audio_counts(
    targets: list[EvalTarget],
    expected_count: int,
    allow_mismatch: bool,
) -> None:
    if expected_count <= 0:
        print("Audio count check: disabled")
        return

    mismatches = [
        target for target in targets if target.audio_count != expected_count
    ]
    if not mismatches:
        print(
            f"Audio count check: all {len(targets)} variant directories have "
            f"{expected_count} audio files"
        )
        return

    lines = [
        "Audio count check failed:",
        f"expected {expected_count} audio files in each variant directory.",
    ]
    lines.extend(
        f"  {target.label}: found={target.audio_count} path={target.path}"
        for target in mismatches
    )
    message = "\n".join(lines)
    if allow_mismatch:
        print(f"WARNING: {message}")
        return
    raise SystemExit(message)


def discover_targets(
    tts_root: Path,
    tts_names: list[str],
    protection_names: list[str],
    variant_names: list[str],
    all_tts: bool,
    all_protections: bool,
    all_variants: bool,
    recursive: bool,
    exclude_dir_patterns: Iterable[str],
) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    tts_dirs = selected_child_dirs(tts_root, tts_names, use_all=all_tts)

    for tts_dir in tts_dirs:
        protection_dirs = selected_child_dirs(
            tts_dir,
            protection_names,
            use_all=all_protections,
        )
        for protection_dir in protection_dirs:
            variant_dirs = selected_child_dirs(
                protection_dir,
                variant_names,
                use_all=all_variants,
            )
            for variant_dir in variant_dirs:
                audio_count = count_audio_files(
                    variant_dir,
                    recursive=recursive,
                    exclude_dir_patterns=exclude_dir_patterns,
                )
                if audio_count == 0:
                    print(f"Skipping variant with no audio: {variant_dir}")
                    continue
                label = f"{tts_dir.name}/{protection_dir.name}/{variant_dir.name}"
                targets.append(
                    EvalTarget(
                        label=label,
                        tts=tts_dir.name,
                        protection=protection_dir.name,
                        variant=variant_dir.name,
                        path=variant_dir.resolve(),
                        audio_count=audio_count,
                    )
                )
    return targets


def collect_pairs(
    clean_dir: Path,
    targets: list[EvalTarget],
    recursive: bool,
    match_mode: str,
    limit: int | None,
    exclude_dir_patterns: Iterable[str],
) -> tuple[dict[str, list[Pair]], dict[str, int]]:
    clean_files = list(iter_audio_files(clean_dir, recursive=recursive))
    clean_indexes = build_clean_indexes(clean_files, clean_dir)
    pairs_by_label: dict[str, list[Pair]] = {}
    unmatched_by_label: dict[str, int] = {}

    for target in targets:
        pairs: list[Pair] = []
        unmatched = 0
        for eval_path in iter_audio_files(
            target.path,
            recursive=recursive,
            exclude_dir_patterns=exclude_dir_patterns,
        ):
            clean_path = match_clean_file(
                gen_path=eval_path,
                gen_dir=target.path,
                clean_dir=clean_dir,
                clean_indexes=clean_indexes,
                mode=match_mode,
            )
            if clean_path is None:
                unmatched += 1
                continue
            pairs.append(
                Pair(
                    eval_set=target.label,
                    clean_path=clean_path,
                    gen_path=eval_path,
                    rel_path=eval_path.relative_to(target.path).as_posix(),
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
        pairs_by_label[target.label] = pairs
        unmatched_by_label[target.label] = unmatched

    return pairs_by_label, unmatched_by_label


def parse_models(model_names: Iterable[str]) -> list[str]:
    models: list[str] = []
    for name in model_names:
        model = MODEL_ALIASES[name]
        if model not in models:
            models.append(model)
    return models


def safe_path_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def target_output_dir(output_dir: Path, target: EvalTarget) -> Path:
    return (
        output_dir
        / safe_path_part(target.tts)
        / safe_path_part(target.protection)
        / safe_path_part(target.variant)
    )


def read_target_map(targets: list[EvalTarget]) -> dict[str, EvalTarget]:
    return {target.label: target for target in targets}


def summarize_scores(
    rows: list[dict[str, object]],
    target: EvalTarget,
    model_name: str,
    pair_count: int,
    unmatched: int,
    failed: int,
) -> dict[str, object]:
    scores = [float(row["similarity"]) for row in rows]
    cosines = [float(row["cosine"]) for row in rows]
    decisions = [int(row["same_speaker_pred"]) for row in rows]
    return {
        "tts": target.tts,
        "protection": target.protection,
        "variant": target.variant,
        "eval_set": target.label,
        "model": model_name,
        "audio_count": target.audio_count,
        "pairs": pair_count,
        "scored": len(rows),
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


def make_error_row(
    pair: Pair,
    target: EvalTarget,
    model_name: str,
    threshold: float,
    error: str,
) -> dict[str, object]:
    return {
        "tts": target.tts,
        "protection": target.protection,
        "variant": target.variant,
        "eval_set": target.label,
        "model": model_name,
        "rel_path": pair.rel_path,
        "clean_path": str(pair.clean_path),
        "eval_path": str(pair.gen_path),
        "cosine": None,
        "similarity": None,
        "threshold": threshold,
        "same_speaker_pred": None,
        "error": error,
    }


def write_outputs(
    output_dir: Path,
    targets: list[EvalTarget],
    detail_rows_by_label: dict[str, list[dict[str, object]]],
    summary_rows: list[dict[str, object]],
    write_global_summary: bool,
) -> list[Path]:
    detail_fields = [
        "tts",
        "protection",
        "variant",
        "eval_set",
        "model",
        "rel_path",
        "clean_path",
        "eval_path",
        "cosine",
        "similarity",
        "threshold",
        "same_speaker_pred",
        "error",
    ]
    summary_fields = [
        "tts",
        "protection",
        "variant",
        "eval_set",
        "model",
        "audio_count",
        "pairs",
        "scored",
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

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    if write_global_summary:
        all_detail_rows = [
            row for target in targets for row in detail_rows_by_label[target.label]
        ]
        all_detail_path = output_dir / "speaker_similarity_detail.csv"
        all_summary_csv_path = output_dir / "speaker_similarity_summary.csv"
        all_summary_json_path = output_dir / "speaker_similarity_summary.json"

        write_csv(all_detail_path, all_detail_rows, detail_fields)
        write_csv(all_summary_csv_path, summary_rows, summary_fields)
        all_summary_json_path.write_text(
            json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written_paths.extend(
            [all_detail_path, all_summary_csv_path, all_summary_json_path]
        )

    for target in targets:
        set_output_dir = target_output_dir(output_dir, target)
        set_output_dir.mkdir(parents=True, exist_ok=True)
        target_summary_rows = [
            row for row in summary_rows if row["eval_set"] == target.label
        ]
        detail_path = set_output_dir / "speaker_similarity_detail.csv"
        summary_csv_path = set_output_dir / "speaker_similarity_summary.csv"
        summary_json_path = set_output_dir / "speaker_similarity_summary.json"

        write_csv(detail_path, detail_rows_by_label[target.label], detail_fields)
        write_csv(summary_csv_path, target_summary_rows, summary_fields)
        summary_json_path.write_text(
            json.dumps(target_summary_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written_paths.extend([detail_path, summary_csv_path, summary_json_path])
    return written_paths


def main() -> None:
    args = parse_args()
    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
        if args.device == "auto":
            args.device = "cuda:0"

    recursive = not args.no_recursive
    clean_dir = args.clean_dir.expanduser().resolve()
    tts_root = args.tts_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not clean_dir.is_dir():
        raise SystemExit(f"Clean directory does not exist: {clean_dir}")
    if not tts_root.is_dir():
        raise SystemExit(f"TTS root directory does not exist: {tts_root}")

    exclude_dir_patterns = []
    if not args.no_default_excludes:
        exclude_dir_patterns.extend(DEFAULT_EXCLUDE_DIRS)
    exclude_dir_patterns.extend(args.exclude_dir)

    targets = discover_targets(
        tts_root=tts_root,
        tts_names=args.tts,
        protection_names=args.protections,
        variant_names=args.variants,
        all_tts=args.all_tts,
        all_protections=args.all_protections,
        all_variants=args.all_variants,
        recursive=recursive,
        exclude_dir_patterns=exclude_dir_patterns,
    )
    if not targets:
        raise SystemExit("No evaluated audio directories found.")

    validate_audio_counts(
        targets=targets,
        expected_count=args.expected_count,
        allow_mismatch=args.allow_count_mismatch,
    )

    pairs_by_label, unmatched_by_label = collect_pairs(
        clean_dir=clean_dir,
        targets=targets,
        recursive=recursive,
        match_mode=args.match,
        limit=args.limit,
        exclude_dir_patterns=exclude_dir_patterns,
    )
    total_pairs = sum(len(pairs) for pairs in pairs_by_label.values())
    if total_pairs == 0:
        raise SystemExit("No evaluated files could be matched to clean files.")

    print(f"Clean dir: {clean_dir}")
    print(f"TTS root: {tts_root}")
    print(f"Output dir: {output_dir}")
    print(f"Exclude dirs: {', '.join(exclude_dir_patterns) or '(none)'}")
    print(f"Targets: {len(targets)}")
    print(f"Matched pairs: {total_pairs}")
    for target in targets:
        pairs = pairs_by_label[target.label]
        print(
            f"  {target.label}: audio_count={target.audio_count} matched={len(pairs)} "
            f"unmatched={unmatched_by_label[target.label]} path={target.path}"
        )

    if args.dry_run:
        return

    require_torch()
    import torch
    import torch.nn.functional as F

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)
    thresholds = {
        "ecapa": args.ecapa_threshold,
        "xvector": args.xvector_threshold,
        "dvector": args.dvector_threshold,
    }
    target_by_label = read_target_map(targets)
    clean_paths = sorted(
        {pair.clean_path for pairs in pairs_by_label.values() for pair in pairs}
    )
    model_names = parse_models(args.models)

    detail_rows_by_label: dict[str, list[dict[str, object]]] = {
        target.label: [] for target in targets
    }
    summary_rows: list[dict[str, object]] = []

    for internal_model_name in model_names:
        display_model_name = MODEL_DISPLAY_NAMES[internal_model_name]
        print(f"\nLoading {display_model_name} model on {args.device}")
        model = load_eval_model(internal_model_name, args)

        print(f"Embedding {len(clean_paths)} clean files with {display_model_name}")
        clean_cache = {}
        for path in clean_paths:
            try:
                clean_cache[path] = (
                    embed_file(model, path, args.sample_rate, device),
                    None,
                )
            except Exception as exc:
                clean_cache[path] = (None, f"clean embedding failed: {exc}")

        threshold = thresholds[internal_model_name]
        for target in targets:
            pairs = pairs_by_label[target.label]
            print(
                f"Scoring {target.label} with {display_model_name} "
                f"({len(pairs)} pairs)"
            )
            rows_for_summary: list[dict[str, object]] = []
            failed = 0
            for pair in pairs:
                clean_emb, clean_error = clean_cache[pair.clean_path]
                if clean_error is not None or clean_emb is None:
                    failed += 1
                    detail_rows_by_label[target.label].append(
                        make_error_row(
                            pair,
                            target,
                            display_model_name,
                            threshold,
                            clean_error or "clean embedding failed",
                        )
                    )
                    continue

                try:
                    eval_emb = embed_file(model, pair.gen_path, args.sample_rate, device)
                except Exception as exc:
                    failed += 1
                    detail_rows_by_label[target.label].append(
                        make_error_row(
                            pair,
                            target,
                            display_model_name,
                            threshold,
                            f"evaluated embedding failed: {exc}",
                        )
                    )
                    continue

                cosine = float(F.cosine_similarity(clean_emb, eval_emb, dim=0).item())
                similarity = model_similarity_score(internal_model_name, cosine)
                same_speaker_pred = int(similarity >= threshold)
                row = {
                    "tts": target.tts,
                    "protection": target.protection,
                    "variant": target.variant,
                    "eval_set": target.label,
                    "model": display_model_name,
                    "rel_path": pair.rel_path,
                    "clean_path": str(pair.clean_path),
                    "eval_path": str(pair.gen_path),
                    "cosine": cosine,
                    "similarity": similarity,
                    "threshold": threshold,
                    "same_speaker_pred": same_speaker_pred,
                    "error": "",
                }
                detail_rows_by_label[target.label].append(row)
                rows_for_summary.append(row)

            summary_rows.append(
                summarize_scores(
                    rows_for_summary,
                    target=target_by_label[target.label],
                    model_name=display_model_name,
                    pair_count=len(pairs),
                    unmatched=unmatched_by_label[target.label],
                    failed=failed,
                )
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    written_paths = write_outputs(
        output_dir=output_dir,
        targets=targets,
        detail_rows_by_label=detail_rows_by_label,
        summary_rows=summary_rows,
        write_global_summary=args.write_global_summary,
    )

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['eval_set']:36s} {row['model']:12s} "
            f"pairs={row['pairs']} failed={row['failed']} "
            f"sim_mean={format_metric(row['similarity_mean'])} "
            f"sva={format_metric(row['sva_rate'])} "
            f"asr={format_metric(row['asr_rate'])}"
        )

    print("\nWrote outputs:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
