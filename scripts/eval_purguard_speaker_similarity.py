#!/usr/bin/env python3
"""Evaluate PurGuard audio against the original test_900 speakers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

from eval_speaker_similarity import (
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
DEFAULT_PURGUARD_DIR = Path(
    "/mnt/wht/exp/dual_attack_outputs/test_900_protected/PurGuard"
)
DEFAULT_OUTPUT_DIR = (
    Path("/mnt/wht/exp/dual_attack_outputs/test_900_protected")
    / "PurGuard_speaker_similarity"
)


@dataclass(frozen=True)
class EvalTarget:
    label: str
    method: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PurGuard audio against original /mnt/wht/exp/test_900 audio "
            "with ECAPA-TDNN, x-vector, and d-vector speaker similarity."
        )
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help="Original clean audio directory. Default: /mnt/wht/exp/test_900",
    )
    parser.add_argument(
        "--purguard-dir",
        type=Path,
        default=DEFAULT_PURGUARD_DIR,
        help=(
            "PurGuard audio directory. Default: "
            "/mnt/wht/exp/dual_attack_outputs/test_900_protected/PurGuard"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for detail and summary CSV/JSON outputs.",
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
        choices=("auto", "relative", "filename", "stem"),
        default="auto",
        help="How PurGuard files are matched to clean files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of matched files to score.",
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
        "--dvector-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["dvector"],
        help="Threshold for d-vector same-speaker verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matched/unmatched files; do not load models.",
    )
    return parser.parse_args()


def iter_audio_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def collect_pairs(
    clean_dir: Path,
    purguard_dir: Path,
    match_mode: str,
    limit: int | None,
) -> tuple[list[Pair], int]:
    clean_files = list(iter_audio_files(clean_dir))
    clean_indexes = build_clean_indexes(clean_files, clean_dir)
    pairs: list[Pair] = []
    unmatched = 0

    for eval_path in iter_audio_files(purguard_dir):
        clean_path = match_clean_file(
            gen_path=eval_path,
            gen_dir=purguard_dir,
            clean_dir=clean_dir,
            clean_indexes=clean_indexes,
            mode=match_mode,
        )
        if clean_path is None:
            unmatched += 1
            continue
        pairs.append(
            Pair(
                eval_set="PurGuard",
                clean_path=clean_path,
                gen_path=eval_path,
                rel_path=eval_path.relative_to(purguard_dir).as_posix(),
            )
        )
        if limit is not None and len(pairs) >= limit:
            break

    return pairs, unmatched


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
        "method": target.method,
        "eval_set": target.label,
        "model": model_name,
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
        "method": target.method,
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


def main() -> None:
    args = parse_args()
    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
        if args.device == "auto":
            args.device = "cuda:0"

    clean_dir = args.clean_dir.expanduser().resolve()
    purguard_dir = args.purguard_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not clean_dir.is_dir():
        raise SystemExit(f"Clean directory does not exist: {clean_dir}")
    if not purguard_dir.is_dir():
        raise SystemExit(f"PurGuard directory does not exist: {purguard_dir}")

    target = EvalTarget(label="PurGuard", method="PurGuard", path=purguard_dir)
    pairs, unmatched = collect_pairs(
        clean_dir=clean_dir,
        purguard_dir=purguard_dir,
        match_mode=args.match,
        limit=args.limit,
    )
    if not pairs:
        raise SystemExit("No PurGuard files could be matched to clean files.")

    print(f"Clean dir: {clean_dir}")
    print(f"PurGuard dir: {purguard_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Unmatched files: {unmatched}")
    if args.gpu is not None and str(args.gpu).strip() != "":
        print(
            "GPU: "
            f"physical={str(args.gpu).strip()} "
            f"CUDA_DEVICE_ORDER={os.environ['CUDA_DEVICE_ORDER']} "
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"process_device={args.device}"
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
    clean_paths = sorted({pair.clean_path for pair in pairs})
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for model_name in args.models:
        print(f"\nLoading {model_name} model on {args.device}")
        model = load_eval_model(model_name, args)

        print(f"Embedding {len(clean_paths)} clean files with {model_name}")
        clean_cache = {}
        for path in clean_paths:
            try:
                clean_cache[path] = (
                    embed_file(model, path, args.sample_rate, device),
                    None,
                )
            except Exception as exc:
                clean_cache[path] = (None, f"clean embedding failed: {exc}")

        threshold = thresholds[model_name]
        rows_for_summary: list[dict[str, object]] = []
        failed = 0

        print(f"Scoring PurGuard with {model_name} ({len(pairs)} pairs)")
        for pair in pairs:
            clean_emb, clean_error = clean_cache[pair.clean_path]
            if clean_error is not None or clean_emb is None:
                failed += 1
                detail_rows.append(
                    make_error_row(
                        pair,
                        target,
                        model_name,
                        threshold,
                        clean_error or "clean embedding failed",
                    )
                )
                continue

            try:
                eval_emb = embed_file(model, pair.gen_path, args.sample_rate, device)
            except Exception as exc:
                failed += 1
                detail_rows.append(
                    make_error_row(
                        pair,
                        target,
                        model_name,
                        threshold,
                        f"evaluated embedding failed: {exc}",
                    )
                )
                continue

            cosine = float(F.cosine_similarity(clean_emb, eval_emb, dim=0).item())
            similarity = model_similarity_score(model_name, cosine)
            same_speaker_pred = int(similarity >= threshold)
            row = {
                "method": target.method,
                "eval_set": target.label,
                "model": model_name,
                "rel_path": pair.rel_path,
                "clean_path": str(pair.clean_path),
                "eval_path": str(pair.gen_path),
                "cosine": cosine,
                "similarity": similarity,
                "threshold": threshold,
                "same_speaker_pred": same_speaker_pred,
                "error": "",
            }
            detail_rows.append(row)
            rows_for_summary.append(row)

        summary_rows.append(
            summarize_scores(
                rows_for_summary,
                target=target,
                model_name=model_name,
                pair_count=len(pairs),
                unmatched=unmatched,
                failed=failed,
            )
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_fields = [
        "method",
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
        "method",
        "eval_set",
        "model",
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

    detail_path = output_dir / "speaker_similarity_detail.csv"
    summary_csv_path = output_dir / "speaker_similarity_summary.csv"
    summary_json_path = output_dir / "speaker_similarity_summary.json"
    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(summary_csv_path, summary_rows, summary_fields)
    summary_json_path.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['eval_set']:12s} {row['model']:8s} "
            f"pairs={row['pairs']} failed={row['failed']} "
            f"sim_mean={format_metric(row['similarity_mean'])} "
            f"sva={format_metric(row['sva_rate'])} "
            f"asr={format_metric(row['asr_rate'])}"
        )

    print("\nWrote outputs:")
    print(f"  {detail_path}")
    print(f"  {summary_csv_path}")
    print(f"  {summary_json_path}")


if __name__ == "__main__":
    main()
