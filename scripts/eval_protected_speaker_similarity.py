#!/usr/bin/env python3
"""Evaluate speaker similarity for test_900 protected audio sets.

The script compares audio under paths such as
  /mnt/wht/exp/test_900_protected/Antifake/adv
against the original clean files under
  /mnt/wht/exp/test_900

It reuses the ECAPA-TDNN, x-vector, and d-vector evaluation code from
scripts/eval_speaker_similarity.py, but adds method/variant discovery and stable
labels such as "Antifake/adv" so multiple adv directories do not overwrite each
other.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable


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
DEFAULT_PROTECTED_ROOT = Path("/mnt/wht/exp/test_900_protected")
DEFAULT_METHODS = ("Antifake", "E2E-VGuard", "SafeSpeech")
DEFAULT_VARIANTS = ("adv",)
DEFAULT_OUTPUT_DIR = DEFAULT_PROTECTED_ROOT / "speaker_similarity"
DEFAULT_EXCLUDE_DIRS = (".*", "*-mfa", "*-textgrid", "logs", "text")


@dataclass(frozen=True)
class EvalTarget:
    label: str
    method: str
    variant: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare protected/purified test_900 audio against original audio "
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
        "--protected-root",
        type=Path,
        default=DEFAULT_PROTECTED_ROOT,
        help="Root containing method directories. Default: /mnt/wht/exp/test_900_protected",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Attack/protection method directories to evaluate.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=(
            "Variant subdirectories under each method, for example: adv DualPure "
            "audiopure phonepure wavepurifier. Ignored when --all-variants is set."
        ),
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Evaluate every direct subdirectory under each selected method.",
    )
    parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Extra or standalone directory to evaluate. Use LABEL=PATH to control "
            "the output label. Can be passed multiple times."
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
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate used before embedding extraction.",
    )
    parser.add_argument(
        "--match",
        choices=("auto", "relative", "filename", "stem"),
        default="auto",
        help="How evaluated files are matched to clean files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of matched files per evaluated directory.",
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
        help="Only scan files directly under each directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matched/unmatched files; do not load models.",
    )
    return parser.parse_args()


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
        if any(
            fnmatch.fnmatch(part, pattern)
            for part in path.relative_to(root).parts[:-1]
            for pattern in patterns
        ):
            continue
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def has_audio(
    root: Path,
    recursive: bool,
    exclude_dir_patterns: Iterable[str],
) -> bool:
    return any(
        iter_audio_files(
            root,
            recursive=recursive,
            exclude_dir_patterns=exclude_dir_patterns,
        )
    )


def parse_eval_dir(value: str) -> EvalTarget:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        label = label.strip().strip("/")
    else:
        path = Path(value).expanduser().resolve()
        label = path.name

    if not label:
        label = path.name
    parts = label.split("/", 1)
    method = parts[0]
    variant = parts[1] if len(parts) > 1 else path.name
    return EvalTarget(label=label, method=method, variant=variant, path=path)


def discover_targets(
    protected_root: Path,
    methods: list[str],
    variants: list[str],
    all_variants: bool,
    extra_eval_dirs: list[str],
    recursive: bool,
    exclude_dir_patterns: Iterable[str],
) -> list[EvalTarget]:
    targets: list[EvalTarget] = []

    for method in methods:
        method_dir = protected_root / method
        if not method_dir.is_dir():
            print(f"Skipping missing method directory: {method_dir}")
            continue

        selected_variants = (
            [path.name for path in sorted(method_dir.iterdir()) if path.is_dir()]
            if all_variants
            else variants
        )
        for variant in selected_variants:
            path = method_dir / variant
            if not path.is_dir():
                print(f"Skipping missing variant directory: {path}")
                continue
            if not has_audio(path, recursive, exclude_dir_patterns):
                print(f"Skipping variant with no audio: {path}")
                continue
            targets.append(
                EvalTarget(
                    label=f"{method}/{variant}",
                    method=method,
                    variant=variant,
                    path=path.resolve(),
                )
            )

    targets.extend(parse_eval_dir(value) for value in extra_eval_dirs)
    return dedupe_labels(targets)


def dedupe_labels(targets: list[EvalTarget]) -> list[EvalTarget]:
    seen: dict[str, int] = {}
    unique_targets: list[EvalTarget] = []
    for target in targets:
        count = seen.get(target.label, 0)
        seen[target.label] = count + 1
        if count == 0:
            unique_targets.append(target)
            continue
        label = f"{target.label}#{count + 1}"
        unique_targets.append(
            EvalTarget(
                label=label,
                method=target.method,
                variant=target.variant,
                path=target.path,
            )
        )
    return unique_targets


def safe_label(label: str) -> str:
    return (
        label.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


def safe_path_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def target_output_dir(output_dir: Path, target: EvalTarget) -> Path:
    return output_dir / safe_path_part(target.method) / safe_path_part(target.variant)


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


def summarize_scores(
    rows: list[dict[str, object]],
    target: EvalTarget,
    model_name: str,
    unmatched: int,
    failed: int,
    pair_count: int,
) -> dict[str, object]:
    scores = [float(row["similarity"]) for row in rows]
    cosines = [float(row["cosine"]) for row in rows]
    decisions = [int(row["same_speaker_pred"]) for row in rows]
    return {
        "method": target.method,
        "variant": target.variant,
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


def read_target_map(targets: list[EvalTarget]) -> dict[str, EvalTarget]:
    return {target.label: target for target in targets}


def main() -> None:
    args = parse_args()
    recursive = not args.no_recursive
    clean_dir = args.clean_dir.expanduser().resolve()
    protected_root = args.protected_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not clean_dir.is_dir():
        raise SystemExit(f"Clean directory does not exist: {clean_dir}")

    exclude_dir_patterns = []
    if not args.no_default_excludes:
        exclude_dir_patterns.extend(DEFAULT_EXCLUDE_DIRS)
    exclude_dir_patterns.extend(args.exclude_dir)

    targets = discover_targets(
        protected_root=protected_root,
        methods=args.methods,
        variants=args.variants,
        all_variants=args.all_variants,
        extra_eval_dirs=args.eval_dir,
        recursive=recursive,
        exclude_dir_patterns=exclude_dir_patterns,
    )
    targets = [target for target in targets if target.path.is_dir()]
    if not targets:
        raise SystemExit("No evaluated audio directories found.")

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

    target_by_label = read_target_map(targets)
    print(f"Clean dir: {clean_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Exclude dirs: {', '.join(exclude_dir_patterns) or '(none)'}")
    print(f"Matched pairs: {total_pairs}")
    for target in targets:
        pairs = pairs_by_label[target.label]
        print(
            f"  {target.label}: matched={len(pairs)} "
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
    clean_paths = sorted(
        {pair.clean_path for pairs in pairs_by_label.values() for pair in pairs}
    )

    detail_rows_by_label: dict[str, list[dict[str, object]]] = {
        target.label: [] for target in targets
    }
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
        for target in targets:
            pairs = pairs_by_label[target.label]
            print(f"Scoring {target.label} with {model_name} ({len(pairs)} pairs)")
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
                    detail_rows_by_label[target.label].append(
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
                    "variant": target.variant,
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
                detail_rows_by_label[target.label].append(row)
                rows_for_summary.append(row)

            summary_rows.append(
                summarize_scores(
                    rows_for_summary,
                    target=target_by_label[target.label],
                    model_name=model_name,
                    unmatched=unmatched_by_label[target.label],
                    failed=failed,
                    pair_count=len(pairs),
                )
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_fields = [
        "method",
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
        "method",
        "variant",
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

    all_detail_rows = [
        row for target in targets for row in detail_rows_by_label[target.label]
    ]
    all_detail_path = output_dir / "speaker_similarity_detail.csv"
    summary_csv_path = output_dir / "speaker_similarity_summary.csv"
    summary_json_path = output_dir / "speaker_similarity_summary.json"

    write_csv(all_detail_path, all_detail_rows, detail_fields)
    write_csv(summary_csv_path, summary_rows, summary_fields)
    summary_json_path.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    written_paths = [all_detail_path, summary_csv_path, summary_json_path]
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

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['eval_set']:28s} {row['model']:8s} "
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
