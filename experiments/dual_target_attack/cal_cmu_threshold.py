#!/usr/bin/env python3
"""
Calibrate SVA thresholds (EER) for ECAPA, x-vector, and Resemblyzer
on a CMU-style speaker directory.

Expected layout:
    /mnt/wht/exp/test_900/
        bdl/*.wav
        slt/*.wav
        ...

By default, each speaker's first shuffled 10 utterances are used as enrollment,
and all remaining utterances are used for target and impostor trials.

Usage:
    python cal_cmu_threshold.py --device cuda:0
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_DATA_ROOT = Path("/mnt/wht/exp/test_900")
DEFAULT_ECAPA_PATH = SCRIPT_DIR / "models" / "ecapa_tdnn_pretrained"
DEFAULT_XVECTOR_PATH = SCRIPT_DIR / "models" / "xvector_tdnn_pretrained"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioItem:
    key: str
    speaker: str
    name: str
    path: Path | None = field(default=None, compare=False, hash=False)
    wav_bytes: bytes | None = field(default=None, compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class SpeakerSplit:
    enroll_paths: list[AudioItem]
    test_paths: list[AudioItem]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate ECAPA, x-vector, and Resemblyzer speaker verification "
            "thresholds on /mnt/wht/exp/test_900-style audio directories."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=3.0,
        help="Pad/truncate each utterance to this length. Use 0 for full utterances.",
    )
    parser.add_argument("--enroll-per-speaker", type=int, default=10)
    parser.add_argument(
        "--calibration-tail-per-speaker",
        type=int,
        default=0,
        help=(
            "Use only the last N utterances per speaker before enrollment/test "
            "splitting. 0 means use all available utterances."
        ),
    )
    parser.add_argument(
        "--target-trials-per-speaker",
        type=int,
        default=0,
        help="0 means use all remaining utterances after enrollment.",
    )
    parser.add_argument(
        "--impostor-pairs-per-speaker",
        type=int,
        default=0,
        help=(
            "0 means exhaustive impostor trials. Positive values mimic the "
            "original script by sampling that many impostor speakers per claimed speaker."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("ecapa", "xvector", "resemblyzer"),
        choices=("ecapa", "xvector", "resemblyzer", "dvector"),
        help="Models to calibrate. dvector is an alias of resemblyzer.",
    )
    parser.add_argument("--ecapa-model-path", type=Path, default=DEFAULT_ECAPA_PATH)
    parser.add_argument("--xvector-model-path", type=Path, default=DEFAULT_XVECTOR_PATH)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write calibration results as JSON.",
    )
    return parser.parse_args()


def normalize_model_names(model_names: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for name in model_names:
        name = "resemblyzer" if name == "dvector" else name
        if name not in normalized:
            normalized.append(name)
    return normalized


def maybe_tail(items: list[AudioItem], tail_per_speaker: int) -> list[AudioItem]:
    if tail_per_speaker > 0:
        return items[-tail_per_speaker:]
    return items


def collect_arrow_items(speaker_dir: Path, tail_per_speaker: int) -> list[AudioItem]:
    arrow_path = speaker_dir / "data-00000-of-00001.arrow"
    if not arrow_path.exists():
        return []

    try:
        import pyarrow.ipc as ipc
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Found HuggingFace Arrow files but pyarrow is not installed in this "
            "Python environment. Install pyarrow, or export the selected audio "
            "rows to a wav directory first."
        ) from exc

    with arrow_path.open("rb") as handle:
        table = ipc.open_stream(handle).read_all()

    start = max(0, table.num_rows - tail_per_speaker) if tail_per_speaker > 0 else 0
    length = tail_per_speaker if tail_per_speaker > 0 else table.num_rows
    rows = table.slice(start, length).to_pylist()

    items: list[AudioItem] = []
    for absolute_index, row in enumerate(rows, start=start):
        name = row.get("file") or row.get("audio", {}).get("path")
        if not name:
            name = f"{speaker_dir.name}-{absolute_index:05d}.wav"
        audio = row.get("audio") or {}
        wav_bytes = audio.get("bytes")
        if wav_bytes is None:
            continue
        items.append(
            AudioItem(
                key=f"{speaker_dir.name}/{absolute_index:05d}/{name}",
                speaker=speaker_dir.name,
                name=str(name),
                wav_bytes=wav_bytes,
            )
        )
    return items


def collect_speaker_wavs(
    data_root: Path,
    tail_per_speaker: int = 0,
) -> dict[str, list[AudioItem]]:
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    speaker_wavs: dict[str, list[AudioItem]] = {}
    for speaker_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        wav_paths = sorted(
            path
            for path in speaker_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if wav_paths:
            items = [
                AudioItem(
                    key=f"{speaker_dir.name}/{path.name}",
                    speaker=speaker_dir.name,
                    name=path.name,
                    path=path,
                )
                for path in wav_paths
            ]
            speaker_wavs[speaker_dir.name] = maybe_tail(items, tail_per_speaker)
            continue

        arrow_items = collect_arrow_items(speaker_dir, tail_per_speaker)
        if arrow_items:
            speaker_wavs[speaker_dir.name] = arrow_items
    return speaker_wavs


def build_splits(
    speaker_wavs: dict[str, list[AudioItem]],
    enroll_per_speaker: int,
    target_trials_per_speaker: int,
    rng: random.Random,
) -> dict[str, SpeakerSplit]:
    splits: dict[str, SpeakerSplit] = {}
    for speaker, wavs in speaker_wavs.items():
        wavs = list(wavs)
        rng.shuffle(wavs)
        if len(wavs) <= enroll_per_speaker:
            print(
                f"Skipping {speaker}: needs more than {enroll_per_speaker} wavs, "
                f"found {len(wavs)}"
            )
            continue

        enroll_paths = wavs[:enroll_per_speaker]
        test_paths = wavs[enroll_per_speaker:]
        if target_trials_per_speaker > 0:
            test_paths = test_paths[:target_trials_per_speaker]
        if test_paths:
            splits[speaker] = SpeakerSplit(
                enroll_paths=enroll_paths,
                test_paths=test_paths,
            )
    return splits


def load_wav(item: AudioItem, sample_rate: int, audio_samples: int | None) -> torch.Tensor:
    try:
        if item.wav_bytes is not None:
            data, sr = sf.read(io.BytesIO(item.wav_bytes), dtype="float32", always_2d=True)
        elif item.path is not None:
            data, sr = sf.read(str(item.path), dtype="float32", always_2d=True)
        else:
            raise ValueError(f"No audio payload for {item.key}")
        wav = torch.from_numpy(data.T).mean(dim=0)
    except Exception:
        if item.path is None:
            raise
        wav, sr = torchaudio.load(str(item.path))
        wav = wav.float().mean(dim=0)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    if audio_samples is not None:
        if wav.shape[0] < audio_samples:
            wav = F.pad(wav, (0, audio_samples - wav.shape[0]))
        else:
            wav = wav[:audio_samples]
    return wav.contiguous()


def compute_eer(scores: list[float], labels: list[int]) -> tuple[float, float, float, float]:
    """Return best_eer, threshold, far, frr on raw cosine scores."""
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)

    if not np.any(labels_np == 1) or not np.any(labels_np == 0):
        raise ValueError("EER needs both target and impostor trials")

    best_eer = float("inf")
    best_thresh = 0.0
    best_far = 0.0
    best_frr = 0.0

    for threshold in np.sort(np.unique(scores_np)):
        preds = (scores_np >= threshold).astype(np.int64)
        far = float(np.mean(preds[labels_np == 0]))
        frr = float(np.mean(1 - preds[labels_np == 1]))
        eer = (far + frr) / 2.0
        if eer < best_eer:
            best_eer = eer
            best_thresh = float(threshold)
            best_far = far
            best_frr = frr

    return best_eer, best_thresh, best_far, best_frr


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def load_model(model_name: str, args: argparse.Namespace, device: torch.device):
    if model_name == "resemblyzer":
        from resemblyzer import VoiceEncoder, preprocess_wav

        return {
            "encoder": VoiceEncoder(device=str(device), verbose=False),
            "preprocess_wav": preprocess_wav,
        }

    from models.speaker_model import ECAPASpeakerEncoder, XVectorSpeakerEncoder

    if model_name == "ecapa":
        return ECAPASpeakerEncoder(model_path=str(args.ecapa_model_path), device=device)
    if model_name == "xvector":
        return XVectorSpeakerEncoder(model_path=str(args.xvector_model_path), device=device)
    raise ValueError(f"Unsupported model: {model_name}")


def embed_wav(
    model_name: str,
    model,
    wav: torch.Tensor,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor | None:
    if model_name == "resemblyzer":
        wav_np = wav.detach().cpu().numpy().astype(np.float32)
        wav_np = model["preprocess_wav"](wav_np, source_sr=sample_rate)
        if wav_np.size == 0:
            return None
        emb = torch.tensor(model["encoder"].embed_utterance(wav_np), dtype=torch.float32)
        return F.normalize(emb, p=2, dim=0).cpu()

    with torch.no_grad():
        emb = model.get_embedding(wav.unsqueeze(0).to(device)).squeeze(0)
    return F.normalize(emb.detach().cpu(), p=2, dim=0)


def compute_embeddings(
    model_name: str,
    model,
    paths: list[AudioItem],
    sample_rate: int,
    audio_samples: int | None,
    device: torch.device,
) -> dict[AudioItem, torch.Tensor]:
    embeddings: dict[AudioItem, torch.Tensor] = {}
    for index, path in enumerate(paths, start=1):
        wav = load_wav(path, sample_rate=sample_rate, audio_samples=audio_samples)
        emb = embed_wav(model_name, model, wav, sample_rate=sample_rate, device=device)
        if emb is not None:
            embeddings[path] = emb
        if index % 100 == 0 or index == len(paths):
            print(f"    embedded {index}/{len(paths)} files")
    return embeddings


def build_centroids(
    splits: dict[str, SpeakerSplit],
    embeddings: dict[AudioItem, torch.Tensor],
) -> dict[str, torch.Tensor]:
    centroids: dict[str, torch.Tensor] = {}
    for speaker, split in splits.items():
        enroll_embeddings = [
            embeddings[path] for path in split.enroll_paths if path in embeddings
        ]
        if not enroll_embeddings:
            continue
        centroid = torch.stack(enroll_embeddings).mean(dim=0)
        centroids[speaker] = F.normalize(centroid, p=2, dim=0)
    return centroids


def score_trials(
    splits: dict[str, SpeakerSplit],
    centroids: dict[str, torch.Tensor],
    embeddings: dict[AudioItem, torch.Tensor],
    impostor_pairs_per_speaker: int,
    rng: random.Random,
) -> tuple[list[float], list[int], dict[str, int]]:
    scores: list[float] = []
    labels: list[int] = []
    counts = {"target": 0, "impostor": 0, "missing": 0}

    speakers = sorted(speaker for speaker in splits if speaker in centroids)

    for speaker in speakers:
        centroid = centroids[speaker]
        for path in splits[speaker].test_paths:
            emb = embeddings.get(path)
            if emb is None:
                counts["missing"] += 1
                continue
            scores.append(cosine(emb, centroid))
            labels.append(1)
            counts["target"] += 1

    for speaker in speakers:
        centroid = centroids[speaker]
        impostor_speakers = [candidate for candidate in speakers if candidate != speaker]
        if impostor_pairs_per_speaker > 0:
            impostor_speakers = rng.sample(
                impostor_speakers,
                min(impostor_pairs_per_speaker, len(impostor_speakers)),
            )
            for impostor_speaker in impostor_speakers:
                path = rng.choice(splits[impostor_speaker].test_paths)
                emb = embeddings.get(path)
                if emb is None:
                    counts["missing"] += 1
                    continue
                scores.append(cosine(emb, centroid))
                labels.append(0)
                counts["impostor"] += 1
        else:
            for impostor_speaker in impostor_speakers:
                for path in splits[impostor_speaker].test_paths:
                    emb = embeddings.get(path)
                    if emb is None:
                        counts["missing"] += 1
                        continue
                    scores.append(cosine(emb, centroid))
                    labels.append(0)
                    counts["impostor"] += 1

    return scores, labels, counts


def score_threshold_for_config(model_name: str, cosine_threshold: float) -> float:
    if model_name in {"ecapa", "xvector"}:
        return (cosine_threshold + 1.0) / 2.0
    return cosine_threshold


def calibrate_model(
    model_name: str,
    args: argparse.Namespace,
    splits: dict[str, SpeakerSplit],
    all_paths: list[AudioItem],
    audio_samples: int | None,
    device: torch.device,
) -> dict[str, object]:
    print(f"\nLoading {model_name} on {device}")
    model = load_model(model_name, args=args, device=device)

    print(f"Embedding {len(all_paths)} files with {model_name}")
    embeddings = compute_embeddings(
        model_name=model_name,
        model=model,
        paths=all_paths,
        sample_rate=args.sample_rate,
        audio_samples=audio_samples,
        device=device,
    )

    centroids = build_centroids(splits, embeddings)
    if not centroids:
        raise SystemExit(f"No enrollment centroids could be built for {model_name}")

    trial_rng = random.Random(args.seed)
    scores, labels, counts = score_trials(
        splits=splits,
        centroids=centroids,
        embeddings=embeddings,
        impostor_pairs_per_speaker=args.impostor_pairs_per_speaker,
        rng=trial_rng,
    )
    eer, cosine_threshold, far, frr = compute_eer(scores, labels)
    score_threshold = score_threshold_for_config(model_name, cosine_threshold)

    return {
        "eer": eer,
        "far": far,
        "frr": frr,
        "cosine_threshold": cosine_threshold,
        "score_threshold": score_threshold,
        "target_trials": counts["target"],
        "impostor_trials": counts["impostor"],
        "missing_trials": counts["missing"],
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
    }


def main() -> None:
    args = parse_args()
    models = normalize_model_names(args.models)
    rng = random.Random(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    audio_samples = (
        int(args.sample_rate * args.audio_seconds)
        if args.audio_seconds and args.audio_seconds > 0
        else None
    )

    speaker_wavs = collect_speaker_wavs(
        args.data_root,
        tail_per_speaker=args.calibration_tail_per_speaker,
    )
    splits = build_splits(
        speaker_wavs=speaker_wavs,
        enroll_per_speaker=args.enroll_per_speaker,
        target_trials_per_speaker=args.target_trials_per_speaker,
        rng=rng,
    )
    if len(splits) < 2:
        raise SystemExit("Need at least two speakers with enrollment and test audio")

    all_paths = sorted(
        {
            path
            for split in splits.values()
            for path in (*split.enroll_paths, *split.test_paths)
        },
        key=lambda item: item.key,
    )

    print(f"Data root: {args.data_root}")
    print(f"Speakers used: {len(splits)} / {len(speaker_wavs)}")
    print(f"Files used: {len(all_paths)}")
    print(f"Enrollment per speaker: {args.enroll_per_speaker}")
    if args.calibration_tail_per_speaker > 0:
        print(f"Calibration tail per speaker: {args.calibration_tail_per_speaker}")
    print(
        "Target trials per speaker: "
        f"{args.target_trials_per_speaker or 'all remaining'}"
    )
    print(
        "Impostor trials: "
        + (
            "exhaustive"
            if args.impostor_pairs_per_speaker == 0
            else f"{args.impostor_pairs_per_speaker} sampled per speaker"
        )
    )
    print(
        "Audio length: "
        + (f"{args.audio_seconds:.2f}s fixed" if audio_samples else "full utterance")
    )

    results = {
        "data_root": str(args.data_root),
        "sample_rate": args.sample_rate,
        "audio_seconds": args.audio_seconds if audio_samples else None,
        "seed": args.seed,
        "enroll_per_speaker": args.enroll_per_speaker,
        "calibration_tail_per_speaker": args.calibration_tail_per_speaker,
        "target_trials_per_speaker": args.target_trials_per_speaker,
        "impostor_pairs_per_speaker": args.impostor_pairs_per_speaker,
        "speakers_total": len(speaker_wavs),
        "speakers_used": len(splits),
        "files_used": len(all_paths),
        "models": {},
    }

    for model_name in models:
        results["models"][model_name] = calibrate_model(
            model_name=model_name,
            args=args,
            splits=splits,
            all_paths=all_paths,
            audio_samples=audio_samples,
            device=device,
        )

    print("\n=== CMU Calibration Results ===")
    for model_name in models:
        result = results["models"][model_name]
        label = "Resemblyzer" if model_name == "resemblyzer" else model_name.upper()
        print(
            f"{label:12s}: EER={result['eer']:.4f}  "
            f"FAR={result['far']:.4f}  FRR={result['frr']:.4f}  "
            f"cosine_threshold={result['cosine_threshold']:.4f}  "
            f"score_threshold={result['score_threshold']:.4f}  "
            f"target={result['target_trials']} impostor={result['impostor_trials']}"
        )

    print("\nAdd to config.py / evaluation args:")
    if "ecapa" in results["models"]:
        print(
            "    ecapa_sva_threshold = "
            f"{results['models']['ecapa']['score_threshold']:.4f}"
        )
    if "xvector" in results["models"]:
        print(
            "    xvector_sva_threshold = "
            f"{results['models']['xvector']['score_threshold']:.4f}"
        )
    if "resemblyzer" in results["models"]:
        print(
            "    resemblyzer_sva_threshold = "
            f"{results['models']['resemblyzer']['score_threshold']:.4f}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
