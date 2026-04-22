"""
Study repeated purification on adversarial samples.

For each sample pair (clean, adv), this script computes speaker similarity to the
original clean sample after k repeated purification passes:

1. sim(clean, purify^k(adv))
2. sim(clean, purify^k(clean))

The second curve is the control: it tells us how much repeated purification
itself destroys speaker identity, independent of the attack.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from config import config
from models.purification import load_purification_model
from models.speaker_model import load_speaker_model


DEFAULT_AUDIO_BASE = Path("/mnt/data/wht/antispoof/audio_diffattack_20260420_234725")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze repeated purification recovery of DiffAttack samples."
    )
    parser.add_argument(
        "--audio-base-dir",
        type=Path,
        default=DEFAULT_AUDIO_BASE,
        help="Directory containing adv/ and optionally purified/ subdirectories.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Optional diffattack result json. If omitted, infer from audio-base-dir timestamp.",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=None,
        help="Fallback clean root if result json is unavailable.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Evaluate rounds k=0..max_rounds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
        help="Batch size for purification/evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of paired samples for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save csv/json/plot outputs.",
    )
    return parser.parse_args()


def infer_result_json(audio_base_dir: Path) -> Path | None:
    match = re.search(r"audio_diffattack_(\d{8}_\d{6})$", audio_base_dir.name)
    if not match:
        return None
    run_ts = match.group(1)
    return (
        Path(__file__).resolve().parent
        / "results"
        / "logs"
        / f"diffattack_attack_results_{run_ts}.json"
    )


def load_result_clean_paths(result_json: Path) -> list[Path]:
    with result_json.open() as f:
        data = json.load(f)

    clean_paths = []
    for item in data.get("per_sample_metrics", []):
        file_paths = item.get("file_paths", [])
        if isinstance(file_paths, list):
            for path in file_paths:
                clean_paths.append(Path(path))
    return clean_paths


def resolve_pairs(audio_base_dir: Path, result_json: Path | None, clean_root: Path | None):
    adv_dir = audio_base_dir / "adv"
    if not adv_dir.is_dir():
        raise FileNotFoundError(f"adv dir not found: {adv_dir}")

    pairs = []
    seen = set()

    if result_json is not None and result_json.exists():
        for clean_path in load_result_clean_paths(result_json):
            adv_path = adv_dir / f"adv_{clean_path.name}"
            if not adv_path.exists():
                continue
            key = (str(clean_path), str(adv_path))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "sample_name": clean_path.name,
                    "clean_path": clean_path,
                    "adv_path": adv_path,
                }
            )
        if pairs:
            return pairs

    if clean_root is None:
        clean_root = Path(config.data_root)

    if not clean_root.exists():
        raise FileNotFoundError(
            "Could not resolve clean pairs from result json, and clean root does not exist: "
            f"{clean_root}"
        )

    for adv_path in sorted(adv_dir.glob("adv_*.wav")):
        clean_name = adv_path.name.removeprefix("adv_")
        clean_path = clean_root / clean_name
        if clean_path.exists():
            pairs.append(
                {
                    "sample_name": clean_name,
                    "clean_path": clean_path,
                    "adv_path": adv_path,
                }
            )
    return pairs


def load_waveform(path: Path, sample_rate: int, audio_length: float) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    target_length = int(audio_length * sample_rate)
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

    return waveform.squeeze(0)


def cosine01(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (F.cosine_similarity(x, y, dim=1) + 1.0) / 2.0


def summarize_round(round_rows: list[dict]) -> dict:
    adv_vals = torch.tensor([row["adv_clean_sim"] for row in round_rows], dtype=torch.float32)
    clean_vals = torch.tensor(
        [row["clean_clean_sim"] for row in round_rows], dtype=torch.float32
    )

    return {
        "n": len(round_rows),
        "adv_clean_sim_mean": adv_vals.mean().item(),
        "adv_clean_sim_std": adv_vals.std(unbiased=False).item(),
        "adv_clean_sim_min": adv_vals.min().item(),
        "adv_clean_sim_max": adv_vals.max().item(),
        "clean_clean_sim_mean": clean_vals.mean().item(),
        "clean_clean_sim_std": clean_vals.std(unbiased=False).item(),
        "clean_clean_sim_min": clean_vals.min().item(),
        "clean_clean_sim_max": clean_vals.max().item(),
    }


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, summary_rows: list[dict]):
    rounds = [row["round"] for row in summary_rows]
    adv_mean = [row["adv_clean_sim_mean"] for row in summary_rows]
    clean_mean = [row["clean_clean_sim_mean"] for row in summary_rows]
    adv_std = [row["adv_clean_sim_std"] for row in summary_rows]
    clean_std = [row["clean_clean_sim_std"] for row in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, adv_mean, marker="o", label="sim(clean, purify^k(adv))")
    plt.plot(rounds, clean_mean, marker="o", label="sim(clean, purify^k(clean))")
    plt.fill_between(
        rounds,
        [m - s for m, s in zip(adv_mean, adv_std)],
        [m + s for m, s in zip(adv_mean, adv_std)],
        alpha=0.15,
    )
    plt.fill_between(
        rounds,
        [m - s for m, s in zip(clean_mean, clean_std)],
        [m + s for m, s in zip(clean_mean, clean_std)],
        alpha=0.15,
    )
    plt.xlabel("Purification Round k")
    plt.ylabel("Speaker Similarity to Original Clean")
    plt.title("Repeated Purification Study")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    audio_base_dir = args.audio_base_dir.resolve()
    result_json = args.result_json.resolve() if args.result_json else infer_result_json(audio_base_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (audio_base_dir / "repeated_purification_study").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = resolve_pairs(audio_base_dir, result_json, args.clean_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        raise RuntimeError("No clean/adv pairs found for repeated purification study.")

    print(f"[study] audio_base_dir={audio_base_dir}")
    print(f"[study] result_json={result_json if result_json else 'None'}")
    print(f"[study] resolved_pairs={len(pairs)}")
    print(f"[study] output_dir={output_dir}")
    print(f"[study] device={config.device}")
    print(f"[study] batch_size={args.batch_size}")
    print(f"[study] max_rounds={args.max_rounds}")

    speaker_model = load_speaker_model(
        model_path=config.speaker_model_path, device=config.device
    )
    purification_model = load_purification_model(
        model_type=config.purification_type,
        model_path=config.purification_model_path,
        device=config.device,
        reverse_timestep=config.purification_reverse_timestep,
        step_stride=config.purification_step_stride,
        use_checkpoint=config.purification_use_checkpoint,
    )

    per_sample_rows = []
    by_round = defaultdict(list)

    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), args.batch_size), desc="Repeated purification"):
            batch_pairs = pairs[start : start + args.batch_size]
            clean_batch = torch.stack(
                [
                    load_waveform(
                        pair["clean_path"], config.sample_rate, config.audio_length
                    )
                    for pair in batch_pairs
                ]
            ).to(config.device)
            adv_batch = torch.stack(
                [
                    load_waveform(pair["adv_path"], config.sample_rate, config.audio_length)
                    for pair in batch_pairs
                ]
            ).to(config.device)

            clean_embed = speaker_model.get_embedding(clean_batch)
            adv_curr = adv_batch
            clean_curr = clean_batch

            for round_idx in range(args.max_rounds + 1):
                if round_idx > 0:
                    adv_curr = purification_model(adv_curr)
                    clean_curr = purification_model(clean_curr)

                adv_embed = speaker_model.get_embedding(adv_curr)
                clean_round_embed = speaker_model.get_embedding(clean_curr)

                adv_sims = cosine01(adv_embed, clean_embed).cpu().tolist()
                clean_sims = cosine01(clean_round_embed, clean_embed).cpu().tolist()

                for pair, adv_sim, clean_sim in zip(batch_pairs, adv_sims, clean_sims):
                    row = {
                        "sample_name": pair["sample_name"],
                        "clean_path": str(pair["clean_path"]),
                        "adv_path": str(pair["adv_path"]),
                        "round": round_idx,
                        "adv_clean_sim": adv_sim,
                        "clean_clean_sim": clean_sim,
                    }
                    per_sample_rows.append(row)
                    by_round[round_idx].append(row)

    summary_rows = []
    round0_rows = by_round[0]
    round0_adv_mean = torch.tensor(
        [row["adv_clean_sim"] for row in round0_rows], dtype=torch.float32
    ).mean()

    for round_idx in range(args.max_rounds + 1):
        row = {"round": round_idx}
        row.update(summarize_round(by_round[round_idx]))
        row["adv_gain_vs_round0"] = row["adv_clean_sim_mean"] - round0_adv_mean.item()
        row["clean_loss_vs_identity"] = row["clean_clean_sim_mean"] - 1.0
        summary_rows.append(row)

    summary = {
        "audio_base_dir": str(audio_base_dir),
        "result_json": str(result_json) if result_json is not None else None,
        "num_pairs": len(pairs),
        "max_rounds": args.max_rounds,
        "batch_size": args.batch_size,
        "config": {
            "device": str(config.device),
            "sample_rate": config.sample_rate,
            "audio_length": config.audio_length,
            "speaker_model_path": config.speaker_model_path,
            "purification_model_path": config.purification_model_path,
            "purification_reverse_timestep": config.purification_reverse_timestep,
            "purification_step_stride": config.purification_step_stride,
        },
        "summary_by_round": summary_rows,
    }

    save_csv(
        output_dir / "per_sample_repeated_purification.csv",
        per_sample_rows,
        [
            "sample_name",
            "clean_path",
            "adv_path",
            "round",
            "adv_clean_sim",
            "clean_clean_sim",
        ],
    )
    save_csv(
        output_dir / "summary_repeated_purification.csv",
        summary_rows,
        [
            "round",
            "n",
            "adv_clean_sim_mean",
            "adv_clean_sim_std",
            "adv_clean_sim_min",
            "adv_clean_sim_max",
            "clean_clean_sim_mean",
            "clean_clean_sim_std",
            "clean_clean_sim_min",
            "clean_clean_sim_max",
            "adv_gain_vs_round0",
            "clean_loss_vs_identity",
        ],
    )
    with (output_dir / "summary_repeated_purification.json").open("w") as f:
        json.dump(summary, f, indent=2)
    save_plot(output_dir / "repeated_purification_curve.png", summary_rows)

    print("\n[study] summary")
    for row in summary_rows:
        print(
            f"  round={row['round']}: "
            f"adv->clean={row['adv_clean_sim_mean']:.4f} "
            f"(delta={row['adv_gain_vs_round0']:+.4f}) | "
            f"clean->clean={row['clean_clean_sim_mean']:.4f} "
            f"(delta={row['clean_loss_vs_identity']:+.4f})"
        )
    print(f"\n[study] wrote {output_dir / 'summary_repeated_purification.json'}")
    print(f"[study] wrote {output_dir / 'summary_repeated_purification.csv'}")
    print(f"[study] wrote {output_dir / 'per_sample_repeated_purification.csv'}")
    print(f"[study] wrote {output_dir / 'repeated_purification_curve.png'}")


if __name__ == "__main__":
    main()
