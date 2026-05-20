#!/usr/bin/env python3
"""Clone Xiaomi TTS voices using paired reference audio and text files.

Default pairing:
  /home/wht/PyProj/survey_antispoof/aew/aew-a0001.wav
  /home/wht/PyProj/survey_antispoof/aew_text/000_aew-a0001.txt

The output directory contains only generated WAV files.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from scipy.io import wavfile
from scipy.signal import resample_poly


DEFAULT_AUDIO_DIR = Path("/home/wht/PyProj/survey_antispoof/aew")
DEFAULT_TEXT_DIR = Path("/home/wht/PyProj/survey_antispoof/aew_text")
DEFAULT_OUTPUT_DIR = Path("/home/wht/PyProj/survey_antispoof/outputs/aew_reference_text")
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-tts-voiceclone"
DEFAULT_TARGET_SR = 16000
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Xiaomi voice-clone TTS from paired reference audio/text."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory containing reference audio files.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=DEFAULT_TEXT_DIR,
        help="Directory containing reference text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated WAV files.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MIMO_BASE_URL", DEFAULT_BASE_URL),
        help="MiMo API base URL, for example https://token-plan-cn.xiaomimimo.com/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MIMO_API_KEY", "tp-ceg0j9si496jhfw7wmozgx2ena35x9to7gago7g4qvuj5t8w"),
        help="MiMo API key. Can also be set via MIMO_API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MiMo TTS model id.")
    parser.add_argument(
        "--target-sr",
        type=int,
        default=DEFAULT_TARGET_SR,
        help="Output sample rate after local resampling.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of pairs to process. 0 means all.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP read timeout in seconds for each request.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="HTTP connect timeout in seconds for each request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient HTTP failures.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each successful request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist in the output directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next pair after one request fails.",
    )
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def text_key(path: Path) -> str:
    stem = path.stem
    return re.sub(r"^\d+_", "", stem)


def audio_text_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"^adv_", "", stem)
    return stem


def get_audio_files(audio_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(audio_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]


def get_text_map(text_dir: Path) -> dict[str, Path]:
    text_files = sorted(path for path in text_dir.iterdir() if path.is_file() and path.suffix == ".txt")
    mapping: dict[str, Path] = {}
    for path in text_files:
        key = text_key(path)
        if key in mapping:
            raise ValueError(f"Duplicate text key '{key}': {mapping[key]} and {path}")
        mapping[key] = path
    return mapping


def mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed in {"audio/wav", "audio/x-wav"}:
        return "audio/wav"
    if guessed in {"audio/mpeg", "audio/mp3"}:
        return "audio/mpeg"
    raise ValueError(f"Unsupported reference audio type: {path}")


def encode_reference_audio(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    payload = f"data:{mime_type_for_path(path)};base64,{b64}"
    if len(payload.encode("utf-8")) > 10 * 1024 * 1024:
        raise ValueError(f"Reference audio is too large after base64 encoding: {path}")
    return payload


def get_path(obj: Any, path: Iterable[Any]) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
    return cur


def extract_audio_b64(payload: dict[str, Any]) -> str:
    candidate_paths = [
        ("choices", 0, "message", "audio", "data"),
        ("choices", 0, "message", "audio"),
        ("audio", "data"),
        ("audio",),
    ]
    for path in candidate_paths:
        value = get_path(payload, path)
        if isinstance(value, str) and value:
            return value
    raise KeyError("Could not find base64 audio data in API response.")


def request_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: tuple[float, float],
    retries: int,
) -> requests.Response:
    for attempt in range(retries + 1):
        started_at = time.monotonic()
        print(f"  API request attempt {attempt + 1}/{retries + 1}...", flush=True)
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = time.monotonic() - started_at
            if response.status_code < 400:
                print(f"  API response OK in {elapsed:.1f}s", flush=True)
                return response

            should_retry = response.status_code in {408, 425, 429, 500, 502, 503, 504}
            if not should_retry or attempt == retries:
                raise RuntimeError(
                    f"HTTP {response.status_code} from MiMo API: {response.text[:2000]}"
                )
            wait_seconds = min(2**attempt, 8)
            print(
                f"  API returned HTTP {response.status_code} in {elapsed:.1f}s; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == retries:
                raise
            wait_seconds = min(2**attempt, 8)
            elapsed = time.monotonic() - started_at
            print(
                f"  API request failed after {elapsed:.1f}s: {exc}; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Unexpected retry loop exit")


def call_xiaomi_tts(
    *,
    base_url: str,
    api_key: str,
    model: str,
    reference_audio: str,
    text: str,
    timeout: tuple[float, float],
    retries: int,
) -> bytes:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": text},
        ],
        "audio": {
            "format": "wav",
            "voice": reference_audio,
        },
    }
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    response = request_with_retries(
        url=normalize_base_url(base_url),
        headers=headers,
        payload=payload,
        timeout=timeout,
        retries=retries,
    )
    data = response.json()
    return base64.b64decode(extract_audio_b64(data))


def decode_wav_bytes(audio_bytes: bytes) -> tuple[int, np.ndarray]:
    with io.BytesIO(audio_bytes) as buffer:
        sample_rate, audio = wavfile.read(buffer)
    return int(sample_rate), audio


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def resample_to_target(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(source_sr, target_sr)
    up = target_sr // gcd
    down = source_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def write_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = np.round(clipped * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, int16_audio)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set --api-key or MIMO_API_KEY.")

    audio_dir = args.audio_dir.expanduser().resolve()
    text_dir = args.text_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory does not exist: {audio_dir}")
    if not text_dir.is_dir():
        raise SystemExit(f"Text directory does not exist: {text_dir}")

    text_map = get_text_map(text_dir)
    pairs: list[tuple[Path, Path]] = []
    for audio_path in get_audio_files(audio_dir):
        text_path = text_map.get(audio_text_key(audio_path))
        if text_path is None:
            print(f"Skip missing text for {audio_path.name}", flush=True)
            continue
        pairs.append((audio_path, text_path))

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No audio/text pairs found.")

    for index, (audio_path, text_path) in enumerate(pairs):
        output_path = output_dir / f"{audio_path.stem}.wav"
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"[{index + 1}/{len(pairs)}] skip empty text: {text_path.name}", flush=True)
            continue
        if output_path.exists() and not args.overwrite:
            print(f"[{index + 1}/{len(pairs)}] skip existing: {output_path}", flush=True)
            continue

        print(f"[{index + 1}/{len(pairs)}] {audio_path.name} + {text_path.name}", flush=True)
        try:
            reference_audio = encode_reference_audio(audio_path)
            audio_bytes = call_xiaomi_tts(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                reference_audio=reference_audio,
                text=text,
                timeout=(args.connect_timeout, args.timeout),
                retries=args.retries,
            )
            source_sr, audio = decode_wav_bytes(audio_bytes)
            audio = to_mono_float32(audio)
            audio = resample_to_target(audio, source_sr, args.target_sr)
            write_wav(output_path, args.target_sr, audio)
            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"  ERROR: {exc}", flush=True)

    print(f"Done. WAV files saved to: {output_dir}")


if __name__ == "__main__":
    main()
