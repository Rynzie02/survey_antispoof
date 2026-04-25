"""
Extract 20 test utterances per speaker (skipping first 10 enrollment utts)
from vox1_test_wav.zip -> test_800/wav_{spk}_{video}_{utt}.wav
"""

import io
import os
import random
import zipfile

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from collections import defaultdict

ZIP_PATH = "/mnt/wht/tts_related/antipurify/voxceleb1/vox1/vox1_test_wav.zip"
OUT_DIR = "/mnt/wht/tts_related/antipurify/voxceleb1/test_800"
ENROLL_PER_SPEAKER = 10
TEST_PER_SPEAKER = 20
SAMPLE_RATE = 16000
AUDIO_SAMPLES = int(SAMPLE_RATE * 3.0)
SEED = 42

random.seed(SEED)


def load_wav(zf, path):
    with zf.open(path) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] < AUDIO_SAMPLES:
        wav = F.pad(wav, (0, AUDIO_SAMPLES - wav.shape[0]))
    else:
        wav = wav[:AUDIO_SAMPLES]
    return wav


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    zf = zipfile.ZipFile(ZIP_PATH)

    spk_wavs = defaultdict(list)
    for name in zf.namelist():
        if name.endswith(".wav"):
            spk = name.split("/")[1]
            spk_wavs[spk].append(name)

    total = 0
    for spk in sorted(spk_wavs.keys()):
        wavs = sorted(spk_wavs[spk])
        random.shuffle(wavs)
        test_paths = wavs[ENROLL_PER_SPEAKER: ENROLL_PER_SPEAKER + TEST_PER_SPEAKER]
        if len(test_paths) < TEST_PER_SPEAKER:
            print(f"WARNING: {spk} only has {len(test_paths)} test utts (need {TEST_PER_SPEAKER})")

        for path in test_paths:
            parts = path.split("/")  # wav/{spk}/{video}/{utt}.wav
            fname = f"wav_{parts[1]}_{parts[2]}_{parts[3]}"
            out_path = os.path.join(OUT_DIR, fname)
            wav = load_wav(zf, path)
            sf.write(out_path, wav.numpy(), SAMPLE_RATE)
            total += 1

    print(f"Done. Saved {total} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
