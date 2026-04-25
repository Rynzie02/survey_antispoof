"""
Calibrate SVA thresholds (EER) for xvector, ECAPA, and resemblyzer
on the VoxCeleb1 test set zip.

Usage:
    python calibrate_thresholds.py [--zip /path/to/vox1_test_wav.zip] [--device cuda:5]
"""

import argparse
import io
import random
import zipfile
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

ENROLL_PER_SPEAKER = 10
IMPOSTOR_PAIRS_PER_SPEAKER = 10
SAMPLE_RATE = 16000
AUDIO_SECONDS = 3.0
AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_SECONDS)


def load_wav_from_zip(zf, path, target_sr=SAMPLE_RATE, length=AUDIO_SAMPLES):
    with zf.open(path) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(1)  # stereo -> mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] < length:
        wav = F.pad(wav, (0, length - wav.shape[0]))
    else:
        wav = wav[:length]
    return wav  # (T,)


def compute_eer(scores, labels):
    """scores: array of cosine similarities, labels: 1=target, 0=impostor"""
    scores, labels = np.array(scores), np.array(labels)
    thresholds = np.sort(np.unique(scores))
    best_eer, best_thresh = 1.0, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        far = np.mean(preds[labels == 0])   # false accept
        frr = np.mean(1 - preds[labels == 1])  # false reject
        eer = (far + frr) / 2
        if eer < best_eer:
            best_eer, best_thresh = eer, t
    return best_eer, best_thresh


def get_speechbrain_embedding(model, wav, device):
    x = wav.unsqueeze(0).to(device)
    feats = model._classifier.mods.compute_features(x)
    feats = model._classifier.mods.mean_var_norm(feats, torch.ones(1, device=device))
    emb = model._encoder(feats).squeeze(1)
    return F.normalize(emb, p=2, dim=1).squeeze(0).cpu()


def get_resemblyzer_embedding(encoder, wav, sr=SAMPLE_RATE):
    from resemblyzer import preprocess_wav
    wav_np = wav.numpy().astype(np.float32)
    wav_np = preprocess_wav(wav_np, source_sr=sr)
    if wav_np.size == 0:
        return None
    return torch.tensor(encoder.embed_utterance(wav_np))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="/mnt/wht/tts_related/antipurify/voxceleb1/vox1/vox1_test_wav.zip")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- collect wav paths per speaker ---
    zf = zipfile.ZipFile(args.zip)
    spk_wavs = defaultdict(list)
    for name in zf.namelist():
        if name.endswith(".wav"):
            spk = name.split("/")[1]
            spk_wavs[spk].append(name)
    speakers = sorted(spk_wavs.keys())
    print(f"Speakers: {len(speakers)}, loading models...")

    # --- load models ---
    import sys, os
    _here = os.path.dirname(__file__)
    sys.path.insert(0, _here)
    from models.speaker_model import ECAPASpeakerEncoder, XVectorSpeakerEncoder
    from resemblyzer import VoiceEncoder

    ecapa = ECAPASpeakerEncoder(device=device)
    xvec = XVectorSpeakerEncoder(device=device)
    resemblyzer_enc = VoiceEncoder(device=str(device), verbose=False)
    print("Models loaded.")

    # --- build enrollment centroids and dev trial lists ---
    ecapa_scores, ecapa_labels = [], []
    xvec_scores, xvec_labels = [], []
    resem_scores, resem_labels = [], []

    spk_ecapa_centroids = {}
    spk_xvec_centroids = {}
    spk_resem_centroids = {}
    spk_test_wavs = {}

    print("Computing enrollment centroids...")
    for spk in speakers:
        wavs = spk_wavs[spk]
        random.shuffle(wavs)
        enroll_paths = wavs[:ENROLL_PER_SPEAKER]
        test_paths = wavs[ENROLL_PER_SPEAKER:]
        if not test_paths:
            continue

        enroll_wavs = torch.stack([load_wav_from_zip(zf, p) for p in enroll_paths])  # (N, T)

        with torch.no_grad():
            ec = get_speechbrain_embedding(ecapa, enroll_wavs[0], device)
            ec_all = torch.stack([get_speechbrain_embedding(ecapa, enroll_wavs[i], device) for i in range(len(enroll_wavs))])
            spk_ecapa_centroids[spk] = F.normalize(ec_all.mean(0), p=2, dim=0)

            xv_all = torch.stack([get_speechbrain_embedding(xvec, enroll_wavs[i], device) for i in range(len(enroll_wavs))])
            spk_xvec_centroids[spk] = F.normalize(xv_all.mean(0), p=2, dim=0)

        re_all = [get_resemblyzer_embedding(resemblyzer_enc, enroll_wavs[i]) for i in range(len(enroll_wavs))]
        re_all = [e for e in re_all if e is not None]
        if re_all:
            spk_resem_centroids[spk] = F.normalize(torch.stack(re_all).mean(0), p=2, dim=0)

        spk_test_wavs[spk] = test_paths

    print("Computing trial scores...")
    spk_list = [s for s in speakers if s in spk_test_wavs and s in spk_ecapa_centroids]

    for i, spk in enumerate(spk_list):
        test_paths = spk_test_wavs[spk]

        # target trials (same speaker)
        for path in test_paths[:5]:
            wav = load_wav_from_zip(zf, path)
            with torch.no_grad():
                ec_emb = get_speechbrain_embedding(ecapa, wav, device)
                xv_emb = get_speechbrain_embedding(xvec, wav, device)
            re_emb = get_resemblyzer_embedding(resemblyzer_enc, wav)

            ecapa_scores.append(float(F.cosine_similarity(ec_emb.unsqueeze(0), spk_ecapa_centroids[spk].unsqueeze(0))))
            ecapa_labels.append(1)
            xvec_scores.append(float(F.cosine_similarity(xv_emb.unsqueeze(0), spk_xvec_centroids[spk].unsqueeze(0))))
            xvec_labels.append(1)
            if re_emb is not None and spk in spk_resem_centroids:
                resem_scores.append(float(F.cosine_similarity(re_emb.unsqueeze(0), spk_resem_centroids[spk].unsqueeze(0))))
                resem_labels.append(1)

        # impostor trials (different speakers)
        impostors = random.sample([s for s in spk_list if s != spk], min(IMPOSTOR_PAIRS_PER_SPEAKER, len(spk_list) - 1))
        for imp_spk in impostors:
            path = random.choice(spk_test_wavs[imp_spk])
            wav = load_wav_from_zip(zf, path)
            with torch.no_grad():
                ec_emb = get_speechbrain_embedding(ecapa, wav, device)
                xv_emb = get_speechbrain_embedding(xvec, wav, device)
            re_emb = get_resemblyzer_embedding(resemblyzer_enc, wav)

            ecapa_scores.append(float(F.cosine_similarity(ec_emb.unsqueeze(0), spk_ecapa_centroids[spk].unsqueeze(0))))
            ecapa_labels.append(0)
            xvec_scores.append(float(F.cosine_similarity(xv_emb.unsqueeze(0), spk_xvec_centroids[spk].unsqueeze(0))))
            xvec_labels.append(0)
            if re_emb is not None and spk in spk_resem_centroids:
                resem_scores.append(float(F.cosine_similarity(re_emb.unsqueeze(0), spk_resem_centroids[spk].unsqueeze(0))))
                resem_labels.append(0)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(spk_list)} speakers done")

    ecapa_eer, ecapa_thresh = compute_eer(ecapa_scores, ecapa_labels)
    xvec_eer, xvec_thresh = compute_eer(xvec_scores, xvec_labels)
    resem_eer, resem_thresh = compute_eer(resem_scores, resem_labels)

    print("\n=== Calibration Results ===")
    print(f"ECAPA:       EER={ecapa_eer:.4f}  threshold={ecapa_thresh:.4f}")
    print(f"XVector:     EER={xvec_eer:.4f}  threshold={xvec_thresh:.4f}")
    print(f"Resemblyzer: EER={resem_eer:.4f}  threshold={resem_thresh:.4f}")
    print("\nAdd to config.py:")
    print(f"    ecapa_sva_threshold = {ecapa_thresh:.4f}")
    print(f"    xvector_sva_threshold = {xvec_thresh:.4f}")
    print(f"    resemblyzer_sva_threshold = {resem_thresh:.4f}")


if __name__ == "__main__":
    main()
