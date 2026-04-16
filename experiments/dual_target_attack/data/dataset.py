"""
Dataset loader for VoxCeleb
"""

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import os
import random


class VoxCelebDataset(Dataset):
    """VoxCeleb dataset for speaker recognition"""

    def __init__(
        self,
        data_root,
        num_samples=100,
        audio_length=3.0,
        sample_rate=16000,
        split="test",
    ):
        self.data_root = data_root
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.split = split

        # Load audio file paths and labels
        self.audio_files, self.labels = self._load_dataset()

    def _load_dataset(self):
        """Load dataset file paths and labels"""
        print(f"Loading {self.split} dataset from {self.data_root}")

        # Support both flat dir and audio/ subdir
        audio_dir = self.data_root
        if not any(f.endswith(".wav") for f in os.listdir(audio_dir)):
            sub = os.path.join(audio_dir, "audio")
            if os.path.isdir(sub):
                audio_dir = sub

        wav_files = sorted(
            [
                os.path.join(audio_dir, f)
                for f in os.listdir(audio_dir)
                if f.endswith(".wav")
            ]
        )[: self.num_samples]

        def parse_speaker(path):
            name = os.path.basename(path)
            parts = name.split("_")
            if len(parts) < 2:
                return name
            spk = parts[1]
            # librispeech: p3570-5694-0012 → p3570
            # voxceleb:    id10270-xxx     → id10270 (no change needed)
            return spk.split("-")[0]

        speakers = sorted(set(parse_speaker(f) for f in wav_files))
        spk2idx = {s: i for i, s in enumerate(speakers)}
        labels = [spk2idx[parse_speaker(f)] for f in wav_files]

        return wav_files, labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Returns:
            audio: Audio waveform (samples,)
            label: Speaker label
            file_path: Audio file path
        """
        file_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio (or generate dummy audio for demonstration)
        if os.path.exists(file_path):
            data, sr = sf.read(file_path, dtype="float32")
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )
        else:
            # Generate dummy audio
            num_samples = int(self.audio_length * self.sample_rate)
            waveform = torch.randn(1, num_samples) * 0.1

        # Ensure correct length
        target_length = int(self.audio_length * self.sample_rate)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio = waveform.squeeze(0)  # (samples,)

        return audio, label, file_path


def get_dataloader(config, split="test"):
    """Create dataloader for experiments"""

    dataset = VoxCelebDataset(
        data_root=config.data_root,
        num_samples=config.num_samples,
        audio_length=config.audio_length,
        sample_rate=config.sample_rate,
        split=split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return dataloader
