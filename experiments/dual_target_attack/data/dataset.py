"""Dataset loader for speaker-attack experiments."""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio
import os


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


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
        self.data_root = str(data_root)
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.split = split

        # Load audio file paths and labels
        self.audio_files, self.labels = self._load_dataset()

    def _load_dataset(self):
        """Load dataset file paths and labels"""
        print(f"Loading {self.split} dataset from {self.data_root}")

        data_path = Path(self.data_root).expanduser()
        if data_path.is_file():
            audio_files = self._read_audio_filelist(data_path)
        elif data_path.is_dir():
            audio_files = self._scan_audio_dir(data_path)
        else:
            return self._build_synthetic_index(reason="path is missing")

        if self.num_samples and self.num_samples > 0:
            audio_files = audio_files[: self.num_samples]

        if not audio_files:
            return self._build_synthetic_index(reason="no audio files were found")

        root_dir = data_path.resolve() if data_path.is_dir() else None
        loaded_from_filelist = data_path.is_file()
        parent_names = {Path(path).parent.name for path in audio_files}

        def parse_speaker(path):
            path_obj = Path(path)
            parent_name = path_obj.parent.name
            name = path_obj.name
            is_nested_dir_sample = (
                root_dir is not None
                and path_obj.parent.resolve() != root_dir
            )
            should_use_parent = (
                parent_name
                and parent_name != "."
                and (
                    loaded_from_filelist
                    or is_nested_dir_sample
                    or (len(parent_names) > 1 and "_" not in name)
                )
            )
            if should_use_parent:
                return parent_name

            parts = name.split("_")
            if len(parts) < 2:
                return name
            spk = parts[1]
            # librispeech: p3570-5694-0012 → p3570
            # voxceleb:    id10270-xxx     → id10270 (no change needed)
            return spk.split("-")[0]

        speakers = sorted(set(parse_speaker(f) for f in audio_files))
        spk2idx = {s: i for i, s in enumerate(speakers)}
        labels = [spk2idx[parse_speaker(f)] for f in audio_files]

        print(f"Loaded {len(audio_files)} audio files from {len(speakers)} speakers.")
        return audio_files, labels

    def _scan_audio_dir(self, data_path):
        """Recursively scan flat or speaker-nested audio directories."""
        return sorted(
            str(path)
            for path in data_path.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

    def _read_audio_filelist(self, filelist_path):
        """Read one audio path per line, optionally from pipe-delimited rows."""
        audio_files = []
        with filelist_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw_path = line.split("|", 1)[0].strip()
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = (filelist_path.parent / path).resolve()
                if path.suffix.lower() in AUDIO_EXTENSIONS:
                    audio_files.append(str(path))
        return sorted(audio_files)

    def _build_synthetic_index(self, reason):
        count = max(2, min(self.num_samples, 16))
        print(f"Dataset fallback enabled because {reason}; using {count} synthetic samples.")
        file_paths = [
            os.path.join(self.data_root, f"synthetic_spk{i % 4}_{i:03d}.wav")
            for i in range(count)
        ]
        labels = [i % 4 for i in range(count)]
        return file_paths, labels

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
            data, sr = sf.read(file_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
            waveform = waveform.squeeze(0).T
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
