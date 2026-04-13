"""
Speaker recognition model implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class ResNetSpeakerModel(nn.Module):
    """ResNet-based speaker recognition model"""

    def __init__(self, num_speakers=1251, embedding_dim=512):
        super().__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim

        # Feature extractor (simplified ResNet34)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # Pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embedding = nn.Linear(512, embedding_dim)

        # Classification head
        self.fc_classifier = nn.Linear(embedding_dim, num_speakers)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def extract_features(self, x):
        """Extract mel-spectrogram features"""
        # x: (batch, samples)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=80
        ).to(x.device)

        features = mel_spec(x)  # (batch, n_mels, time)
        features = features.unsqueeze(1)  # (batch, 1, n_mels, time)
        features = torch.log(features + 1e-8)  # Log mel-spectrogram
        return features

    def forward(self, x, return_embedding=False):
        """
        Args:
            x: Audio waveform (batch, samples)
            return_embedding: If True, return embedding instead of logits
        Returns:
            logits or embedding
        """
        # Extract features
        features = self.extract_features(x)

        # Forward through ResNet
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Embedding
        embedding = self.fc_embedding(x)
        embedding = F.normalize(embedding, p=2, dim=1)

        if return_embedding:
            return embedding

        # Classification
        logits = self.fc_classifier(embedding)
        return logits

    def get_embedding(self, x):
        """Get speaker embedding"""
        return self.forward(x, return_embedding=True)


def load_speaker_model(model_path=None, num_speakers=1251, device='cuda'):
    """Load pretrained speaker recognition model"""
    model = ResNetSpeakerModel(num_speakers=num_speakers)

    if model_path and torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded speaker model from {model_path}")
        except:
            print(f"Could not load model from {model_path}, using random initialization")
    else:
        print("Using randomly initialized speaker model")

    model = model.to(device)
    model.eval()
    return model
