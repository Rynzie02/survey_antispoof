"""
Utility functions for attacks
"""
import torch
import numpy as np

def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    max_val = torch.max(torch.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def compute_perturbation_norm(x_clean, x_adv, norm_type='l2'):
    """Compute perturbation norm"""
    perturbation = x_adv - x_clean

    if norm_type == 'l2':
        return torch.norm(perturbation, p=2, dim=1).mean().item()
    elif norm_type == 'linf':
        return torch.max(torch.abs(perturbation)).item()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
