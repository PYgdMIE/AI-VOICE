"""Shared LCNN forward from checkpoint (used by infer.py and eval scripts)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchaudio

from src.audio_io import load_audio_file
from src.model import LCNNSpoofDetector


def load_checkpoint_bundle(
    ckpt_path: str | Path, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module, str, int]:
    """Returns model, feature transform (mel or mfcc), feature name, sample_rate."""
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    n_fft = ckpt.get("n_fft", 512)
    hop = ckpt.get("hop", 160)
    sr = ckpt.get("sample_rate", 16000)
    feature = ckpt.get("feature")
    if feature is None:
        feature = "mfcc" if "n_mfcc" in ckpt else "mel"

    if feature == "mfcc":
        n_mfcc = ckpt.get("n_mfcc", 40)
        n_mel_filters = ckpt.get("n_mel_filters", 64)
        n_freq = n_mfcc
        transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop,
                "n_mels": n_mel_filters,
                "f_min": 20.0,
                "f_max": float(sr // 2),
            },
        )
    else:
        n_mels = ckpt.get("n_mels", 128)
        n_freq = n_mels
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=20,
            f_max=sr // 2,
        )

    model = LCNNSpoofDetector(n_freq_bins=n_freq).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    transform = transform.to(device)
    return model, transform, feature, sr


@torch.no_grad()
def predict_waveform(
    wav_1d: torch.Tensor,
    orig_sr: int,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    feature: str,
    target_sr: int,
    device: torch.device,
    max_seconds: float = 10.0,
) -> Tuple[float, float]:
    """Returns P(human), P(spoof)."""
    if orig_sr != target_sr:
        wav_1d = torchaudio.functional.resample(
            wav_1d.unsqueeze(0), orig_sr, target_sr
        ).squeeze(0)
    max_len = int(target_sr * max_seconds)
    if wav_1d.numel() > max_len:
        wav_1d = wav_1d[:max_len]
    m = transform(wav_1d.to(device))
    if feature == "mel":
        m = torch.log(m + 1e-6)
    m = m.unsqueeze(0).unsqueeze(1)
    logits = model(m)
    prob = torch.softmax(logits, dim=1)[0]
    return prob[0].item(), prob[1].item()


def predict_file(
    path: str | Path,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    feature: str,
    target_sr: int,
    device: torch.device,
) -> Tuple[float, float]:
    wav, sr = load_audio_file(path)
    return predict_waveform(wav, sr, model, transform, feature, target_sr, device)
