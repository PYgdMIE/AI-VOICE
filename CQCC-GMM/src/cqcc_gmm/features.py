from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import warnings

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct


@dataclass(frozen=True)
class CqccConfig:
    sample_rate: int = 16_000
    hop_ms: float = 10.0
    fmin: float = 15.0
    bins_per_octave: int = 96
    n_octaves: int = 9
    n_linear: int = 256
    n_ceps: int = 30
    delta_width: int = 9
    use_deltas: bool = True
    eps: float = 1e-10

    @property
    def hop_length(self) -> int:
        return max(1, round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def n_bins(self) -> int:
        return self.bins_per_octave * self.n_octaves


def load_audio(path: str | Path, config: CqccConfig) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as soundfile_error:
        audio = load_audio_with_ffmpeg(path, config, soundfile_error)
        sample_rate = config.sample_rate
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != config.sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=config.sample_rate)
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")
    return audio.astype(np.float32, copy=False)


def extract_cqcc(path: str | Path, config: CqccConfig | None = None) -> np.ndarray:
    cfg = config or CqccConfig()
    audio = load_audio(path, cfg)
    audio = preemphasis(audio)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"n_fft=.*is too large.*", category=UserWarning)
        cqt = librosa.cqt(
            audio,
            sr=cfg.sample_rate,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            n_bins=cfg.n_bins,
            bins_per_octave=cfg.bins_per_octave,
            filter_scale=1.0,
            pad_mode="reflect",
        )
    power = np.abs(cqt) ** 2

    cq_freqs = librosa.cqt_frequencies(
        n_bins=cfg.n_bins,
        fmin=cfg.fmin,
        bins_per_octave=cfg.bins_per_octave,
    )
    linear_freqs = np.linspace(cq_freqs[0], cq_freqs[-1], cfg.n_linear)
    linear_power = np.vstack(
        [np.interp(linear_freqs, cq_freqs, power[:, frame]) for frame in range(power.shape[1])]
    )

    log_power = np.log(linear_power + cfg.eps)
    cepstra = dct(log_power, type=2, norm="ortho", axis=1)[:, : cfg.n_ceps]
    cepstra = per_utterance_cmvn(cepstra)

    if cfg.use_deltas:
        width = min(cfg.delta_width, cepstra.shape[0] if cepstra.shape[0] % 2 else cepstra.shape[0] - 1)
        if width >= 3:
            delta = librosa.feature.delta(cepstra.T, width=width, order=1).T
            delta_delta = librosa.feature.delta(cepstra.T, width=width, order=2).T
            cepstra = np.concatenate([cepstra, delta, delta_delta], axis=1)

    finite_rows = np.isfinite(cepstra).all(axis=1)
    features = cepstra[finite_rows].astype(np.float32, copy=False)
    if features.size == 0:
        raise ValueError(f"No finite CQCC frames extracted from: {path}")
    return features


def per_utterance_cmvn(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def preemphasis(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    emphasized = np.empty_like(audio, dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - coefficient * audio[:-1]
    return emphasized


def load_audio_with_ffmpeg(path: str | Path, config: CqccConfig, original_error: Exception) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise original_error

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(config.sample_rate),
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed after soundfile failed for {path}: {stderr}") from original_error

    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"ffmpeg decoded no audio from: {path}") from original_error
    return audio


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()
