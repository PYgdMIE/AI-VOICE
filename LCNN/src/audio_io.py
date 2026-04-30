"""
Load audio without torchaudio.load (torchaudio 2.11+ may require torchcodec otherwise).

Primary: soundfile (fast). Fallback: librosa audioread / ffmpeg when libsndfile fails on some FLAC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _load_via_audioread_librosa(path: Path) -> tuple[torch.Tensor, int]:
    """Decode using librosa's audioread path (ffmpeg/gstreamer backends), not libsndfile."""
    import warnings

    from librosa.core.audio import __audioread_load, to_mono

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        y, sr = __audioread_load(str(path), 0, None, np.float32)
    if y.size == 0:
        raise RuntimeError(f"empty audio: {path}")
    if y.ndim == 2:
        y = to_mono(y)
    wav = torch.from_numpy(np.asarray(y, dtype=np.float32))
    return wav, int(sr)


def _load_via_ffmpeg_librosa(path: Path) -> tuple[torch.Tensor, int]:
    """Force ffmpeg decoder via audioread (helps when libsndfile flac fails)."""
    import warnings

    import audioread.ffdec
    import librosa

    aro = audioread.ffdec.FFmpegAudioFile(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        y, sr = librosa.load(aro, sr=None, mono=True)
    wav = torch.from_numpy(np.asarray(y, dtype=np.float32))
    return wav, int(sr)


def load_audio_file(path: str | Path) -> tuple[torch.Tensor, int]:
    """
    Returns mono waveform float32 tensor (time,) and sample rate.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".mp3":
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=True)
        wav = torch.from_numpy(np.asarray(y, dtype=np.float32))
        return wav, int(sr)

    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError):
        try:
            return _load_via_ffmpeg_librosa(path)
        except Exception:
            return _load_via_audioread_librosa(path)

    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data[:, 0]
    wav = torch.from_numpy(np.asarray(data, dtype=np.float32))
    return wav, int(sr)
