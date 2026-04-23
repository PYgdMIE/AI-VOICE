"""
Loads WAV from two folders: human (label 0) and ai (label 1),
or ASVspoof 2019 LA official protocols (bonafide=0, spoof=1).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from src.audio_io import load_audio_file


def load_waveform(
    path: Path,
    sample_rate: int,
    max_len: int,
) -> torch.Tensor:
    wav, sr = load_audio_file(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, sample_rate
        ).squeeze(0)
    if wav.numel() > max_len:
        start = torch.randint(0, wav.numel() - max_len + 1, (1,)).item()
        wav = wav[start : start + max_len]
    elif wav.numel() < max_len:
        pad = max_len - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav


class SpoofAudioDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16000,
        max_seconds: float = 4.0,
        extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3"),
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_seconds)
        self.items: List[Tuple[Path, int]] = []
        human_dir = self.root / "human"
        ai_dir = self.root / "ai"
        if not human_dir.is_dir() or not ai_dir.is_dir():
            raise FileNotFoundError(
                f"需要目录结构: {human_dir} 与 {ai_dir}，分别放真人/AI 音频"
            )
        for p in sorted(human_dir.rglob("*")):
            if p.suffix.lower() in extensions:
                self.items.append((p, 0))
        for p in sorted(ai_dir.rglob("*")):
            if p.suffix.lower() in extensions:
                self.items.append((p, 1))
        if not self.items:
            raise RuntimeError(f"在 {self.root} 下未找到音频文件")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        return load_waveform(path, self.sample_rate, self.max_len), label


def _parse_la_cm_line(line: str) -> Tuple[str, int] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    utt_id = parts[1]
    key = parts[-1].lower()
    if key == "bonafide":
        return utt_id, 0
    if key == "spoof":
        return utt_id, 1
    return None


def _read_protocol(protocol_path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    text = protocol_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parsed = _parse_la_cm_line(line)
        if parsed:
            rows.append(parsed)
    return rows


class ASVspoofLADataset(Dataset):
    """
    ASVspoof 2019 Logical Access corpus using official CM protocol files.

    ``la_root`` must contain (after unpacking LA.zip):
      ASVspoof2019_LA_train/, ASVspoof2019_LA_dev/,
      ASVspoof2019_LA_cm_protocols/

    Labels: bonafide -> 0 (human), spoof -> 1 (synthetic attack).
    """

    _PROTO_NAMES = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    }

    def __init__(
        self,
        la_root: str | Path,
        splits: Sequence[str] = ("train", "dev"),
        sample_rate: int = 16000,
        max_seconds: float = 4.0,
    ) -> None:
        self.la_root = Path(la_root).resolve()
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_seconds)
        self.items: List[Tuple[Path, int]] = []

        proto_dir = self.la_root / "ASVspoof2019_LA_cm_protocols"
        if not proto_dir.is_dir():
            raise FileNotFoundError(
                f"Missing ASVspoof2019_LA_cm_protocols under {self.la_root}. "
                "Unpack LA.zip so these folders sit under the same parent."
            )

        for split in splits:
            split = split.lower().strip()
            if split not in self._PROTO_NAMES:
                raise ValueError(f"Unknown split {split!r}; use train and/or dev")
            proto_name = self._PROTO_NAMES[split]
            proto_path = proto_dir / proto_name
            if not proto_path.is_file():
                raise FileNotFoundError(f"Protocol not found: {proto_path}")
            subset_dir = self.la_root / f"ASVspoof2019_LA_{split}" / "flac"
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"Missing flac directory: {subset_dir}")

            for utt_id, label in _read_protocol(proto_path):
                flac_path = subset_dir / f"{utt_id}.flac"
                if not flac_path.is_file():
                    continue
                self.items.append((flac_path, label))

        if not self.items:
            raise RuntimeError(
                f"No usable utterances under {self.la_root}. Check unzip paths and protocols."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        return load_waveform(path, self.sample_rate, self.max_len), label
