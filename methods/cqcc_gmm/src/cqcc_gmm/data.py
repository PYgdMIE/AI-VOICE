from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
BONAFIDE_LABELS = {"bonafide", "real", "genuine", "human"}
SPOOF_LABELS = {"spoof", "fake", "ai", "synthetic"}


def collect_audio_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file extension: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)

    files = [
        file
        for file in root.rglob("*")
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files)


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    if value in BONAFIDE_LABELS:
        return "bonafide"
    if value in SPOOF_LABELS:
        return "spoof"
    raise ValueError(
        f"Unsupported label '{label}'. Use bonafide/real/human or spoof/fake/ai."
    )


def read_manifest(path: str | Path) -> tuple[list[Path], list[str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    delimiter = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
    files: list[Path] = []
    labels: list[str] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Manifest is empty.")
        lower_names = {name.lower(): name for name in reader.fieldnames}
        if "path" not in lower_names or "label" not in lower_names:
            raise ValueError("Manifest must contain 'path' and 'label' columns.")

        path_col = lower_names["path"]
        label_col = lower_names["label"]
        for row in reader:
            audio_path = Path(row[path_col]).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            files.append(audio_path)
            labels.append(normalize_label(row[label_col]))

    return files, labels


def split_by_label(files: Iterable[Path], labels: Iterable[str]) -> tuple[list[Path], list[Path]]:
    bonafide: list[Path] = []
    spoof: list[Path] = []
    for file, label in zip(files, labels, strict=True):
        normalized = normalize_label(label)
        if normalized == "bonafide":
            bonafide.append(file)
        else:
            spoof.append(file)

    if not bonafide:
        raise ValueError("No bonafide samples found.")
    if not spoof:
        raise ValueError("No spoof samples found.")
    return bonafide, spoof
