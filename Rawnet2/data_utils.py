import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac"}
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 64600


@dataclass(frozen=True)
class CMTrial:
    utt_id: str
    attack_id: str
    label: str

    @property
    def label_index(self):
        return 1 if self.label == "bonafide" else 0


def load_asvspoof_2019_protocol(protocol_path):
    protocol_path = Path(protocol_path)
    trials = []
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if len(tokens) != 5:
                raise ValueError(f"Unexpected protocol line in {protocol_path}: {line.strip()}")
            trials.append(CMTrial(utt_id=tokens[1], attack_id=tokens[3], label=tokens[4]))
    return trials


def load_asvspoof_2021_eval_list(protocol_path):
    protocol_path = Path(protocol_path)
    with open(protocol_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_csv_audio_manifest(csv_path, id_column="utterance_id"):
    csv_path = Path(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or id_column not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} must contain the column '{id_column}'."
            )
        rows = []
        for row in reader:
            utt_id = (row.get(id_column) or "").strip()
            if not utt_id:
                continue
            normalized = {key: (value if value is not None else "") for key, value in row.items()}
            normalized[id_column] = utt_id
            rows.append(normalized)
    return rows


def discover_ljspeech_audio(dataset_root):
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "wavs"
    metadata_path = dataset_root / "metadata.csv"

    if not wav_root.exists():
        raise FileNotFoundError(f"LJSpeech wav directory was not found: {wav_root}")

    if metadata_path.exists():
        audio_paths = []
        with open(metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                utt_id = line.split("|", 1)[0].strip()
                if not utt_id:
                    continue
                audio_path = wav_root / f"{utt_id}.wav"
                if audio_path.exists():
                    audio_paths.append(audio_path)
        if audio_paths:
            return audio_paths

    return sorted(
        path
        for path in wav_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def discover_audio_files(input_dir):
    input_dir = Path(input_dir)
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def maybe_resample(audio, src_sr, target_sr):
    if src_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / float(src_sr)
    target_length = max(int(round(duration * target_sr)), 1)
    src_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    dst_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def locate_ffmpeg():
    candidates = []
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    candidates.extend(
        [
            shutil.which("ffmpeg"),
            Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
            Path(sys.prefix) / "bin" / "ffmpeg",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def read_audio_with_ffmpeg(audio_path):
    ffmpeg_path = locate_ffmpeg()
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg was not found in PATH or the current environment.")

    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("ffmpeg returned an empty audio buffer.")
    return audio


def read_audio(audio_path):
    try:
        audio, sample_rate = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        return maybe_resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    except Exception as soundfile_error:
        try:
            return read_audio_with_ffmpeg(audio_path)
        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"Failed to decode {audio_path} with soundfile ({soundfile_error}) "
                f"and ffmpeg ({ffmpeg_error})."
            ) from ffmpeg_error


def pad_audio(audio, max_len=MAX_AUDIO_LENGTH):
    if audio.size == 0:
        raise ValueError("Encountered an empty audio file.")
    if audio.shape[0] >= max_len:
        return audio[:max_len]
    repeat_count = int(np.ceil(max_len / float(audio.shape[0])))
    return np.tile(audio, repeat_count)[:max_len]


def load_audio_tensor(audio_path):
    audio = read_audio(audio_path)
    padded = pad_audio(audio)
    return torch.from_numpy(np.array(padded, dtype=np.float32, copy=True))


class ASVspoof2019TrainDataset(Dataset):
    def __init__(self, trials, dataset_root, supplemental_bonafide_paths=None):
        self.trials = trials
        self.dataset_root = Path(dataset_root)
        self.supplemental_bonafide_paths = [
            Path(path) for path in (supplemental_bonafide_paths or [])
        ]

    def __len__(self):
        return len(self.trials) + len(self.supplemental_bonafide_paths)

    def __getitem__(self, index):
        if index < len(self.trials):
            trial = self.trials[index]
            audio_path = self.dataset_root / "flac" / f"{trial.utt_id}.flac"
            return load_audio_tensor(audio_path), trial.label_index

        audio_path = self.supplemental_bonafide_paths[index - len(self.trials)]
        return load_audio_tensor(audio_path), 1


class ASVspoof2019EvalDataset(Dataset):
    def __init__(self, trials, dataset_root):
        self.trials = trials
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]
        audio_path = self.dataset_root / "flac" / f"{trial.utt_id}.flac"
        meta = {
            "utt_id": trial.utt_id,
            "attack_id": trial.attack_id,
            "label": trial.label,
        }
        return load_audio_tensor(audio_path), meta


class ASVspoof2021EvalDataset(Dataset):
    def __init__(self, utt_ids, dataset_root):
        self.utt_ids = utt_ids
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        audio_path = self.dataset_root / "flac" / f"{utt_id}.flac"
        return load_audio_tensor(audio_path), utt_id


class CSVAudioDataset(Dataset):
    def __init__(self, rows, audio_root, id_column="utterance_id"):
        self.rows = rows
        self.audio_root = Path(audio_root)
        self.id_column = id_column

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        utt_id = row[self.id_column]
        audio_path = self.audio_root / f"{utt_id}.flac"
        row["resolved_audio_path"] = str(audio_path)
        return load_audio_tensor(audio_path), row


class AudioFolderDataset(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = [Path(path) for path in audio_paths]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        return load_audio_tensor(audio_path), str(audio_path)
