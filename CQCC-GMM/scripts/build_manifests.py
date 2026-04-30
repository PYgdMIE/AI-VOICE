from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manifests for ASVspoof/LJSpeech setup.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_2019 = build_asvspoof2019_train(data_root)
    ljspeech = build_ljspeech(data_root)
    eval_2021 = build_asvspoof2021_eval(data_root)

    write_manifest(output_dir / "asv2019_la_train.csv", train_2019)
    write_manifest(output_dir / "ljspeech_bonafide.csv", ljspeech)
    write_manifest(output_dir / "train_2019_la_plus_ljspeech.csv", train_2019 + ljspeech)
    write_manifest(output_dir / "asv2021_la_eval.csv", eval_2021)

    print_summary("asv2019_la_train.csv", train_2019)
    print_summary("ljspeech_bonafide.csv", ljspeech)
    print_summary("train_2019_la_plus_ljspeech.csv", train_2019 + ljspeech)
    print_summary("asv2021_la_eval.csv", eval_2021)
    return 0


def build_asvspoof2019_train(data_root: Path) -> list[dict[str, str]]:
    base = data_root / "LA" / "LA"
    protocol = base / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = base / "ASVspoof2019_LA_train" / "flac"
    rows: list[dict[str, str]] = []
    for line in protocol.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        utt_id = parts[1]
        label = parts[-1]
        audio = audio_dir / f"{utt_id}.flac"
        require_file(audio)
        rows.append({"path": str(audio.resolve()), "label": label, "source": "asv2019_la_train", "id": utt_id})
    return rows


def build_ljspeech(data_root: Path) -> list[dict[str, str]]:
    base = data_root / "LJSpeech-1.1"
    metadata = base / "metadata.csv"
    audio_dir = base / "wavs"
    rows: list[dict[str, str]] = []
    with metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            utt_id = line.split("|", 1)[0]
            audio = audio_dir / f"{utt_id}.wav"
            require_file(audio)
            rows.append({"path": str(audio.resolve()), "label": "bonafide", "source": "ljspeech", "id": utt_id})
    return rows


def build_asvspoof2021_eval(data_root: Path) -> list[dict[str, str]]:
    key = data_root / "LA-keys-stage-1" / "keys" / "CM" / "trial_metadata.txt"
    audio_dir = data_root / "ASVspoof2021_LA_eval" / "flac"
    rows: list[dict[str, str]] = []
    for line in key.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        utt_id = parts[1]
        label = parts[5]
        audio = audio_dir / f"{utt_id}.flac"
        require_file(audio)
        rows.append({"path": str(audio.resolve()), "label": label, "source": "asv2021_la_eval", "id": utt_id})
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["path", "label", "source", "id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def print_summary(name: str, rows: list[dict[str, str]]) -> None:
    bonafide = sum(1 for row in rows if row["label"] == "bonafide")
    spoof = sum(1 for row in rows if row["label"] == "spoof")
    print(f"{name}: total={len(rows)} bonafide={bonafide} spoof={spoof}")


if __name__ == "__main__":
    raise SystemExit(main())
