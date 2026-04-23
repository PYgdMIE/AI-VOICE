from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    root = Path("smoke_data")
    (root / "bonafide").mkdir(parents=True, exist_ok=True)
    (root / "spoof").mkdir(parents=True, exist_ok=True)

    sample_rate = 16_000
    t = np.linspace(0, 1.2, int(sample_rate * 1.2), endpoint=False)
    rng = np.random.default_rng(42)
    rows: list[list[str]] = []

    for idx, freq in enumerate([220, 260]):
        audio = 0.25 * np.sin(2 * np.pi * freq * t)
        audio += 0.08 * np.sin(2 * np.pi * (freq * 2) * t)
        path = root / "bonafide" / f"real_{idx}.wav"
        sf.write(path, audio, sample_rate)
        rows.append([str(path.relative_to(root)), "bonafide"])

    for idx, freq in enumerate([420, 510]):
        carrier = np.sin(2 * np.pi * freq * t)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 9 * t)
        audio = 0.22 * carrier * mod + 0.03 * rng.normal(size=t.shape)
        path = root / "spoof" / f"fake_{idx}.wav"
        sf.write(path, audio, sample_rate)
        rows.append([str(path.relative_to(root)), "spoof"])

    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    print(root / "manifest.csv")


if __name__ == "__main__":
    main()
