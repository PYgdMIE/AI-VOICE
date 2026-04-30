#!/usr/bin/env python3
"""
Evaluate an LCNN checkpoint only on the utterances listed in a subset CSV.

The subset CSV is expected to include an `utterance_id` column.
Labels are read from an LA keys zip that contains:
  LA-keys-stage-1/keys/CM/trial_metadata.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.device import pick_device
from src.infer_lcnn import load_checkpoint_bundle, predict_file
from src.metrics_cm import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LCNN on a 2021 subset defined by a CSV list")
    p.add_argument("--eval-root", type=str, required=True)
    p.add_argument("--subset-csv", type=str, required=True)
    p.add_argument("--keys-zip", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--metrics-out", type=str, required=True)
    p.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def read_subset_ids(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            utt = (row.get("utterance_id") or row.get("utterance") or "").strip()
            if utt:
                out.append(utt)
    if not out:
        raise SystemExit(f"No utterance ids found in {path}")
    return out


def read_cm_labels_from_zip(zip_path: Path) -> dict[str, tuple[int, str]]:
    labels: dict[str, tuple[int, str]] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        member = "LA-keys-stage-1/keys/CM/trial_metadata.txt"
        if member not in names:
            alt = next((n for n in names if n.endswith("/keys/CM/trial_metadata.txt")), None)
            if alt is None:
                raise SystemExit(f"Could not find keys/CM/trial_metadata.txt in {zip_path}")
            member = alt
        text = z.read(member).decode("utf-8", "replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        utt = parts[1]
        subset = parts[-1]
        label = None
        for p in parts:
            q = p.lower()
            if q == "bonafide":
                label = 0
                break
            if q == "spoof":
                label = 1
                break
        if label is not None:
            labels[utt] = (label, subset)
    return labels


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root).resolve()
    subset_csv = Path(args.subset_csv).resolve()
    keys_zip = Path(args.keys_zip).resolve()

    flac_dir = eval_root / "flac"
    if not flac_dir.is_dir():
        raise SystemExit(f"Missing {flac_dir}")

    subset_ids = read_subset_ids(subset_csv)
    labels = read_cm_labels_from_zip(keys_zip)

    device = pick_device(args.device)
    model, transform, feature, sr = load_checkpoint_bundle(args.ckpt, device)
    print(f"device={device} feature={feature} sr={sr} subset_trials={len(subset_ids)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true: list[int] = []
    p_spoof: list[float] = []
    missing_files = 0
    missing_labels = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utterance", "p_human", "p_spoof", "pred_argmax", "gt_label", "subset"])
        for utt in tqdm(subset_ids, desc="subset-eval"):
            flac = flac_dir / f"{utt}.flac"
            if not flac.is_file():
                w.writerow([utt, "", "", "", "missing_file", ""])
                missing_files += 1
                continue
            ph, ps = predict_file(flac, model, transform, feature, sr, device)
            pred = 1 if ps >= ph else 0
            meta = labels.get(utt)
            if meta is None:
                w.writerow([utt, f"{ph:.8f}", f"{ps:.8f}", pred, "", ""])
                missing_labels += 1
                continue
            gt, subset = meta
            w.writerow([utt, f"{ph:.8f}", f"{ps:.8f}", pred, gt, subset])
            y_true.append(gt)
            p_spoof.append(ps)

    met = compute_metrics(y_true, p_spoof, threshold_acc=0.5)
    yt = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(p_spoof, dtype=np.float64)
    pred_eer = (s >= met.eer_threshold).astype(np.int64)
    acc_eer = float(np.mean(pred_eer == yt)) if yt.size else float("nan")

    summary = {
        "subset_rows_requested": len(subset_ids),
        "subset_rows_with_label": met.n,
        "missing_files": missing_files,
        "missing_labels": missing_labels,
        "accuracy_threshold_0.5": met.accuracy,
        "accuracy_at_eer_threshold": acc_eer,
        "eer": met.eer,
        "eer_threshold_on_p_spoof": met.eer_threshold,
        "confusion_at_threshold_0.5": {
            "tp": met.tp,
            "tn": met.tn,
            "fp": met.fp,
            "fn": met.fn,
        },
        "fpr_at_threshold_0.5": met.fpr_at_05,
        "fnr_at_threshold_0.5": met.fnr_at_05,
        "sources": {
            "subset_csv": str(subset_csv),
            "keys_zip": str(keys_zip),
            "eval_root": str(eval_root),
            "ckpt": str(Path(args.ckpt).resolve()),
        },
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {metrics_out}")


if __name__ == "__main__":
    main()
