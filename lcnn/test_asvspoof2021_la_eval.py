"""
Run LCNN checkpoint on ASVspoof 2021 LA evaluation set.

Audio: extracted ASVspoof2021_LA_eval/ with flac/ and ASVspoof2021.LA.cm.eval.trl.txt

Labels + Accuracy + EER: download LA-keys-full.tar.gz from asvspoof.org, unpack, then pass:
  --trial-metadata path/to/keys/LA/CM/trial_metadata.txt
or
  --keys-root path/to/keys   (auto-finds .../LA/CM/trial_metadata.txt)

Official eval subset uses metadata lines ending with \"eval\" (default).

Usage:
  python test_asvspoof2021_la_eval.py --eval-root \"...\\ASVspoof2021_LA_eval\" ^
    --ckpt checkpoints/best.pt --out results/scores.csv ^
    --trial-metadata \"...\\keys\\LA\\CM\\trial_metadata.txt\" ^
    --metrics-out results/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.device import pick_device
from src.infer_lcnn import load_checkpoint_bundle, predict_file
from src.metrics_cm import compute_metrics, parse_la_cm_trial_metadata


def read_eval_trial_list(protocol_path: Path) -> list[str]:
    lines = protocol_path.read_text(encoding="utf-8", errors="replace").splitlines()
    utts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        utts.append(parts[0])
    return utts


def read_simple_keys(path: Path) -> dict[str, int]:
    m: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) < 2:
            continue
        utt = parts[0]
        key = parts[-1].lower()
        if key in ("bonafide", "b", "0", "human"):
            m[utt] = 0
        elif key in ("spoof", "s", "1", "attack"):
            m[utt] = 1
        elif key.isdigit():
            m[utt] = int(key)
    return m


def find_trial_metadata(keys_root: Path) -> Path | None:
    for rel in (
        Path("LA") / "CM" / "trial_metadata.txt",
        Path("CM") / "trial_metadata.txt",
        Path("trial_metadata.txt"),
    ):
        p = keys_root / rel
        if p.is_file():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="ASVspoof 2021 LA eval scoring with LCNN ckpt")
    p.add_argument(
        "--eval-root",
        type=str,
        required=True,
        help="Path to ASVspoof2021_LA_eval (contains flac/ and eval .trl.txt)",
    )
    p.add_argument("--ckpt", type=str, default="./checkpoints/best.pt")
    p.add_argument("--out", type=str, default="./results/asv2021_la_eval_scores.csv")
    p.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--limit", type=int, default=0, help="Only first N trials (debug)")
    p.add_argument(
        "--trial-metadata",
        type=str,
        default="",
        help="LA-keys-full: keys/LA/CM/trial_metadata.txt (bonafide/spoof + subsets)",
    )
    p.add_argument(
        "--keys-root",
        type=str,
        default="",
        help="Folder that contains LA/CM/trial_metadata.txt (alternative to --trial-metadata)",
    )
    p.add_argument(
        "--metadata-subsets",
        type=str,
        default="eval",
        help="Comma-separated subset tokens (last column), e.g. eval or eval,progress",
    )
    p.add_argument(
        "--simple-keys-file",
        type=str,
        default="",
        help="Optional: simple two-column utt + bonafide/spoof file (if not using trial_metadata)",
    )
    p.add_argument(
        "--metrics-out",
        type=str,
        default="./results/asv2021_la_eval_metrics.json",
        help="Write accuracy, EER, confusion counts (needs labels)",
    )
    args = p.parse_args()

    root = Path(args.eval_root).resolve()
    trl_names = ("ASVspoof2021.LA.cm.eval.trl.txt", "ASVspoof2021.LA.cm.eval.trl")
    protocol = next((root / n for n in trl_names if (root / n).is_file()), None)
    if protocol is None:
        raise SystemExit(f"No eval protocol in {root}. Expected one of: {trl_names}")
    flac_dir = root / "flac"
    if not flac_dir.is_dir():
        raise SystemExit(f"Missing {flac_dir}")

    utts = read_eval_trial_list(protocol)
    if args.limit > 0:
        utts = utts[: args.limit]

    keys: dict[str, int] = {}
    if args.trial_metadata:
        tm = Path(args.trial_metadata)
        if not tm.is_file():
            raise SystemExit(f"--trial-metadata not found: {tm}")
        subs = frozenset(s.strip() for s in args.metadata_subsets.split(",") if s.strip())
        keys = parse_la_cm_trial_metadata(str(tm), subsets=subs)
        print(f"Loaded {len(keys)} labels from trial_metadata (subsets={sorted(subs)})")
    elif args.keys_root:
        kr = Path(args.keys_root)
        tm = find_trial_metadata(kr)
        if tm is None:
            raise SystemExit(
                f"Could not find LA/CM/trial_metadata.txt under {kr}. "
                "Set --trial-metadata explicitly."
            )
        subs = frozenset(s.strip() for s in args.metadata_subsets.split(",") if s.strip())
        keys = parse_la_cm_trial_metadata(str(tm), subsets=subs)
        print(f"Loaded {len(keys)} labels from {tm} (subsets={sorted(subs)})")
    elif args.simple_keys_file:
        sk = Path(args.simple_keys_file)
        if sk.is_file():
            keys = read_simple_keys(sk)
            print(f"Loaded {len(keys)} labels from simple key file")
        else:
            print(f"Warning: --simple-keys-file not found: {sk}")

    device = pick_device(args.device)
    model, transform, feature, sr = load_checkpoint_bundle(args.ckpt, device)
    print(f"device={device} feature={feature} sr={sr} trials={len(utts)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_list: list[int] = []
    p_spoof_list: list[float] = []

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utterance", "p_human", "p_spoof", "pred_argmax", "gt_label"])
        for utt in tqdm(utts, desc="eval"):
            flac = flac_dir / f"{utt}.flac"
            if not flac.is_file():
                w.writerow([utt, "", "", "", "missing_file"])
                continue
            ph, ps = predict_file(flac, model, transform, feature, sr, device)
            pred = 1 if ps >= ph else 0
            gt = keys.get(utt)
            gt_cell = "" if gt is None else gt
            w.writerow([utt, f"{ph:.8f}", f"{ps:.8f}", pred, gt_cell])
            if gt is not None:
                y_true_list.append(gt)
                p_spoof_list.append(ps)

    print(f"Wrote {out_path}")

    if not y_true_list:
        print(
            "No metrics: provide --trial-metadata (or --keys-root / --simple-keys-file) "
            "from LA-keys-full to compute accuracy and EER."
        )
        return

    met = compute_metrics(y_true_list, p_spoof_list, threshold_acc=0.5)
    yt = np.asarray(y_true_list, dtype=np.int64)
    s = np.asarray(p_spoof_list, dtype=np.float64)
    pred_eer = (s >= met.eer_threshold).astype(np.int64)
    acc_eer = float(np.mean(pred_eer == yt)) if yt.size else float("nan")

    summary = {
        "n_scored_with_label": met.n,
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
        "note": "EER: threshold sweep on P(spoof); min |FPR-FNR|. "
        "Accuracy@0.5: pred = (P_spoof>=0.5).",
    }

    mp = Path(args.metrics_out)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {mp}")
    print(
        f"Accuracy @0.5 (P_spoof): {met.accuracy:.4f}  |  "
        f"Accuracy @EER threshold: {acc_eer:.4f}  |  "
        f"EER: {met.eer:.4f}  (thr={met.eer_threshold:.4f})"
    )
    print(f"Confusion @0.5  TP={met.tp} TN={met.tn} FP={met.fp} FN={met.fn}")


if __name__ == "__main__":
    main()
