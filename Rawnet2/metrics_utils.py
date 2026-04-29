import json
import zipfile
from pathlib import Path

import numpy as np

from tDCF_python.eval_metrics import compute_eer


def load_cm_trial_metadata(source_path):
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as zf:
            member = _find_cm_metadata_member(zf.namelist())
            text = zf.read(member).decode("utf-8")
    else:
        text = source_path.read_text(encoding="utf-8")

    metadata = {}
    for line in text.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        utt_id = tokens[1]
        metadata[utt_id] = {
            "true_label": tokens[5],
            "true_split": tokens[7],
        }
    return metadata


def _find_cm_metadata_member(names):
    candidates = [
        name
        for name in names
        if name.endswith("keys/CM/trial_metadata.txt")
        or name.endswith("/CM/trial_metadata.txt")
        or name.endswith("CM/trial_metadata.txt")
    ]
    if not candidates:
        raise FileNotFoundError("Could not find CM trial_metadata.txt in zip archive.")
    return candidates[0]


def attach_truth_to_row(row, truth_index, id_column="utterance_id"):
    utt_id = row[id_column]
    truth = truth_index.get(utt_id)
    if truth is None:
        row["true_label"] = ""
        row["true_split"] = ""
        row["is_correct"] = ""
        return row

    row["true_label"] = truth["true_label"]
    row["true_split"] = truth["true_split"]
    row["is_correct"] = str(row["prediction"] == row["true_label"]).lower()
    return row


def compute_metrics_from_rows(rows):
    filtered = [row for row in rows if row.get("true_label") in {"bonafide", "spoof"}]
    result = {
        "num_rows": len(rows),
        "num_rows_with_truth": len(filtered),
        "overall": _compute_split_metrics(filtered),
        "by_split": {},
    }

    splits = sorted({row["true_split"] for row in filtered if row.get("true_split")})
    for split_name in splits:
        split_rows = [row for row in filtered if row.get("true_split") == split_name]
        result["by_split"][split_name] = _compute_split_metrics(split_rows)
    return result


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def _compute_split_metrics(rows):
    if not rows:
        return {}

    true_binary = np.array([1 if row["true_label"] == "bonafide" else 0 for row in rows], dtype=int)
    pred_binary = np.array([1 if row["prediction"] == "bonafide" else 0 for row in rows], dtype=int)
    bonafide_scores = np.array([float(row["bonafide_score"]) for row in rows], dtype=float)

    tp = int(np.sum((true_binary == 1) & (pred_binary == 1)))
    tn = int(np.sum((true_binary == 0) & (pred_binary == 0)))
    fp = int(np.sum((true_binary == 0) & (pred_binary == 1)))
    fn = int(np.sum((true_binary == 1) & (pred_binary == 0)))

    metrics = {
        "num_rows": int(len(rows)),
        "label_counts": {
            "bonafide": int(np.sum(true_binary == 1)),
            "spoof": int(np.sum(true_binary == 0)),
        },
        "prediction_counts": {
            "bonafide": int(np.sum(pred_binary == 1)),
            "spoof": int(np.sum(pred_binary == 0)),
        },
        "confusion_matrix": {
            "true_bonafide_pred_bonafide": tp,
            "true_bonafide_pred_spoof": fn,
            "true_spoof_pred_bonafide": fp,
            "true_spoof_pred_spoof": tn,
        },
        "accuracy": _safe_div(tp + tn, len(rows)),
        "balanced_accuracy": (
            _safe_div(tp, tp + fn) + _safe_div(tn, tn + fp)
        ) / 2.0,
        "bonafide_metrics": _class_metrics(tp, fp, fn),
        "spoof_metrics": _class_metrics(tn, fn, fp),
        "mean_bonafide_score": {
            "all": float(np.mean(bonafide_scores)),
            "true_bonafide": float(np.mean(bonafide_scores[true_binary == 1]))
            if np.any(true_binary == 1)
            else None,
            "true_spoof": float(np.mean(bonafide_scores[true_binary == 0]))
            if np.any(true_binary == 0)
            else None,
        },
    }

    bona_scores = bonafide_scores[true_binary == 1]
    spoof_scores = bonafide_scores[true_binary == 0]
    if bona_scores.size > 0 and spoof_scores.size > 0:
        eer, threshold = compute_eer(bona_scores, spoof_scores)
        metrics["eer"] = float(eer)
        metrics["eer_percent"] = float(eer * 100.0)
        metrics["eer_threshold"] = float(threshold)
    else:
        metrics["eer"] = None
        metrics["eer_percent"] = None
        metrics["eer_threshold"] = None

    return metrics


def _class_metrics(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_div(a, b):
    return float(a / b) if b else 0.0
