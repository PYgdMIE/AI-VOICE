from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def write_training_report(report_dir: Path, stats: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(report_dir / "training_summary.csv", stats)

    classes = ["bonafide", "spoof"]
    file_counts = [stats.get(f"{label}_files", 0) for label in classes]
    frame_counts = [stats.get(f"{label}_frames_used", 0) for label in classes]
    lower_bounds = [stats.get(f"{label}_lower_bound", 0.0) for label in classes]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(classes, file_counts, color=["#2f7d32", "#b3261e"])
    axes[0].set_title("Training files")
    axes[0].set_ylabel("count")

    axes[1].bar(classes, frame_counts, color=["#2f7d32", "#b3261e"])
    axes[1].set_title("CQCC frames used")
    axes[1].set_ylabel("frames")

    axes[2].bar(classes, lower_bounds, color=["#2f7d32", "#b3261e"])
    axes[2].set_title("GMM lower bound")
    axes[2].set_ylabel("average log likelihood")

    fig.tight_layout()
    fig.savefig(report_dir / "training_summary.png", dpi=160)
    plt.close(fig)


def write_evaluation_report(report_dir: Path, rows: list[dict[str, Any]]) -> dict[str, float]:
    report_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.array([1 if row["label"] == "bonafide" else 0 for row in rows], dtype=int)
    scores = np.array([float(row["score"]) for row in rows], dtype=float)
    y_pred = np.array([1 if row["prediction"] == "bonafide" else 0 for row in rows], dtype=int)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    roc_auc = float(auc(fpr, tpr))
    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    accuracy = float((y_true == y_pred).mean())

    metrics = {
        "accuracy": accuracy,
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold": eer_threshold,
        "roc_auc": roc_auc,
        "bonafide_total": int((y_true == 1).sum()),
        "spoof_total": int((y_true == 0).sum()),
        "true_bonafide_pred_bonafide": int(matrix[0, 0]),
        "true_bonafide_pred_spoof": int(matrix[0, 1]),
        "true_spoof_pred_bonafide": int(matrix[1, 0]),
        "true_spoof_pred_spoof": int(matrix[1, 1]),
    }
    write_dict_csv(report_dir / "evaluation_metrics.csv", metrics)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}, EER={eer * 100.0:.2f}%", color="#1f5eff")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[y_true == 1], bins=80, alpha=0.65, label="bonafide", color="#2f7d32")
    ax.hist(scores[y_true == 0], bins=80, alpha=0.65, label="spoof", color="#b3261e")
    ax.axvline(0.0, color="#222222", linewidth=1.2, linestyle="--", label="threshold=0")
    ax.set_xlabel("CQCC-GMM score")
    ax.set_ylabel("utterances")
    ax.set_title("Score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "score_histogram.png", dpi=160)
    plt.close(fig)

    return metrics


def compute_eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    fnr = 1.0 - tpr
    diff = fpr - fnr
    crossing = np.where(np.diff(np.sign(diff)) != 0)[0]
    if crossing.size:
        idx = int(crossing[0])
        x0, x1 = diff[idx], diff[idx + 1]
        weight = 0.0 if x1 == x0 else -x0 / (x1 - x0)
        eer = fpr[idx] + weight * (fpr[idx + 1] - fpr[idx])
        threshold = thresholds[idx] + weight * (thresholds[idx + 1] - thresholds[idx])
        return float(eer), float(threshold)

    idx = int(np.nanargmin(np.abs(diff)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def write_dict_csv(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key, value in values.items():
            writer.writerow([key, value])
