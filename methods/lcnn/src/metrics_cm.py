"""
Countermeasure metrics: accuracy, confusion matrix, EER (binary bonafide vs spoof),
and tandem Detection Cost Function (t-DCF).

EER: threshold sweep on scores where higher = more spoof-like; find where FPR ≈ FNR.

t-DCF: measures the combined cost of a CM + ASV tandem system.
Formula (ASVspoof 2019/2021 standard):
    a0 = C_miss * P_tar * P_fa_asv
    b0 = C_fa   * P_spoof
    tDCF(tau) = a0 * P_fn_cm(tau) + b0 * P_fa_cm(tau)
    C0 = min(a0, b0)          # trivial CM baseline
    norm_min_tDCF = min_tau[tDCF(tau)] / C0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class CMMetrics:
    n: int
    accuracy: float
    eer: float
    eer_threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    fpr_at_05: float
    fnr_at_05: float


def compute_metrics(
    y_true: Iterable[int],
    y_score_spoof: Iterable[float],
    threshold_acc: float = 0.5,
) -> CMMetrics:
    """
    y_true: 0=bonafide, 1=spoof
    y_score_spoof: P(spoof) or any score where higher => more spoof
    threshold_acc: for reporting accuracy with y_pred = (score >= t)
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    s = np.asarray(list(y_score_spoof), dtype=np.float64)
    m = np.isfinite(s) & np.isin(y_true, (0, 1))
    y_true, s = y_true[m], s[m]
    n = int(y_true.size)
    if n == 0:
        return CMMetrics(0, float("nan"), float("nan"), float("nan"), 0, 0, 0, 0, float("nan"), float("nan"))

    y_pred = (s >= threshold_acc).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = float((tp + tn) / n) if n else float("nan")

    n_neg = int(np.sum(y_true == 0))
    n_pos = int(np.sum(y_true == 1))
    fpr_05 = fp / n_neg if n_neg else float("nan")
    fnr_05 = fn / n_pos if n_pos else float("nan")

    eer, thr_eer = compute_eer(y_true, s)

    return CMMetrics(
        n=n,
        accuracy=acc,
        eer=float(eer),
        eer_threshold=float(thr_eer),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        fpr_at_05=fpr_05,
        fnr_at_05=fnr_05,
    )


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    Equal Error Rate: min over thresholds of max(FPR, FNR) or intersection of FPR/FNR curves.
    Returns (eer_fraction, threshold).

    Standard approach: find threshold where FPR and FNR are closest; EER = their value at that point.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_neg = np.sum(y_true == 0)
    n_pos = np.sum(y_true == 1)
    if n_neg == 0 or n_pos == 0:
        return float("nan"), float("nan")

    thresholds = np.sort(np.unique(np.concatenate([[0.0, 1.0], y_score])))
    min_diff = np.inf
    eer_val = float("nan")
    thr_out = 0.5
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int64)
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fpr = fp / n_neg
        fnr = fn / n_pos
        diff = abs(fpr - fnr)
        if diff < min_diff - 1e-15:
            min_diff = diff
            eer_val = (fpr + fnr) / 2.0
            thr_out = float(t)
    return float(eer_val), thr_out


@dataclass
class TDCFParams:
    """Standard cost/prior parameters for normalized min t-DCF."""

    p_tar: float = 0.9605    # prior probability of target-speaker trial
    p_spoof: float = 0.05    # prior probability of spoofed trial
    c_miss: float = 1.0      # cost of missed spoof (passes to ASV)
    c_fa: float = 10.0       # cost of falsely rejecting bonafide speaker


@dataclass
class TDCFResult:
    min_tdcf: float           # normalized min t-DCF (the main reported value)
    min_tdcf_threshold: float # CM threshold that achieves min t-DCF
    p_fa_asv: float           # ASV false acceptance rate used
    asv_eer: float            # ASV EER (informational)
    a0: float                 # cost coefficient for missed spoofs
    b0: float                 # cost coefficient for false alarms on bonafide
    c0: float                 # normalisation constant = min(a0, b0)
    params: TDCFParams = field(default_factory=TDCFParams)


def parse_asv_scores(score_txt_path: str) -> dict[str, float]:
    """
    Parse ASVTorch_Kaldi/score.txt produced by the ASVspoof 2021 LA eval kit.

    Line format:  - <trial_id> <score>

    Returns mapping {trial_id: score}.
    """
    from pathlib import Path

    text = Path(score_txt_path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        trial_id = parts[1]
        try:
            out[trial_id] = float(parts[2])
        except ValueError:
            continue
    return out


def parse_asv_trial_metadata(
    asv_meta_path: str,
    subsets: frozenset[str] | None = frozenset({"eval"}),
) -> dict[str, int]:
    """
    Parse ASV/trial_metadata.txt from the ASVspoof 2021 LA keys.

    Line format:  - <trial_id> - - - <target|nontarget|spoof> - <subset>

    Returns {trial_id: 1} for target trials and {trial_id: 0} for nontarget.
    Spoof entries are excluded (they are CM-task labels, not ASV speaker-ID labels).
    """
    from pathlib import Path

    text = Path(asv_meta_path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if subsets and parts[-1] not in subsets:
            continue
        trial_id = parts[1]
        label_str = None
        for p in parts:
            if p.lower() in ("target", "nontarget"):
                label_str = p.lower()
                break
        if label_str is None:
            continue  # skip spoof entries
        out[trial_id] = 1 if label_str == "target" else 0
    return out


def compute_asv_eer(
    asv_labels: dict[str, int],
    asv_scores: dict[str, float],
) -> tuple[float, float, float]:
    """
    Compute ASV EER and the false acceptance rate (P_fa_asv) at the EER threshold.

    asv_labels: {trial_id: 1(target) / 0(nontarget)}
    asv_scores: {trial_id: score}  (higher score = more likely same speaker)

    Returns (eer, p_fa_asv_at_eer, threshold).
    P_fa_asv is FPR at the EER operating point, i.e., the rate at which the ASV
    system accepts nontarget (imposter) speakers.
    """
    common = sorted(set(asv_labels) & set(asv_scores))
    if not common:
        return float("nan"), float("nan"), float("nan")
    y = np.array([asv_labels[t] for t in common], dtype=np.int64)
    s = np.array([asv_scores[t] for t in common], dtype=np.float64)

    n_target = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())
    if n_target == 0 or n_nontarget == 0:
        return float("nan"), float("nan"), float("nan")

    order = np.argsort(-s, kind="mergesort")
    y2, s2 = y[order], s[order]
    tp = np.cumsum(y2 == 1).astype(np.float64)
    fp = np.cumsum(y2 == 0).astype(np.float64)
    fnr = 1.0 - tp / n_target
    fpr = fp / n_nontarget

    k = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[k] + fnr[k]) / 2.0)
    p_fa_asv = float(fpr[k])
    threshold = float(s2[k])
    return eer, p_fa_asv, threshold


def compute_min_tdcf(
    y_cm: np.ndarray,
    s_cm: np.ndarray,
    p_fa_asv: float,
    params: TDCFParams | None = None,
) -> TDCFResult:
    """
    Compute normalized min t-DCF for a CM system given a fixed ASV operating point.

    y_cm    : ground-truth labels, 0 = bonafide, 1 = spoof
    s_cm    : CM score (higher value => more spoof-like)
    p_fa_asv: ASV false acceptance rate at its EER operating point
    params  : cost/prior parameters (defaults to ASVspoof 2019 standard values)

    The normalized min t-DCF is defined as:

        a0 = C_miss * P_tar * P_fa_asv
        b0 = C_fa   * P_spoof
        tDCF(tau) = a0 * P_fn_cm(tau) + b0 * P_fa_cm(tau)
        C0 = min(a0, b0)
        norm_min_tDCF = min_tau[tDCF(tau)] / C0

    Where:
        P_fn_cm(tau) = FNR = missed spoof rate (spoof passes through CM to ASV)
        P_fa_cm(tau) = FPR = false alarm rate   (bonafide rejected by CM)
    """
    if params is None:
        params = TDCFParams()

    y_cm = np.asarray(y_cm, dtype=np.int64)
    s_cm = np.asarray(s_cm, dtype=np.float64)
    valid = np.isfinite(s_cm) & np.isin(y_cm, (0, 1))
    y_cm, s_cm = y_cm[valid], s_cm[valid]

    n_spoof = int((y_cm == 1).sum())
    n_bonafide = int((y_cm == 0).sum())
    if n_spoof == 0 or n_bonafide == 0 or not np.isfinite(p_fa_asv):
        nan = float("nan")
        return TDCFResult(nan, nan, p_fa_asv, nan, nan, nan, nan, params)

    a0 = params.c_miss * params.p_tar * p_fa_asv
    b0 = params.c_fa * params.p_spoof
    c0 = min(a0, b0)
    if c0 < 1e-12:
        nan = float("nan")
        return TDCFResult(nan, nan, p_fa_asv, float("nan"), a0, b0, c0, params)

    # Sweep all unique score thresholds
    thresholds = np.sort(np.unique(s_cm))
    best_cost = np.inf
    best_thr = float(thresholds[0])

    for tau in thresholds:
        y_pred = (s_cm >= tau).astype(np.int64)
        fn = int(((y_cm == 1) & (y_pred == 0)).sum())  # missed spoofs
        fp = int(((y_cm == 0) & (y_pred == 1)).sum())  # false alarms on bonafide
        p_fn = fn / n_spoof
        p_fa = fp / n_bonafide
        cost = a0 * p_fn + b0 * p_fa
        if cost < best_cost:
            best_cost = cost
            best_thr = float(tau)

    min_tdcf = best_cost / c0
    return TDCFResult(
        min_tdcf=min_tdcf,
        min_tdcf_threshold=best_thr,
        p_fa_asv=p_fa_asv,
        asv_eer=float("nan"),  # caller fills this in if needed
        a0=a0,
        b0=b0,
        c0=c0,
        params=params,
    )


def parse_la_cm_trial_metadata(
    path: str,
    subsets: frozenset[str] | None = frozenset({"eval"}),
) -> dict[str, int]:
    """
    Parse ASVspoof2021 LA CM trial_metadata.txt (from LA-keys-full).

    Example line:
      LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval

    Returns utterance_id -> 0 bonafide, 1 spoof.
    If subsets is non-empty, keep only lines whose last token is in subsets.
    If subsets is empty, keep all lines.
    """
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        utt = parts[1]
        if subsets is not None and len(subsets) > 0 and parts[-1] not in subsets:
            continue
        key = None
        for p in parts:
            pl = p.lower()
            if pl == "bonafide":
                key = 0
                break
            if pl == "spoof":
                key = 1
                break
        if key is not None:
            out[utt] = key
    return out
