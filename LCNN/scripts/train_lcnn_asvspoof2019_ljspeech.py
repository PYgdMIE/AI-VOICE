#!/usr/bin/env python3
"""
Train LCNN using ASVspoof 2019 LA as the main spoofing corpus, with LJSpeech added
as supplementary human speech.

Train split:
  - ASVspoof2019 LA train bonafide + spoof
  - LJSpeech train split

Validation split:
  - ASVspoof2019 LA dev bonafide + spoof
  - LJSpeech validation split

This script adds resumable training via:
  - best checkpoint
  - last checkpoint
  - optimizer / scheduler state restore
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_waveform
from src.device import pick_device
from src.model import LCNNSpoofDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LCNN on ASVspoof2019 + LJSpeech human supplement")
    p.add_argument("--asvspoof-root", type=str, required=True)
    p.add_argument("--ljspeech-root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-seconds", type=float, default=4.0)
    p.add_argument("--feature", type=str, choices=("mfcc", "mel"), default="mfcc")
    p.add_argument("--n-mfcc", type=int, default=40)
    p.add_argument("--n-mel-filters", type=int, default=64)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop", type=int, default=160)
    p.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--lj-val-ratio", type=float, default=0.15)
    p.add_argument("--max-lj", type=int, default=0, help="Optional cap on LJSpeech utterances")
    p.add_argument("--save-best", type=str, default="checkpoints/lcnn_asvspoof2019_ljspeech_best.pt")
    p.add_argument("--save-last", type=str, default="checkpoints/lcnn_asvspoof2019_ljspeech_last.pt")
    p.add_argument("--logdir", type=str, default="runs/lcnn_asvspoof2019_ljspeech")
    p.add_argument("--log-file", type=str, default="checkpoints/lcnn_asvspoof2019_ljspeech.log")
    p.add_argument("--status-file", type=str, default="checkpoints/lcnn_asvspoof2019_ljspeech_status.txt")
    p.add_argument("--resume", type=str, default="", help="Resume from a last-checkpoint path")
    p.add_argument("--resume-auto", action="store_true", help="Resume from --save-last if it exists")
    p.add_argument("--log-batch-interval", type=int, default=100)
    return p.parse_args()


class TrainLogger:
    def __init__(self, log_path: Path | None) -> None:
        self.log_path = log_path
        self._fp = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(self, msg: str, *, file_only: bool = False) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        if not file_only:
            print(line, flush=True)
        if self._fp:
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None


def _write_status(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def mel_transform(sample_rate: int, n_mels: int, n_fft: int, hop: int):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        f_min=20,
        f_max=sample_rate // 2,
    )


def mfcc_transform(
    sample_rate: int,
    n_mfcc: int,
    n_mel_filters: int,
    n_fft: int,
    hop: int,
) -> torchaudio.transforms.MFCC:
    return torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop,
            "n_mels": n_mel_filters,
            "f_min": 20.0,
            "f_max": float(sample_rate // 2),
        },
    )


def collate_fn(
    batch,
    feat: torchaudio.transforms.MelSpectrogram | torchaudio.transforms.MFCC,
    *,
    feature: str,
    log_eps: float = 1e-6,
):
    wavs, labels = zip(*batch)
    lengths = [w.shape[0] for w in wavs]
    max_len = max(lengths)
    padded = []
    for w in wavs:
        if w.shape[0] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        padded.append(w)
    wave = torch.stack(padded, dim=0)
    if feature == "mfcc":
        m = feat(wave)
    else:
        m = feat(wave)
        m = torch.log(m + log_eps)
    m = m.unsqueeze(1)
    return m, torch.tensor(labels, dtype=torch.long)


def parse_la_protocol(protocol_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in protocol_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        utt = parts[1]
        key = parts[-1].lower()
        if key == "bonafide":
            rows.append((utt, 0))
        elif key == "spoof":
            rows.append((utt, 1))
    return rows


def build_asvspoof_items(la_root: Path, split: str) -> list[tuple[Path, int]]:
    proto_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    proto_name = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    }[split]
    protocol = proto_dir / proto_name
    flac_dir = la_root / f"ASVspoof2019_LA_{split}" / "flac"
    rows = []
    for utt, label in parse_la_protocol(protocol):
        flac = flac_dir / f"{utt}.flac"
        if flac.is_file():
            rows.append((flac, label))
    if not rows:
        raise RuntimeError(f"No ASVspoof items found for split={split} under {la_root}")
    return rows


def split_ljspeech_items(ljspeech_root: Path, val_ratio: float, seed: int, max_lj: int) -> tuple[list[Path], list[Path]]:
    wav_dir = ljspeech_root / "wavs"
    items = sorted(wav_dir.glob("*.wav"))
    if max_lj > 0:
        items = items[:max_lj]
    if len(items) < 2:
        raise RuntimeError(f"Too few LJSpeech wavs under {wav_dir}")
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    n_train = max(1, len(items) - n_val)
    train_items = items[:n_train]
    val_items = items[n_train:]
    if not val_items:
        train_items, val_items = items[:-1], items[-1:]
    return train_items, val_items


class MixedAudioDataset(Dataset):
    def __init__(
        self,
        items: Iterable[tuple[Path, int]],
        *,
        sample_rate: int,
        max_seconds: float,
    ) -> None:
        self.items = list(items)
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_seconds)
        if not self.items:
            raise RuntimeError("Empty dataset")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        return load_waveform(path, self.sample_rate, self.max_len), label


@dataclass
class ResumeState:
    epoch_done: int
    best_acc: float


def maybe_resume(
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    resume_path: Path | None,
    logger: TrainLogger,
) -> ResumeState:
    if resume_path is None or not resume_path.is_file():
        return ResumeState(epoch_done=0, best_acc=0.0)
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        sched.load_state_dict(ckpt["scheduler_state"])
    best_acc = float(ckpt.get("best_acc", ckpt.get("val_acc", 0.0)))
    epoch_done = int(ckpt.get("epoch", 0))
    logger.log(f"Resumed from {resume_path} at epoch={epoch_done} best_acc={best_acc:.4f}")
    return ResumeState(epoch_done=epoch_done, best_acc=best_acc)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    log_path = Path(args.log_file).resolve() if args.log_file else None
    status_path = Path(args.status_file).resolve() if args.status_file else None
    logger = TrainLogger(log_path)
    device = pick_device(args.device)
    logger.log(f"device: {device} pid={os.getpid()}")
    logger.log(f"cmd: {' '.join(sys.argv)}")

    la_root = Path(args.asvspoof_root).resolve()
    lj_root = Path(args.ljspeech_root).resolve()

    asv_train = build_asvspoof_items(la_root, "train")
    asv_dev = build_asvspoof_items(la_root, "dev")
    lj_train, lj_val = split_ljspeech_items(lj_root, args.lj_val_ratio, args.seed, args.max_lj)

    train_items = list(asv_train) + [(p, 0) for p in lj_train]
    val_items = list(asv_dev) + [(p, 0) for p in lj_val]

    n_train_human = sum(label == 0 for _, label in train_items)
    n_train_ai = sum(label == 1 for _, label in train_items)
    n_val_human = sum(label == 0 for _, label in val_items)
    n_val_ai = sum(label == 1 for _, label in val_items)

    logger.log(
        f"train_items={len(train_items)} (human={n_train_human}, ai={n_train_ai}) | "
        f"val_items={len(val_items)} (human={n_val_human}, ai={n_val_ai})"
    )

    train_ds = MixedAudioDataset(train_items, sample_rate=args.sample_rate, max_seconds=args.max_seconds)
    val_ds = MixedAudioDataset(val_items, sample_rate=args.sample_rate, max_seconds=args.max_seconds)

    if args.feature == "mfcc":
        feat = mfcc_transform(args.sample_rate, args.n_mfcc, args.n_mel_filters, args.n_fft, args.hop)
        n_freq = args.n_mfcc
    else:
        feat = mel_transform(args.sample_rate, args.n_mels, args.n_fft, args.hop)
        n_freq = args.n_mels

    def _collate(b):
        return collate_fn(b, feat, feature=args.feature)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    model = LCNNSpoofDetector(n_freq_bins=n_freq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    writer = SummaryWriter(args.logdir)

    save_best = Path(args.save_best).resolve()
    save_last = Path(args.save_last).resolve()
    save_best.parent.mkdir(parents=True, exist_ok=True)
    save_last.parent.mkdir(parents=True, exist_ok=True)

    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
    elif args.resume_auto and save_last.is_file():
        resume_path = save_last
    resume_state = maybe_resume(model=model, opt=opt, sched=sched, resume_path=resume_path, logger=logger)
    best_acc = resume_state.best_acc

    for epoch in range(resume_state.epoch_done + 1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        _write_status(status_path, f"epoch {epoch}/{args.epochs} starting train")
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if args.log_batch_interval > 0 and batch_idx % args.log_batch_interval == 0:
                avg = loss_sum / max(total, 1)
                logger.log(
                    f"epoch {epoch}/{args.epochs} batch {batch_idx}/{len(train_loader)} "
                    f"running_loss={avg:.4f} running_acc={correct/max(total,1):.4f}",
                    file_only=True,
                )
                _write_status(
                    status_path,
                    f"epoch {epoch}/{args.epochs} batch {batch_idx}/{len(train_loader)} "
                    f"running_loss={avg:.4f}",
                )
        sched.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        vt = vc = 0
        vloss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                vloss += loss.item() * y.size(0)
                vc += (logits.argmax(dim=1) == y).sum().item()
                vt += y.size(0)
        val_loss = vloss / max(vt, 1)
        val_acc = vc / max(vt, 1)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        line = (
            f"epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        logger.log(line)
        _write_status(status_path, f"epoch {epoch}/{args.epochs} done val_acc={val_acc:.4f}")

        ckpt_payload = {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict(),
            "feature": args.feature,
            "n_fft": args.n_fft,
            "hop": args.hop,
            "sample_rate": args.sample_rate,
            "val_acc": val_acc,
            "train_config": vars(args),
        }
        if args.feature == "mfcc":
            ckpt_payload["n_mfcc"] = args.n_mfcc
            ckpt_payload["n_mel_filters"] = args.n_mel_filters
        else:
            ckpt_payload["n_mels"] = args.n_mels

        torch.save(ckpt_payload, save_last)
        logger.log(f"  saved last -> {save_last}")

        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_payload["best_acc"] = best_acc
            torch.save(ckpt_payload, save_best)
            logger.log(f"  saved best -> {save_best}")

    writer.close()
    logger.log(f"done. best val acc={best_acc:.4f}")
    logger.close()


if __name__ == "__main__":
    main()
