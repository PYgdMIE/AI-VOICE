"""
训练 LCNN 风格模型区分真人 vs AI 音频。

数据目录（在 --data 下）:
  human/  真人 wav/flac/mp3
  ai/     AI 合成 wav/flac/mp3

CPU 训练: 创建 conda 环境时去掉 pytorch-cuda，或设置 CUDA_VISIBLE_DEVICES=""

用法示例:
  conda env create -f environment.yml
  conda activate aivoice-lcnn
  python train.py --data ./data --epochs 30 --batch-size 16

  ASVspoof 2019 LA: unpack LA.zip, then:
  python train.py --dataset asvspoof --asvspoof-root "path/to/LA_parent" --epochs 30

  使用官方 train / dev 划分（推荐，勿与 train+dev 混洗再 random_split）:
  python train.py --dataset asvspoof --asvspoof-root "path/to/LA" --asvspoof-official-splits

  GPU (RTX 50 等): 需安装带 CUDA 12.8 的 PyTorch，见 scripts/install_pytorch_gpu.ps1
  验证: python scripts/verify_cuda.py

默认使用 MFCC 特征；可用 --feature mel 切换为对数梅尔谱。
训练默认 --device auto（CUDA 可用则用 GPU）。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import ASVspoofLADataset, SpoofAudioDataset
from src.device import pick_device
from src.model import LCNNSpoofDetector


class TrainLogger:
    """Log to console and a text file so background jobs still leave a trace."""

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
        # (B, T) -> (B, n_mfcc, T')，MFCC 内部已含 log-mel + DCT
        m = feat(wave)
    else:
        m = feat(wave)
        m = torch.log(m + log_eps)
    m = m.unsqueeze(1)
    return m, torch.tensor(labels, dtype=torch.long)


def parse_args():
    p = argparse.ArgumentParser(description="LCNN spoof training")
    p.add_argument(
        "--dataset",
        type=str,
        choices=("folder", "asvspoof"),
        default="folder",
        help="folder: human/ vs ai/；asvspoof: ASVspoof 2019 LA 官方协议",
    )
    p.add_argument("--data", type=str, default="./data", help="folder 模式：含 human/ 与 ai/")
    p.add_argument(
        "--asvspoof-root",
        type=str,
        default="",
        help="解压 LA.zip 后的根目录（其下含 ASVspoof2019_LA_train 与 protocols）",
    )
    p.add_argument(
        "--asvspoof-splits",
        type=str,
        default="train,dev",
        help="逗号分隔: train, dev。默认合并后按 --val-ratio 随机划分；若加 --asvspoof-official-splits 则 train→训练、dev→验证",
    )
    p.add_argument(
        "--asvspoof-official-splits",
        action="store_true",
        help="ASVspoof 模式：严格使用官方 LA_cm train 作训练集、LA_cm dev 作验证集（忽略 --val-ratio）",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save", type=str, default="./checkpoints/best.pt")
    p.add_argument("--logdir", type=str, default="./runs/lcnn")
    p.add_argument(
        "--feature",
        type=str,
        choices=("mfcc", "mel"),
        default="mfcc",
        help="mfcc: MFCC；mel: 对数梅尔谱",
    )
    p.add_argument("--n-mfcc", type=int, default=40, help="MFCC 阶数（仅 feature=mfcc）")
    p.add_argument(
        "--n-mel-filters",
        type=int,
        default=64,
        help="MFCC 前的梅尔滤波器个数（仅 feature=mfcc）",
    )
    p.add_argument("--n-mels", type=int, default=128, help="梅尔带数（仅 feature=mel）")
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop", type=int, default=160)
    p.add_argument("--max-seconds", type=float, default=4.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto: try CUDA, fall back to CPU if incompatible (e.g. very new GPUs)",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="./checkpoints/training.log",
        help="追加写入训练日志（无终端/后台跑时用它看进度）；设为空字符串可关闭",
    )
    p.add_argument(
        "--log-batch-interval",
        type=int,
        default=100,
        help="每 N 个训练 batch 写一行到日志；0 表示不写 batch 进度",
    )
    p.add_argument(
        "--status-file",
        type=str,
        default="./checkpoints/last_status.txt",
        help="单行文本，刷新当前 epoch/batch，便于快速打开查看",
    )
    return p.parse_args()


def _write_status(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    log_path = Path(args.log_file).resolve() if args.log_file else None
    logger = TrainLogger(log_path)
    status_path = Path(args.status_file).resolve() if args.status_file else None

    device = pick_device(args.device)
    logger.log(f"device: {device}  pid={os.getpid()}")
    logger.log(f"cmd: {' '.join(sys.argv)}")

    if args.dataset == "folder":
        ds = SpoofAudioDataset(
            args.data,
            sample_rate=args.sample_rate,
            max_seconds=args.max_seconds,
        )
        if len(ds) < 2:
            raise RuntimeError("至少需要 2 条音频才能划分训练/验证集")
        n_val = max(1, int(len(ds) * args.val_ratio))
        n_train = len(ds) - n_val
        if n_train < 1:
            n_train, n_val = len(ds) - 1, 1
        train_ds, val_ds = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        if not args.asvspoof_root:
            raise SystemExit("--dataset asvspoof 需要 --asvspoof-root 指向解压后的 LA 根目录")
        splits = tuple(s.strip() for s in args.asvspoof_splits.split(",") if s.strip())
        if args.asvspoof_official_splits:
            if "train" not in splits or "dev" not in splits:
                raise SystemExit(
                    "--asvspoof-official-splits 需要 --asvspoof-splits 同时包含 train 与 dev（例如 train,dev）"
                )
            train_ds = ASVspoofLADataset(
                args.asvspoof_root,
                splits=("train",),
                sample_rate=args.sample_rate,
                max_seconds=args.max_seconds,
            )
            val_ds = ASVspoofLADataset(
                args.asvspoof_root,
                splits=("dev",),
                sample_rate=args.sample_rate,
                max_seconds=args.max_seconds,
            )
            n_train, n_val = len(train_ds), len(val_ds)
            if n_train < 1 or n_val < 1:
                raise RuntimeError("官方划分下 train 或 dev 为空，请检查数据路径与协议文件")
        else:
            ds = ASVspoofLADataset(
                args.asvspoof_root,
                splits=splits,
                sample_rate=args.sample_rate,
                max_seconds=args.max_seconds,
            )
            if len(ds) < 2:
                raise RuntimeError("至少需要 2 条音频才能划分训练/验证集")
            n_val = max(1, int(len(ds) * args.val_ratio))
            n_train = len(ds) - n_val
            if n_train < 1:
                n_train, n_val = len(ds) - 1, 1
            train_ds, val_ds = random_split(
                ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
            )

    if args.feature == "mfcc":
        feat = mfcc_transform(
            args.sample_rate,
            args.n_mfcc,
            args.n_mel_filters,
            args.n_fft,
            args.hop,
        )
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
    n_batches = len(train_loader)
    total_n = n_train + n_val
    logger.log(
        f"dataset_size={total_n} train={n_train} val={n_val} "
        f"train_batches_per_epoch={n_batches}"
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    # Infer time frames from one batch for model init (AdaptiveAvgPool makes any T ok)
    model = LCNNSpoofDetector(n_freq_bins=n_freq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    writer = SummaryWriter(args.logdir)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        if status_path:
            _write_status(status_path, f"epoch {epoch}/{args.epochs} starting train...")
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
            if (
                args.log_batch_interval > 0
                and batch_idx % args.log_batch_interval == 0
            ):
                avg = loss_sum / max(total, 1)
                logger.log(
                    f"epoch {epoch}/{args.epochs} batch {batch_idx}/{n_batches} "
                    f"running_loss={avg:.4f} running_acc={correct/max(total,1):.4f}",
                    file_only=True,
                )
                if status_path:
                    _write_status(
                        status_path,
                        f"epoch {epoch}/{args.epochs}  batch {batch_idx}/{n_batches}  "
                        f"running_loss={avg:.4f}",
                    )
        sched.step()
        train_loss = loss_sum / total
        train_acc = correct / total

        model.eval()
        vt, vc, vloss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                vloss += loss.item() * y.size(0)
                vc += (logits.argmax(dim=1) == y).sum().item()
                vt += y.size(0)
        val_loss = vloss / vt
        val_acc = vc / vt

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        line = (
            f"epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        logger.log(line)
        if status_path:
            _write_status(
                status_path,
                f"epoch {epoch}/{args.epochs} done  val_acc={val_acc:.4f}",
            )

        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "feature": args.feature,
                "n_fft": args.n_fft,
                "hop": args.hop,
                "sample_rate": args.sample_rate,
                "val_acc": val_acc,
            }
            if args.feature == "mfcc":
                ckpt_payload["n_mfcc"] = args.n_mfcc
                ckpt_payload["n_mel_filters"] = args.n_mel_filters
            else:
                ckpt_payload["n_mels"] = args.n_mels
            torch.save(ckpt_payload, save_path)
            logger.log(f"  saved best -> {save_path}")

    writer.close()
    logger.log(f"done. best val acc={best_acc:.4f}")
    logger.close()


if __name__ == "__main__":
    main()
