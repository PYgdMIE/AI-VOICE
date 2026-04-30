"""对单个音频文件推理: 输出 human / AI 概率。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.device import pick_device
from src.infer_lcnn import load_checkpoint_bundle, predict_waveform
from src.audio_io import load_audio_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="train.py 保存的 .pt")
    p.add_argument("--wav", type=str, required=True)
    p.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, transform, feature, sr = load_checkpoint_bundle(args.ckpt, device)

    wav, orig_sr = load_audio_file(args.wav)
    ph, ps = predict_waveform(wav, orig_sr, model, transform, feature, sr, device)

    print(f"file: {Path(args.wav).name}")
    print(f"feature={feature}  P(human)={ph:.4f}  P(AI/synthetic)={ps:.4f}")
    print("predict:", "AI/synthetic" if ps >= ph else "human")


if __name__ == "__main__":
    main()
