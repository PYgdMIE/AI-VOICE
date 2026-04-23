"""Pick torch device; auto mode probes CUDA (handles GPUs too new for current PyTorch)."""

from __future__ import annotations

import torch

_GPU_FALLBACK_HINT = (
    "If you have an RTX 50-series GPU, install PyTorch CUDA 12.8 wheels, e.g. run "
    "scripts/install_pytorch_gpu.ps1 inside conda env aivoice-lcnn, then retry --device auto or cuda."
)


def pick_device(mode: str = "auto") -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1).cuda()
        conv(torch.zeros(1, 1, 16, 16, device="cuda"))
    except RuntimeError as e:
        print("CUDA present but not usable with this PyTorch build; using CPU.")
        print(_GPU_FALLBACK_HINT)
        print(f"(detail: {e})")
        return torch.device("cpu")
    return torch.device("cuda")
