"""Quick check: CUDA visible, arch, and a small Conv2d on GPU."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No CUDA. Install GPU driver and PyTorch CUDA build (see scripts/install_pytorch_gpu.ps1).")
        return 1
    d = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1).to(d)
    y = conv(torch.zeros(1, 1, 32, 32, device=d))
    print("conv2d test:", y.shape, "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
