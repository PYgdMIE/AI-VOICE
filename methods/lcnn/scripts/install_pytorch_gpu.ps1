# Install PyTorch with CUDA 12.8 wheels (supports RTX 50-series / sm_120 Blackwell).
# Run inside your conda env, e.g.:
#   conda activate aivoice-lcnn
#   .\scripts\install_pytorch_gpu.ps1
#
# Official selector: https://pytorch.org/get-started/locally/
# Requires a recent NVIDIA driver (see PyTorch CUDA notes).

$ErrorActionPreference = "Stop"
Write-Host "Uninstalling old torch/torchvision/torchaudio (if any)..."
pip uninstall -y torch torchvision torchaudio 2>$null
Write-Host "Installing torch + torchvision + torchaudio (CUDA 12.8)..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "Verifying..."
python "$PSScriptRoot\verify_cuda.py"
