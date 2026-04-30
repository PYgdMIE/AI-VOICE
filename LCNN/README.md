# LCNN Baseline

This folder contains the acoustic LCNN baseline from our AI voice deepfake detection project.

The current code supports:

- LCNN training on folder-based data or official ASVspoof 2019 LA protocols
- single-file inference
- ASVspoof 2021 LA evaluation
- EER / accuracy reporting for countermeasure evaluation

## Folder Layout

- `train.py`
  - train the LCNN model
- `infer.py`
  - simple file-level inference
- `test_asvspoof2021_la_eval.py`
  - evaluate the checkpoint on ASVspoof 2021 LA eval
- `src/`
  - reusable LCNN modules, dataset loading, audio IO, metrics
- `checkpoints/best.pt`
  - current best saved LCNN checkpoint
- `results/asv2021_la_eval_metrics.json`
  - current ASVspoof 2021 evaluation summary
- `environment.yml`
  - CPU conda environment
- `environment_gpu.yml`
  - GPU conda environment
- `scripts/install_pytorch_gpu.ps1`
  - helper for installing CUDA 12.8 PyTorch wheels on Windows

## Training Data Setup

Two training modes are supported.

### 1. Folder mode

Expected layout:

```text
data/
  human/
  ai/
```

Run from the repository root:

```powershell
python lcnn/train.py --data .\data --epochs 30 --batch-size 16
```

Or from inside `lcnn/`:

```powershell
python train.py --data ..\data --epochs 30 --batch-size 16
```

### 2. ASVspoof 2019 LA mode

Expected root after unpacking `LA.zip`:

```text
ASVspoof2019_LA_train/
ASVspoof2019_LA_dev/
ASVspoof2019_LA_cm_protocols/
```

Recommended training command:

```powershell
python lcnn/train.py --dataset asvspoof --asvspoof-root "C:\path\to\LA" --asvspoof-official-splits
```

This uses:

- `train` split for training
- `dev` split for validation

## ASVspoof 2021 Evaluation

```powershell
python lcnn/test_asvspoof2021_la_eval.py ^
  --eval-root "C:\path\to\ASVspoof2021_LA_eval" ^
  --ckpt lcnn\checkpoints\best.pt ^
  --out lcnn\results\asv2021_la_eval_scores.csv ^
  --trial-metadata "C:\path\to\LA-keys-full\keys\LA\CM\trial_metadata.txt" ^
  --metrics-out lcnn\results\asv2021_la_eval_metrics.json
```

## Current Reported LCNN Result

From [results/asv2021_la_eval_metrics.json](results/asv2021_la_eval_metrics.json):

- labeled evaluation samples: `148176`
- accuracy at threshold `0.5`: `0.8678`
- EER: `0.1799`

## Additional Notes

- The model currently uses MFCC features by default.
- Validation checkpoint selection is based on best validation accuracy.
- A short explanation of splits, metrics, and hyperparameter tuning is included in:
  - [LCNN_EXPERIMENT_DETAILS.md](LCNN_EXPERIMENT_DETAILS.md)
