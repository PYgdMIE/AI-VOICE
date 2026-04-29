# AI Voice Deepfake Detection

This repository contains multiple voice anti-spoofing model baselines.

## Methods

- `methods/cqcc_gmm`: CQCC feature extraction plus GMM baseline.
- `methods/rawnet2`: RawNet2 end-to-end neural baseline.
- `methods/lcnn`: LCNN acoustic baseline for human vs spoof classification.

## Repository Layout

- `methods/`: model source code, configs, scripts, and manifests.
- `artifacts/checkpoints/`: trained model files.
- `artifacts/reports/`: training and evaluation reports.
- `artifacts/predictions/`: scored CSV or prediction outputs.
- `artifacts/logs/`: training and evaluation logs.
- `tests/`: method-specific test placeholders or helper assets.
- `data/`: local dataset placeholders only; full datasets are not committed.
- `docs/`: project notes and upload documentation.

## Current Packages

```text
methods/
  cqcc_gmm/
  rawnet2/
  lcnn/

artifacts/
  checkpoints/
    cqcc_gmm/
    rawnet2/
    lcnn/
  reports/
    cqcc_gmm/
    rawnet2/
    lcnn/
  predictions/
    cqcc_gmm/
    rawnet2/
    lcnn/
  logs/
    cqcc_gmm/
    rawnet2/
    lcnn/

tests/
  cqcc_gmm/
  rawnet2/
  lcnn/
```
