# AI Voice Model Packages

This repository contains three independent voice anti-spoofing model packages.

## Contents

- `CQCC-GMM/` - CQCC feature extraction plus GMM baseline package.
- `Rawnet2/` - RawNet2 end-to-end neural baseline package.
- `LCNN/` - LCNN acoustic baseline package, including ASVspoof2019 + LJSpeech training scripts and ASVspoof2021 subset evaluation reports.
- `UPLOAD_REPORT.md` - upload report for the previously packaged baselines.
- `LCNN_UPLOAD_REPORT.md` - upload report for the LCNN package.

## Large Files

Large model artifacts are tracked with Git LFS through `.gitattributes`.

Tracked patterns:

```text
*.pth
*.pt
*.joblib
*.zip
```

## Repository Layout

```text
AI-VOICE/
  CQCC-GMM/
  Rawnet2/
  LCNN/
    artifacts/
      checkpoints/
      reports/
      predictions/
      logs/
  README.md
  UPLOAD_REPORT.md
  LCNN_UPLOAD_REPORT.md
  .gitattributes
  .gitignore
```
## Current Baseline Results

All baselines are evaluated on the **same** ASVspoof 2021 LA Whisper-merged subset:

- **Eval file:** `asvspoof2021_eval_whisper_merged_with_labels.csv`
- **Size:** 74,432 utterances  
  - 7,457 bonafide
  - 66,975 spoof
  - approximately 90% spoof
- **Composition:** official `eval` set (67,042) + `progress` set (7,390)  
  - `progress` was not used for any tuning
- **Primary metric:** EER, where lower is better  
  - ROC AUC and accuracy are reported as secondary metrics

| Baseline | Training data | EER ↓ | Acc @ 0.5 | Bal Acc @ EER | Source |
|---|---|---:|---:|---:|---|
| **RawNet2** (LJSpeech-augmented) | ASVspoof 2019 LA + LJSpeech | **0.1098** | 0.9236 | 0.8762 | `Rawnet2/outputs/rawnet2_ljspeech_aug_whisper_metrics.json` |
| **LCNN** (LJSpeech-augmented) | ASVspoof 2019 LA + LJSpeech | 0.1719 | 0.8498 | 0.8280 | `LCNN/artifacts/reports/lcnn_metrics_subset_74432_ljspeechmix.json` |
| **LCNN** (original) | ASVspoof 2019 LA only | 0.1845 | 0.8619 | 0.8155 | `LCNN/artifacts/reports/lcnn_metrics_subset_74432_old.json` |
| **CQCC-GMM** (retrain) | ASVspoof 2019 LA + LJSpeech | 0.6709 | 0.8740 | — | `CQCC-GMM/reports/eval_2021la_whisper_subset_retrain_300k/evaluation_metrics.csv` |
| **AASIST** | — | _pending_ | — | — | not yet uploaded |

### Reading the Numbers

- **RawNet2 is currently the strongest single baseline**, with an EER of approximately 11%.
- **LCNN is the second strongest baseline**, with an EER around 17–18%.
- **CQCC-GMM is a weak traditional reference baseline.** Its 87% accuracy is misleading because the evaluation subset is approximately 90% spoof. At the default threshold, it correctly identifies only **1 out of 7,457** bonafide utterances.
- This is why **EER, rather than accuracy, is used as the primary metric** across this repository.
- **LJSpeech bonafide augmentation improves LCNN**, reducing EER from 0.1845 to 0.1719, which is around a 7% relative improvement.

### Confusion Matrices at the Default Threshold

The neural baselines use a default threshold of 0.5, while CQCC-GMM uses a threshold of 0.

| Baseline | TP: true spoof → spoof | TN: true bonafide → bonafide | FP | FN |
|---|---:|---:|---:|---:|
| RawNet2 | 62,657 | 6,092 | 4,318 | 1,365 |
| LCNN (LJSpeech-augmented) | 57,435 | 5,820 | 1,637 | 9,540 |
| LCNN (original) | 58,963 | 5,193 | 2,264 | 8,012 |
| CQCC-GMM | 65,055 | 1 | 1,920 | 7,456 |

