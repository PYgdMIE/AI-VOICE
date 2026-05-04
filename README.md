# AI Voice Model Packages

This repository contains three independent voice anti-spoofing model packages.

## Contents
- `AASIST/` - AASIST reproduced neural baseline based on Jung et al. (2021), implemented and evaluated in Python.
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
  AASIST/
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

| Baseline | Training data | EER ↓ | Acc @ 0.5 | Bal Acc @ EER | 
|---|---|---:|---:|---:|
| **RawNet2** (LJSpeech-augmented) | ASVspoof 2019 LA + LJSpeech | **0.1098** | 0.9236 | 0.8762 | 
| **LCNN** (LJSpeech-augmented) | ASVspoof 2019 LA + LJSpeech | 0.1719 | 0.8498 | 0.8280 | 
| **LCNN** (original) | ASVspoof 2019 LA only | 0.1845 | 0.8619 | 0.8155 | 
| **CQCC-GMM** (retrain) | ASVspoof 2019 LA + LJSpeech | 0.6709 | 0.8740 | — | 
| **AASIST** | ASVspoof 2019 LA + LJSpeech | 0.2350| 0.9400 | — |

### Reading the Numbers

- **RawNet2 is currently the strongest single baseline**, with an EER of approximately 11%.

- **LCNN is the second strongest baseline**, with an EER around 17%.

- **AASIST achieves high default-threshold accuracy**, but its EER is higher than RawNet2 and LCNN in our current experiment.

- The AASIST result may be affected by the incomplete dataset download, since 745 bonafide files were missing during evaluation.

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
| AASIST | — | 3,532 / 6,712 | — | — |

## Overall Take-away

Among the current baselines, RawNet2 performs best, followed by LCNN.  

Adding LJSpeech improves LCNN by reducing EER and improving bonafide recognition.  

AASIST shows reasonable performance but may be affected by incomplete evaluation data.  

CQCC-GMM remains useful as a traditional reference baseline, but it performs poorly in terms of EER and bonafide recognition.

Overall, these results show that accuracy alone can be misleading on this imbalanced test set.  

Therefore, EER and confusion matrix analysis are more important for evaluating AI voice deepfake detection models.
