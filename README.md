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
