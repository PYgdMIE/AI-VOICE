# AI Voice Deepfake Detection

This repository contains two voice anti-spoofing model baselines.

## Methods

- methods/cqcc_gmm: CQCC feature extraction plus GMM baseline.
- methods/rawnet2: RawNet2 end-to-end neural baseline.

## Repository Layout

- methods/: model source code, configs, scripts, and manifests.
- artifacts/checkpoints/: trained model files.
- artifacts/reports/: training and evaluation reports.
- artifacts/predictions/: scored CSV or prediction outputs.
- data/: local dataset placeholders only; full datasets are not committed.
- docs/: project notes and upload documentation.
