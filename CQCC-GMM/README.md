# CQCC-GMM

This folder packages the current CQCC-GMM local baseline.

Included:

- `src/cqcc_gmm/`
- `scripts/`
- `manifests/`
- `models/cqcc_gmm_2019la_ljspeech_retrain_300k.joblib`
- `reports/train_2019la_ljspeech_retrain_300k/`
- `reports/eval_2021la_whisper_subset_retrain_300k/`

Current retrain setup:

- train: ASVspoof 2019 LA train + LJSpeech bonafide supplement
- test: ASVspoof 2021 LA whisper subset
- components: 16
- max-iter: 30
- max-frames-per-class: 300000

Current CQCC-GMM result:

- Accuracy: `0.8740`
- EER: `67.09%`
- ROC AUC: `0.2566`

