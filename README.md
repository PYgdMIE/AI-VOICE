# AI Voice Deepfake Detection

This repository is organized by method branch.

Current contents:

- [lcnn/](lcnn/)
  - acoustic LCNN baseline
- [CQCC-GMM/](CQCC-GMM/)
  - traditional CQCC feature + Gaussian Mixture Model baseline

The LCNN folder includes:

- training and inference code
- ASVspoof 2021 evaluation script
- saved checkpoint
- current evaluation metrics
- experiment notes covering:
  - training / validation / test splits
  - performance measures
  - parameter and hyperparameter tuning

The CQCC-GMM folder includes:

- CQCC feature extraction code
- bonafide/spoof GMM training and scoring
- ASVspoof/LJSpeech manifest builders
- resumable ASVspoof 2021 LA subset evaluation
- CQCC-GMM metrics and LCNN comparison summary

Start here:

- [lcnn/README.md](lcnn/README.md)
- [lcnn/LCNN_EXPERIMENT_DETAILS.md](lcnn/LCNN_EXPERIMENT_DETAILS.md)
- [CQCC-GMM/README.md](CQCC-GMM/README.md)
