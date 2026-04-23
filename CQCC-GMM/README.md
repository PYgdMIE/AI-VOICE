# CQCC-GMM Baseline

This folder contains a traditional CQCC-GMM baseline for AI voice spoof detection. It is intended as a reproducible reference baseline alongside the LCNN branch in this repository.

The implemented detector uses:

- CQCC-like features from a Constant-Q transform, log power, DCT, CMVN, delta, and delta-delta features
- one Gaussian Mixture Model for `bonafide`
- one Gaussian Mixture Model for `spoof`
- score definition: `score = mean_loglike_bonafide - mean_loglike_spoof`
- default decision rule: `score >= threshold` means `bonafide`, otherwise `spoof`

## Folder Layout

```text
CQCC-GMM/
  environment.yml
  pyproject.toml
  README.md
  src/cqcc_gmm/
    cli.py
    data.py
    features.py
    model.py
    reporting.py
  scripts/
    build_manifests.py
    build_eval_subset_from_ids.py
    evaluate_incremental.py
    make_smoke_data.py
  results/
    evaluation_metrics.csv
    lcnn_vs_cqcc_gmm_comparison.csv
```

Large local artifacts are intentionally not committed:

- ASVspoof and LJSpeech audio datasets
- trained `.joblib` models
- per-utterance `predictions.csv`
- local cache directories

## Environment

Create and activate the conda environment from inside `CQCC-GMM/`:

```powershell
conda env create -f environment.yml
conda activate cqcc-gmm-ai-audio
```

If the environment already exists:

```powershell
conda env update -f environment.yml --prune
conda activate cqcc-gmm-ai-audio
```

The package exposes the command:

```powershell
cqcc-gmm --help
```

## Data Layout

The manifest builder expects this local layout under `CQCC-GMM/data/`:

```text
data/
  LA/
    LA/
      ASVspoof2019_LA_train/
      ASVspoof2019_LA_cm_protocols/
  ASVspoof2021_LA_eval/
    flac/
  LA-keys-stage-1/
    keys/CM/trial_metadata.txt
  LJSpeech-1.1/
    metadata.csv
    wavs/
```

Build manifests:

```powershell
python scripts/build_manifests.py --data-root data --output-dir manifests
```

This creates:

- `manifests/asv2019_la_train.csv`
- `manifests/ljspeech_bonafide.csv`
- `manifests/train_2019_la_plus_ljspeech.csv`
- `manifests/asv2021_la_eval.csv`

To evaluate on a subset matched to another system, such as LCNN/Whisper utterances:

```powershell
python scripts/build_eval_subset_from_ids.py `
  --ids-csv C:\path\to\asvspoof2021_eval_whisper_merged.csv `
  --base-manifest manifests/asv2021_la_eval.csv `
  --output manifests/asv2021_la_eval_whisper_subset.csv
```

The ID CSV must contain an `utterance_id` column.

## Train

Recommended baseline training command:

```powershell
cqcc-gmm train `
  --manifest manifests/train_2019_la_plus_ljspeech.csv `
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib `
  --components 16 `
  --max-iter 100 `
  --max-frames-per-class 300000 `
  --report-dir reports/train_2019la_ljspeech_300k
```

Training outputs:

- `models/cqcc_gmm_2019la_ljspeech_300k.joblib`
- `reports/train_2019la_ljspeech_300k/training_summary.csv`
- `reports/train_2019la_ljspeech_300k/training_summary.png`

## Evaluate

For smaller labeled manifests:

```powershell
cqcc-gmm evaluate `
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib `
  --manifest manifests/asv2021_la_eval_whisper_subset.csv `
  --output reports/eval_2021la_whisper_subset/predictions.csv `
  --report-dir reports/eval_2021la_whisper_subset
```

For long ASVspoof evaluations, use the resumable evaluator. It appends predictions, records undecodable files, and can be rerun until `remaining=0`:

```powershell
$env:TMP = (Resolve-Path .cache/tmp).Path
$env:TEMP = (Resolve-Path .cache/tmp).Path
$env:NUMBA_CACHE_DIR = (Resolve-Path .cache/numba).Path

python scripts/evaluate_incremental.py `
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib `
  --manifest manifests/asv2021_la_eval_whisper_subset.csv `
  --output reports/eval_2021la_whisper_subset/predictions.csv `
  --report-dir reports/eval_2021la_whisper_subset `
  --batch-size 3000
```

The feature loader first tries `soundfile`; if FLAC decoding fails, it falls back to `imageio-ffmpeg`. This fixed the local ASVspoof 2021 LA decode failures observed during evaluation.

## Current Result

The current CQCC-GMM result on the LCNN/Whisper-matched ASVspoof 2021 LA subset is stored in:

- `results/evaluation_metrics.csv`
- `results/lcnn_vs_cqcc_gmm_comparison.csv`

Summary:

| Model | Accuracy | EER | ROC AUC | F1-spoof @ EER |
|---|---:|---:|---:|---:|
| LCNN only | 0.8169 @ EER | 0.1831 | 0.8935 | 0.8892 |
| CQCC-GMM | 0.8698 @ threshold=0 / 0.3180 @ EER | 0.6820 | 0.2453 | 0.4562 |

CQCC-GMM has high default-threshold accuracy because the test set is spoof-heavy. EER, ROC AUC, and F1-spoof @ EER show that it is much weaker than the LCNN baseline on this evaluation.

## Quick Smoke Test

Use synthetic toy data only to verify the installation:

```powershell
python scripts/make_smoke_data.py
cqcc-gmm train --manifest smoke_data/manifest.csv --model models/smoke_cqcc_gmm.joblib --components 2 --max-iter 5 --max-frames-per-class 2000 --report-dir reports/smoke_train
cqcc-gmm evaluate --model models/smoke_cqcc_gmm.joblib --manifest smoke_data/manifest.csv --output reports/smoke_eval/predictions.csv --report-dir reports/smoke_eval
```

## Notes

- This is a traditional baseline, not a neural anti-spoofing model.
- Do not compare by accuracy alone on imbalanced ASVspoof subsets.
- Use EER and ROC AUC as the primary comparison metrics.
- Trained models and full predictions can be regenerated from the commands above.
