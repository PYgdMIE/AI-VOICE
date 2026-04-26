# CQCC-GMM Baseline

Traditional CQCC-GMM detector for AI voice spoof detection. 
For datasets, splits, parameters, and the analysis of the current numbers, see [`CQCC_GMM_EXPERIMENT_DETAILS.md`](CQCC_GMM_EXPERIMENT_DETAILS.md). 

## How it works

1. **Features.** Each utterance is turned into a sequence of CQCC vectors: Constant-Q transform → log power → DCT → per-utterance CMVN → append delta and delta-delta. Final dimension is `90` per frame.
2. **Models.** One Gaussian Mixture Model is fit on the bonafide frames, another on the spoof frames (16 components each, diagonal covariance).
3. **Score.** For a new utterance, `score = mean_loglike_bonafide - mean_loglike_spoof`. Higher means more bonafide-like.
4. **Decision.** `score >= threshold` → predict `bonafide`, else `spoof`. Default threshold is `0.0`.

## Folder layout

```text
CQCC-GMM/
  environment.yml
  pyproject.toml
  README.md                       # this file
  CQCC_GMM_EXPERIMENT_DETAILS.md  # data splits, parameters, results analysis
  evaluation_metrics.csv          # current eval result
  lcnn_vs_cqcc_gmm_comparison.csv # side-by-side with LCNN
  src/cqcc_gmm/                   # package: cli, features, model, reporting, data
  scripts/                        # manifest builder + incremental evaluator + smoke test
```

The following are intentionally **not** committed and must be produced locally:

- ASVspoof and LJSpeech audio
- trained `.joblib` models
- per-utterance `predictions.csv`
- `.cache/` directories

## 1. Set up the environment

From inside `CQCC-GMM/`:

```bash
conda env create -f environment.yml
conda activate cqcc-gmm-ai-audio
```

If the env already exists, update it instead:

```bash
conda env update -f environment.yml --prune
conda activate cqcc-gmm-ai-audio
```

Verify the CLI is on PATH:

```bash
cqcc-gmm --help
```

## 2. Lay out the data

The manifest builder expects this exact tree under `CQCC-GMM/data/`:

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

These are the upstream files unchanged — no renaming needed.

## 3. Build the manifests

```bash
python scripts/build_manifests.py --data-root data --output-dir manifests
```

Produces four CSVs under `manifests/`:

| File | What it contains |
|---|---|
| `asv2019_la_train.csv` | ASVspoof 2019 LA train, bonafide + spoof |
| `ljspeech_bonafide.csv` | All LJSpeech utterances, labeled bonafide |
| `train_2019_la_plus_ljspeech.csv` | The two above concatenated — **the actual training manifest** |
| `asv2021_la_eval.csv` | ASVspoof 2021 LA eval with labels from the keys file |

(Optional) To evaluate on exactly the same utterances LCNN was scored on, build a matched subset from a CSV that has an `utterance_id` column:

```bash
python scripts/build_eval_subset_from_ids.py \
  --ids-csv /path/to/asvspoof2021_eval_whisper_merged.csv \
  --base-manifest manifests/asv2021_la_eval.csv \
  --output manifests/asv2021_la_eval_whisper_subset.csv
```

## 4. Train

```bash
cqcc-gmm train \
  --manifest manifests/train_2019_la_plus_ljspeech.csv \
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib \
  --components 16 \
  --max-iter 100 \
  --max-frames-per-class 300000 \
  --report-dir reports/train_2019la_ljspeech_300k
```

What the flags do (full table in [`CQCC_GMM_EXPERIMENT_DETAILS.md`](CQCC_GMM_EXPERIMENT_DETAILS.md#gmm-back-end)):

- `--components` — Gaussians per class GMM
- `--max-iter` — EM iteration cap
- `--max-frames-per-class` — reservoir-sample at most this many CQCC frames per class, so training time is bounded regardless of corpus size
- `--report-dir` — where to write `training_summary.csv` and `training_summary.png`

Outputs:

- `models/cqcc_gmm_2019la_ljspeech_300k.joblib` — the trained detector
- `reports/train_2019la_ljspeech_300k/training_summary.{csv,png}` — frame counts, GMM lower bounds

## 5. Evaluate

### Small / quick manifests

```bash
cqcc-gmm evaluate \
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib \
  --manifest manifests/asv2021_la_eval_whisper_subset.csv \
  --output reports/eval_2021la_whisper_subset/predictions.csv \
  --report-dir reports/eval_2021la_whisper_subset
```

Writes `predictions.csv` plus an `evaluation_metrics.csv`, ROC curve, and score histogram into `--report-dir`.

### Long ASVspoof evaluations (resumable)

For the full 2021 LA eval, use the incremental script. It appends predictions, records undecodable files in `errors.csv`, and can be rerun until `remaining=0`:

```bash
# Optional: pin temp + numba caches inside the project (helps on Windows/managed disks)
export TMP="$PWD/.cache/tmp"
export TEMP="$PWD/.cache/tmp"
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
mkdir -p "$TMP" "$NUMBA_CACHE_DIR"

python scripts/evaluate_incremental.py \
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib \
  --manifest manifests/asv2021_la_eval_whisper_subset.csv \
  --output reports/eval_2021la_whisper_subset/predictions.csv \
  --report-dir reports/eval_2021la_whisper_subset \
  --batch-size 3000
```

The feature loader tries `soundfile` first and falls back to `imageio-ffmpeg` if FLAC decoding fails — this fixed the local 2021 LA decode failures observed during evaluation.

### PowerShell users

The `\` line-continuations above are Bash. In PowerShell, replace `\` at end-of-line with a backtick `` ` ``, and replace `export VAR=value` with `$env:VAR = value`. Everything else is identical.

## Current result

On the LCNN/Whisper-matched ASVspoof 2021 LA subset (`manifests/asv2021_la_eval_whisper_subset.csv`):

| Model | Accuracy | EER | ROC AUC | F1-spoof @ EER |
|---|---:|---:|---:|---:|
| LCNN only | 0.8169 @ EER | 0.1831 | 0.8935 | 0.8892 |
| CQCC-GMM | 0.8698 @ threshold=0 / 0.3180 @ EER | 0.6820 | 0.2453 | 0.4562 |

CQCC-GMM has high default-threshold accuracy because the test subset is ~90% spoof. EER, ROC AUC, and F1-spoof @ EER show it is much weaker than LCNN on this evaluation. See [`CQCC_GMM_EXPERIMENT_DETAILS.md`](CQCC_GMM_EXPERIMENT_DETAILS.md#current-cqcc-gmm-performance) for the confusion matrix and an interpretation.

Raw numbers: [`evaluation_metrics.csv`](evaluation_metrics.csv), [`lcnn_vs_cqcc_gmm_comparison.csv`](lcnn_vs_cqcc_gmm_comparison.csv).

## Smoke test (no real data needed)

```bash
python scripts/make_smoke_data.py
cqcc-gmm train \
  --manifest smoke_data/manifest.csv \
  --model models/smoke_cqcc_gmm.joblib \
  --components 2 --max-iter 5 --max-frames-per-class 2000 \
  --report-dir reports/smoke_train
cqcc-gmm evaluate \
  --model models/smoke_cqcc_gmm.joblib \
  --manifest smoke_data/manifest.csv \
  --output reports/smoke_eval/predictions.csv \
  --report-dir reports/smoke_eval
```

## Notes

- This is a traditional baseline, not a neural anti-spoofing model.
- On imbalanced ASVspoof subsets, do not compare by accuracy alone — use EER and ROC AUC.
- All trained models and predictions can be regenerated from the commands above.
