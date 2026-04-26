# CQCC-GMM Experiment Details

This note explains how the CQCC-GMM baseline is set up: which data we use, how we measure it, and what knobs exist.

## 1. Training / Validation / Test Splits

### Datasets used

We use three speech corpora:

- **ASVspoof 2019 LA train** — the standard anti-spoofing training set. Contains both real ("bonafide") and fake ("spoof") clips, already labeled.
- **LJSpeech 1.1** — a single-speaker English audiobook dataset. We add it as **extra real speech** so the model sees more variety of bonafide audio during training.
- **ASVspoof 2021 LA eval** — the standard anti-spoofing test set. We never train on this; it is only used for the final scoring.

There is **no separate validation split**. CQCC-GMM is not trained by gradient descent, so there is no "checkpoint per epoch" to choose from. We just fit two Gaussian Mixture Models with the EM algorithm until they converge, then save them. The only training-time diagnostic is each GMM's average log likelihood at convergence, which gets written to `reports/.../training_summary.csv` and `training_summary.png`.

### How the splits are built

The script [`scripts/build_manifests.py`](scripts/build_manifests.py) reads the official ASVspoof protocol files plus the LJSpeech metadata, and produces four CSV files (each row = one audio file + its label):

- `manifests/asv2019_la_train.csv` — all ASVspoof 2019 LA training clips (bonafide + spoof)
- `manifests/ljspeech_bonafide.csv` — every LJSpeech clip, all labeled `bonafide`
- `manifests/train_2019_la_plus_ljspeech.csv` — the two above stuck together. **This is what we actually train on.**
- `manifests/asv2021_la_eval.csv` — the 2021 evaluation clips, with labels read from `LA-keys-stage-1/keys/CM/trial_metadata.txt`

### A matched eval subset for fair comparison with LCNN

We want the CQCC-GMM number to be directly comparable to the LCNN number. To do that, both models need to be scored on the **same** utterances. The script [`scripts/build_eval_subset_from_ids.py`](scripts/build_eval_subset_from_ids.py) takes:

- `--ids-csv` — a file listing the utterance IDs LCNN was scored on (it must have an `utterance_id` column)
- `--base-manifest` — the full 2021 LA eval manifest

and writes `manifests/asv2021_la_eval_whisper_subset.csv` — the intersection of the two. Both CQCC-GMM and LCNN are evaluated on this subset.

### Current evaluation set size

From [`evaluation_metrics.csv`](evaluation_metrics.csv):

- real (bonafide) utterances: `7,457`
- fake (spoof) utterances: `66,975`
- total: `74,432`

About **90% of the test set is spoof**. This matters a lot when reading the metrics in section 2 — a model that just guesses "spoof" every time would already look ~90% accurate.

## 2. Model Performance Measures

For each utterance, the detector outputs one number called the **score**:

- it computes the average per-frame log-likelihood under the bonafide GMM (`bonafide_loglike`)
- it computes the average per-frame log-likelihood under the spoof GMM (`spoof_loglike`)
- `score = bonafide_loglike − spoof_loglike`

Higher score → looks more like real speech. The decision is then a simple threshold:

- `score >= threshold` → predict `bonafide`
- otherwise → predict `spoof`

The default threshold is `0.0`, which means "whichever GMM gave a higher average log-likelihood wins." All metrics below are computed by [`src/cqcc_gmm/reporting.py`](src/cqcc_gmm/reporting.py).

### Accuracy at threshold = 0

Just the fraction of utterances the model gets right at the default threshold. It is a quick sanity check, but on a 90%-spoof set it can be misleading — see the confusion matrix below.

### Confusion matrix

A 2×2 table of "what the model predicted" vs. "what the label says", at `threshold = 0`:

- true bonafide predicted bonafide
- true bonafide predicted spoof
- true spoof predicted bonafide
- true spoof predicted spoof

This is the table that tells you whether high accuracy actually reflects real skill, or whether the model is just always guessing the majority class.

### EER (Equal Error Rate)

The **main spoofing metric** in this repo (LCNN reports the same one). Imagine sliding the threshold from very low to very high; at some point the false positive rate (real speech wrongly flagged as spoof) equals the false negative rate (spoof wrongly accepted as real). That common value is the EER. **Lower is better.** A perfect detector has EER = 0; random guessing gives EER = 0.5.

### ROC AUC

Area under the ROC curve. It is **threshold-independent** — it just measures how well the score, as an ordering, separates real from fake.

- AUC = 1.0 → perfect ordering
- AUC = 0.5 → no better than random
- AUC < 0.5 → the score order is **inverted** (the model is systematically calling fakes "real" and real "fake")

### Current CQCC-GMM performance

From [`evaluation_metrics.csv`](evaluation_metrics.csv) on the LCNN/Whisper-matched subset:

- accuracy at `threshold = 0`: `0.8698`
- EER: `0.6820` (~68%)
- EER threshold: `-3.345`
- ROC AUC: `0.2453`

Confusion matrix at `threshold = 0`:

|              | predicted bonafide | predicted spoof |
|--------------|-------------------:|----------------:|
| true bonafide |                  1 |           7,456 |
| true spoof    |              2,232 |          64,743 |

So the ~87% accuracy is misleading: it comes almost entirely from correctly flagging spoof clips, while the model only catches **1 out of 7,457** real clips. EER above 50% and AUC below 0.5 together mean the score ordering is essentially backwards on this subset — the bonafide GMM is, on average, giving *higher* likelihood to spoof clips than to real ones. See [`README.md`](README.md) for the side-by-side comparison with LCNN.

## 3. Parameter and Hyperparameter Tuning

CQCC-GMM has two separate sets of knobs: the **front end** (how audio gets turned into features) and the **back end** (how the GMMs are fit). Both live as frozen dataclasses in the source, so the defaults are in code rather than scattered across CLI flags.

### Front end: turning audio into CQCC features

Defined by `CqccConfig` in [`src/cqcc_gmm/features.py`](src/cqcc_gmm/features.py). Each utterance becomes a sequence of 90-dimensional vectors (one per frame), through these steps:

- resample the audio to **16 kHz**
- pre-emphasize with coefficient **0.97** (boosts higher frequencies a little)
- frame the audio with a **10 ms** hop (160 samples at 16 kHz)
- compute the **Constant-Q Transform**: `fmin = 15 Hz`, `9` octaves, `96` bins per octave → 864 CQT bins
- linearly re-sample those CQT bins to **256** evenly-spaced frequencies
- take **log power**, then a **DCT** (type-II, orthonormal) — this is the "cepstral" step
- keep the first **30** cepstral coefficients
- apply **per-utterance CMVN** (subtract the mean, divide by the std deviation, computed over this clip's frames only)
- append **delta** and **delta-delta** features with a width-`9` window → 30 × 3 = **90 dimensions per frame**

These numbers follow the standard CQCC recipe from the ASVspoof literature and are **kept fixed across all our experiments**. They are not CLI flags — only `--sample-rate` is exposed. If you want to change anything else, edit `CqccConfig` directly.

### Back end: the two GMMs

Defined by `GmmConfig` in [`src/cqcc_gmm/model.py`](src/cqcc_gmm/model.py). Some are CLI flags, some are code-only:

| Parameter | Default | What we typically use | CLI flag |
|---|---:|---:|---|
| Gaussian components per class | `16` | `16` | `--components` |
| covariance type | `diag` | `diag` | (code only) |
| EM max iterations | `200` | `100` | `--max-iter` |
| max frames per class | `250,000` | `300,000` | `--max-frames-per-class` |
| random seed | `42` | `42` | (code only) |

A few things worth knowing:

- Before fitting, we run a single `StandardScaler` over the union of bonafide + spoof frames. So both GMMs see the same input normalization.
- The "max frames per class" cap is enforced by **reservoir sampling** over the entire training manifest (see `_extract_many` in `model.py`). This means the GMM fit always uses at most that many frames per class, regardless of whether the corpus has 100 hours or 1,000 hours. It keeps training time and memory predictable.

### How we actually tune in practice

The current strategy is **pragmatic, not exhaustive**:

1. take the standard CQCC defaults from the literature (front end is fixed, no sweeping)
2. fit one bonafide GMM and one spoof GMM with EM until either convergence or `--max-iter`
3. log each GMM's lower bound (mean log-likelihood at convergence) into `training_summary.csv` / `training_summary.png` — purely as a diagnostic
4. score once on the LCNN-matched 2021 LA subset and report EER / ROC AUC / accuracy

So at the moment:

- **no validation set drives model selection** — the saved `.joblib` is just whatever EM produced at the last iteration
- **no grid search** has been run over `--components` or `--max-frames-per-class`
- **no sweep** of the CQCC front-end parameters

The current CQCC-GMM should be read as a **stable reference baseline**, meant to put the LCNN number in context — not as a fully tuned competing system.
