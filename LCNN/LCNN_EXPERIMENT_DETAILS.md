# LCNN Experiment Details

This note summarizes the current experimental setup for the acoustic LCNN baseline.

## 1. Training / Validation / Test Splits

### Main LCNN setup used in the current project

The LCNN baseline is trained with the ASVspoof protocol-based setup:

- training set: `ASVspoof 2019 LA train`
- validation set: `ASVspoof 2019 LA dev`
- test set: `ASVspoof 2021 LA eval`

This split is implemented in [train.py](train.py) with the `--asvspoof-official-splits` option. In this mode, the training script does **not** reshuffle `train` and `dev` together. Instead, it keeps the official protocol split:

- `train` is used only for optimization
- `dev` is used only for model selection
- `eval` is never seen during training and is used only for final testing

This is the cleanest setup for reporting generalization to unseen spoof conditions.

### Alternative folder-based mode

The repository also supports a simpler folder-based setup:

- `human/`
- `ai/`

In that mode, the script randomly splits the available data into:

- train: `85%`
- validation: `15%`

using `--val-ratio 0.15` and a fixed random seed.

### Current reported test set size

The currently saved 2021 evaluation file reports:

- labeled evaluation samples: `148,176`

from:

- [asv2021_la_eval_metrics.json](results/asv2021_la_eval_metrics.json)

## 2. Model Performance Measures

The LCNN evaluation currently reports the following measures.

### Accuracy at threshold 0.5

The model outputs `P(human)` and `P(spoof)`.

For the default accuracy report, prediction is defined as:

- predict spoof if `P(spoof) >= 0.5`
- otherwise predict bonafide / human

This gives a straightforward overall correctness rate.

### Confusion matrix

The confusion matrix is reported at the same threshold and includes:

- `TP`: spoof correctly detected as spoof
- `TN`: human correctly detected as human
- `FP`: human incorrectly flagged as spoof
- `FN`: spoof incorrectly classified as human

This helps interpret whether the model is too conservative or too permissive.

### FPR and FNR

- `FPR` = false positive rate
- `FNR` = false negative rate

At the default threshold, they summarize two different error types:

- how often human speech is wrongly rejected
- how often spoofed speech is missed

### EER (Equal Error Rate)

EER is the main spoofing-oriented metric in this repository.

It is computed by sweeping the spoof score threshold and finding the operating point where:

- false positive rate
- false negative rate

are as close as possible.

Lower EER is better.

### Current LCNN performance

From the current saved result:

- accuracy at threshold 0.5: `0.8678`
- accuracy at EER threshold: `0.8201`
- EER: `0.1799`

These numbers come from:

- [asv2021_la_eval_metrics.json](results/asv2021_la_eval_metrics.json)

## 3. Parameter and Hyperparameter Tuning

At the current stage, LCNN tuning is pragmatic rather than exhaustive.

### Acoustic front end

The default setup uses MFCC features with:

- sample rate: `16 kHz`
- max audio duration: `4.0 s`
- `n_mfcc = 40`
- `n_mel_filters = 64`
- `n_fft = 512`
- `hop = 160`

These values were chosen as stable speech-spoofing baseline settings and kept fixed during the main experiments.

### Model / optimization settings

The current default training configuration is:

- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- batch size: `16`
- epochs: `30`
- seed: `42`

The architecture itself is a compact LCNN with:

- max-feature-map style blocks
- batch normalization
- max pooling
- dropout in the classifier head

### How tuning is done in practice

The current tuning strategy is:

1. start from literature-style and engineering-stable defaults
2. train on ASVspoof 2019 LA train
3. monitor validation accuracy on ASVspoof 2019 LA dev
4. save the checkpoint with the best validation accuracy
5. run final evaluation once on ASVspoof 2021 LA eval

So, at the moment:

- model selection is based on **best validation accuracy**
- final performance is reported on the unseen 2021 evaluation set
- we have **not yet performed** a full grid search, Bayesian optimization, or nested cross-validation for LCNN hyperparameters

That means the current model should be described as a strong baseline with sensible hand-tuned settings, rather than a fully hyperparameter-optimized final system.


