# RawNet2 Local Setup

This workspace is prepared to train RawNet2 on ASVspoof 2019 LA and score:

- ASVspoof 2019 LA development data
- ASVspoof 2021 LA evaluation data
- any local folder of `.wav` / `.flac` files

## 1. Create the environment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

The setup script installs the official PyTorch `2.9.1` CUDA 13.0 wheel, which is a better fit for recent NVIDIA Blackwell GPUs on Windows.
It also installs `imageio-ffmpeg`, which provides an `ffmpeg` binary used as a fallback decoder for problematic `.flac` files.

CPU-only:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -CpuOnly
```

## 2. Extract ASVspoof 2019 LA train/dev

The default zip path matches the current machine.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\extract_asvspoof2019_la.ps1
```

After extraction, `data\LA` should contain:

- `ASVspoof2019_LA_cm_protocols`
- `ASVspoof2019_LA_asv_scores`
- `ASVspoof2019_LA_train`
- `ASVspoof2019_LA_dev`

## 3. Extract LJSpeech for bonafide augmentation

The default zip path matches the current machine.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\extract_ljspeech.ps1
```

After extraction, `data\LJSpeech-1.1` should contain:

- `metadata.csv`
- `wavs\`

## 4. Train

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rawnet2.ps1
```

Artifacts are written to `artifacts\<run-name>`.
Training now resumes by default from `artifacts\<run-name>\train_state.pt` if it exists.
By default, the script also injects `data\LJSpeech-1.1` as extra `bonafide` training audio.
Each epoch also refreshes:

- `artifacts\<run-name>\history.csv`
- `artifacts\<run-name>\training_curves.png`
- `artifacts\<run-name>\tensorboard\`

The terminal now prints live batch progress in the form:
`[Train] epoch 001 batch 0010/0794 loss=... avg_loss=... avg_acc=...`

To print more densely, for example every batch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rawnet2.ps1 -LogInterval 1
```

To force a fresh run with the same run name:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rawnet2.ps1 -NoResume
```

To start a new LJSpeech-augmented run without touching the old checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rawnet2.ps1 -RunName rawnet2_ljspeech_aug -NoResume
```

To train without LJSpeech:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rawnet2.ps1 -NoLJSpeech
```

Open TensorBoard for a run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_tensorboard.ps1 -RunName rawnet2_local
```

## 5. Score ASVspoof 2021 LA eval

The default eval root matches the current machine:
`C:\Users\Administrator\Downloads\ASVspoof2021_LA_eval`

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\score_asvspoof2021.ps1
```

If the output file already exists, the script skips finished utterances and continues by default.

## 6. Score your own audio folder

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\score_audio_dir.ps1 -InputDir C:\path\to\audio
```

If the CSV already exists, the script skips finished files and continues by default.

The CSV contains:

- `bonafide_score`: RawNet2 class-1 score used by ASVspoof scoring
- `spoof_probability`
- `bonafide_probability`
- `prediction`

## 7. Score the utterances listed in `asvspoof2021_eval_whisper_merged.csv`

The default script uses:

- `C:\Users\Administrator\Downloads\asvspoof2021_eval_whisper_merged.csv`
- `C:\Users\Administrator\Downloads\ASVspoof2021_LA_eval\flac`
- `C:\Users\Administrator\Downloads\LA-keys-stage-1.zip`
- `C:\Users\Administrator\Downloads\PROJECT_REPORT.html`

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\score_whisper_csv.ps1
```

If the scored CSV already exists, the script skips finished `utterance_id` rows and continues by default, then recomputes the metrics JSON from the full merged output.

During scoring, the terminal prints live progress in the form:
`[ScoreCSV] batch 0010/1163 rows 00640/74432 pct=  0.86 elapsed=00:02:31 eta=04:52:10`

The output CSV keeps the original input columns and appends:

- `resolved_audio_path`
- `bonafide_score`
- `spoof_probability`
- `bonafide_probability`
- `prediction`
- `true_label`
- `true_split`
- `is_correct`

It also writes a metrics JSON summary with overall and per-split metrics.
Additional outputs:

- `outputs\asvspoof2021_eval_whisper_merged_progress.csv`
- `outputs\asvspoof2021_eval_whisper_merged_progress.png`
- `outputs\asvspoof2021_eval_whisper_merged_comparison.csv`
- `outputs\asvspoof2021_eval_whisper_merged_comparison.png`
- `outputs\asvspoof2021_eval_whisper_merged_report.md`
- `outputs\asvspoof2021_eval_whisper_merged_report.json`

If `PROJECT_REPORT.html` exists, the comparison report uses the LCNN metrics reported in that file as the primary reference baseline.

To monitor whether training has finished and trigger evaluation automatically:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\monitor_training_and_eval.ps1
```

## Useful direct commands

Train:

```powershell
conda run -p .\env\rawnet2 python .\main.py train --data-root .\data\LA --output-root .\artifacts
```

Resume training:

```powershell
conda run -p .\env\rawnet2 python .\main.py train --data-root .\data\LA --output-root .\artifacts --run-name rawnet2_local --resume
```

Generate 2019 dev CM scores:

```powershell
conda run -p .\env\rawnet2 python .\main.py score-dev --data-root .\data\LA --model-path .\artifacts\rawnet2_local\best.pth --output-path .\outputs\dev_scores.txt
```

Score a CSV manifest:

```powershell
conda run -p .\env\rawnet2 python .\main.py score-csv --input-csv C:\Users\Administrator\Downloads\asvspoof2021_eval_whisper_merged.csv --audio-root C:\Users\Administrator\Downloads\ASVspoof2021_LA_eval\flac --model-path .\artifacts\rawnet2_local\best.pth --output-path .\outputs\asvspoof2021_eval_whisper_merged_scored.csv
```

Score a CSV manifest and compute metrics from the official CM keys:

```powershell
conda run -p .\env\rawnet2 python .\main.py score-csv --input-csv C:\Users\Administrator\Downloads\asvspoof2021_eval_whisper_merged.csv --audio-root C:\Users\Administrator\Downloads\ASVspoof2021_LA_eval\flac --model-path .\artifacts\rawnet2_local\best.pth --output-path .\outputs\asvspoof2021_eval_whisper_merged_scored.csv --cm-key-path C:\Users\Administrator\Downloads\LA-keys-stage-1.zip --metrics-output .\outputs\asvspoof2021_eval_whisper_merged_metrics.json
```

Resume CSV scoring:

```powershell
conda run -p .\env\rawnet2 python .\main.py score-csv --input-csv C:\Users\Administrator\Downloads\asvspoof2021_eval_whisper_merged.csv --audio-root C:\Users\Administrator\Downloads\ASVspoof2021_LA_eval\flac --model-path .\artifacts\rawnet2_local\best.pth --output-path .\outputs\asvspoof2021_eval_whisper_merged_scored.csv --cm-key-path C:\Users\Administrator\Downloads\LA-keys-stage-1.zip --metrics-output .\outputs\asvspoof2021_eval_whisper_merged_metrics.json --resume
```

Compute 2019 dev EER and min t-DCF:

```powershell
conda run -p .\env\rawnet2 python .\tDCF_python\evaluate_tDCF_asvspoof19_eval_LA.py dev .\outputs\dev_scores.txt --no-plot
```
