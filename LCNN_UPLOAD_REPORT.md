# LCNN Upload Report

## Summary

LCNN was added as the third independent model package in this repository. The upload preserves the existing `CQCC-GMM/` and `Rawnet2/` packages and appends a new root-level `LCNN/` folder.

## Source Paths

- LCNN base package: `C:\Users\mie\Downloads\AI-VOICE-main\AI-VOICE-main\lcnn`
- Latest LCNN training scripts and logs: `C:\Users\mie\Downloads\AIVOICE`
- 2021 subset metrics and prediction outputs: `C:\Users\mie\Downloads\AIVOICE_datasets_baseline`

## Uploaded Layout

```text
LCNN/
  README.md
  LCNN_EXPERIMENT_DETAILS.md
  environment.yml
  environment_gpu.yml
  infer.py
  train.py
  test_asvspoof2021_la_eval.py
  scripts/
  src/
  artifacts/
    checkpoints/
    reports/
    predictions/
    logs/
```

## Package Size

- Files uploaded under `LCNN/`: 35
- Total size under `LCNN/`: 13,159,896 bytes

## Checkpoint SHA256

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| `LCNN/artifacts/checkpoints/best.pt` | 546,089 | `14AB3D3B3E11C451D1E73FCCB659508C1C9C35CF3E95B18C10E1DE71961144D1` |
| `LCNN/artifacts/checkpoints/lcnn_asvspoof2019_ljspeech_best.pt` | 1,632,943 | `A45775CB564C003DACDD18ED645EE588FBB54741E399188A1F6E66F67730610F` |
| `LCNN/artifacts/checkpoints/lcnn_asvspoof2019_ljspeech_last.pt` | 1,632,943 | `39CF757B57D9E85AEB57351483CE6CAE65D9B5BE05FACBBD41D29DCAA698030F` |

## Evaluation Summary

The uploaded LCNN reports include the original LCNN baseline and the ASVspoof2019 + LJSpeech human-supplemented LCNN run on the same 74,432 ASVspoof2021 subset.

| Model variant | Accuracy @ 0.5 | EER | Notes |
| --- | ---: | ---: | --- |
| Original LCNN subset baseline | 0.8619 | 0.1845 | Recomputed on the 74,432-row subset. |
| ASVspoof2019 + LJSpeech LCNN | 0.8498 | 0.1719 | New supplementary-human training run. |

For the ASVspoof2019 + LJSpeech run:

- Requested subset rows: 74,432
- Rows with labels: 74,432
- Missing audio files: 0
- Missing labels: 0
- EER threshold on `P(spoof)`: 0.9225
- Confusion at threshold 0.5: TP=57,435, TN=5,820, FP=1,637, FN=9,540

## Git LFS

Git LFS is enabled for large artifacts through `.gitattributes`.

```text
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
```

## Commit

- Commit hash: `40e0bf799cf3f2e2c0154b67a28e100abe0fb3a1`
