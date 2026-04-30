# Upload Report

Generated: 2026-04-30 12:29:53 +12:00

## Operation

Deleted the previous repository working-tree contents and re-uploaded the two requested model packages into the repository root.

## Source Paths

- CQCC-GMM package source: C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\CQCC-GMM
- CQCC-GMM model source: C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\CQCC-GMM\models\cqcc_gmm_2019la_ljspeech_retrain_300k.joblib
- RawNet2 package source: C:\Users\Administrator\Downloads\RawNet2\package\Rawnet2
- RawNet2 extra artifact source: C:\Users\Administrator\Downloads\RawNet2\artifacts\rawnet2_ljspeech_aug

## Uploaded Layout

- CQCC-GMM/ contains 34 files, 40798857 bytes.
- Rawnet2/ contains 27 files, 525945252 bytes.
- Root files: README.md, .gitignore, .gitattributes, UPLOAD_REPORT.md.

## Model Files

| File | Bytes | SHA256 |
|---|---:|---|
| CQCC-GMM\models\cqcc_gmm_2019la_ljspeech_retrain_300k.joblib | 51592 | 9bcc695e2136e85a7e0bf698105ecfb3f64efc713ba77ad6cdafb2b84daa5239 |
| Rawnet2\artifacts\rawnet2_ljspeech_aug\best.pth | 101803473 | 63d5ef9b5b5a9e17cf3ce68338534ccfb58517338639cf0ca5a06fec9b2f4e82 |
| Rawnet2\artifacts\rawnet2_ljspeech_aug\last.pth | 101803473 | 94631b8b94be2663997d69b0e4c52cdd9174d7d79ac51459d38b0a378648357a |
| Rawnet2\artifacts\rawnet2_ljspeech_aug\train_state.pt | 305350415 | b7c3582314e302401f24d4c31ac8ca853e9f3e6cda04f67fd2c2a906bc2693ee |

## CQCC-GMM Summary

- Training bonafide files: 15680
- Training spoof files: 22800
- Frames per class: 300000
- GMM components: 16
- Whisper subset accuracy: 0.8740326741186586
- Whisper subset EER percent: 67.09219858156028
- Evaluation report directory: CQCC-GMM\reports\eval_2021la_whisper_subset_retrain_300k

## RawNet2 Summary

- Run name: rawnet2_ljspeech_aug
- Device: cuda
- Train examples: 38480
- Dev examples: 24844
- LJSpeech examples: 13100
- Best dev accuracy: 99.60553856061826
- Whisper subset rows: 74432
- Whisper subset accuracy: 0.9236484307824592
- Whisper subset balanced accuracy: 0.8762393492254685
- Whisper subset EER percent: 10.983085854982527
- Evaluation outputs directory: Rawnet2\outputs

## Git/LFS Notes

- Git author configured as PYgdMIE <PYgdMIE@users.noreply.github.com>.
- .pth, .pt, .joblib, and .zip files are tracked with Git LFS.
- RawNet2 package originally contained best.pth; last.pth and train_state.pt were copied from the training artifact directory to match the provided artifact list.