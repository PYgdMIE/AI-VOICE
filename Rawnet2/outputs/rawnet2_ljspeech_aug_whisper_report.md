# RawNet2 Evaluation Summary

## Local Evaluation

- Model: RawNet2 rawnet2_ljspeech_aug best.pth
- Evaluated subset: User-selected CSV aligned with ASVspoof2021 LA CM keys
- Rows with truth: 74432
- Accuracy: 0.9236
- Balanced accuracy: 0.8762
- F1-spoof: 0.9566
- EER: 10.9831%

## Reference Baselines

- LCNN baseline from PROJECT_REPORT.html: EER 18.31%, Accuracy 0.8678, AUC 0.8935, rows 66931
  Source: PROJECT_REPORT.html LCNN branch metrics (C:\Users\Administrator\Downloads\PROJECT_REPORT.html)

## Delta vs References

- Against LCNN baseline from PROJECT_REPORT.html: delta EER -7.3269 percentage points, relative improvement +40.02%. Caveat: Sample count differs: local rows=74432, reference rows=66931.

## Notes

- Primary comparison uses the LCNN baseline reported in PROJECT_REPORT.html.
- PROJECT_REPORT.html reports LCNN results on 66931 samples.
- Sample count differs: local rows=74432, reference rows=66931.
