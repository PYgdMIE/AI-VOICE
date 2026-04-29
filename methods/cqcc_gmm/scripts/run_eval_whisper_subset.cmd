@echo off
cd /d C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda
if not exist reports\eval_2021la_whisper_subset mkdir reports\eval_2021la_whisper_subset
if not exist .cache\tmp mkdir .cache\tmp
if not exist .cache\numba mkdir .cache\numba
set TMP=C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\.cache\tmp
set TEMP=C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\.cache\tmp
set NUMBA_CACHE_DIR=C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\.cache\numba
echo started %date% %time% > reports\eval_2021la_whisper_subset\eval.status.txt
C:\Users\Administrator\anaconda3\envs\cqcc-gmm-ai-audio\python.exe scripts\evaluate_incremental.py --model models\cqcc_gmm_2019la_ljspeech_300k.joblib --manifest manifests\asv2021_la_eval_whisper_subset.csv --output reports\eval_2021la_whisper_subset\predictions.csv --report-dir reports\eval_2021la_whisper_subset --batch-size 1000 > reports\eval_2021la_whisper_subset\eval.out.log 2> reports\eval_2021la_whisper_subset\eval.err.log
echo %ERRORLEVEL% > reports\eval_2021la_whisper_subset\eval.exitcode.txt
echo finished %date% %time% >> reports\eval_2021la_whisper_subset\eval.status.txt
