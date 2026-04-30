@echo off
setlocal

cd /d C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda

if not exist .cache\tmp mkdir .cache\tmp
if not exist .cache\numba mkdir .cache\numba
if not exist reports\train_2019la_ljspeech_retrain_300k mkdir reports\train_2019la_ljspeech_retrain_300k
if not exist reports\eval_2021la_whisper_subset_retrain_300k mkdir reports\eval_2021la_whisper_subset_retrain_300k

set TMP=C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\.cache\tmp
set TEMP=C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\.cache\tmp
set NUMBA_CACHE_DIR=C:\Users\Administrator\Documents\Codex\2026-04-21-cqcc-gmm-ai-conda\.cache\numba

echo started %date% %time% > reports\train_2019la_ljspeech_retrain_300k\train.status.txt
C:\Users\Administrator\anaconda3\envs\cqcc-gmm-ai-audio\python.exe -m cqcc_gmm.cli train --manifest manifests\train_2019_la_plus_ljspeech.csv --model models\cqcc_gmm_2019la_ljspeech_retrain_300k.joblib --components 16 --max-iter 30 --max-frames-per-class 300000 --report-dir reports\train_2019la_ljspeech_retrain_300k > reports\train_2019la_ljspeech_retrain_300k\train.out.log 2> reports\train_2019la_ljspeech_retrain_300k\train.err.log
echo %ERRORLEVEL% > reports\train_2019la_ljspeech_retrain_300k\train.exitcode.txt
echo finished %date% %time% >> reports\train_2019la_ljspeech_retrain_300k\train.status.txt
if not exist models\cqcc_gmm_2019la_ljspeech_retrain_300k.joblib exit /b 1

echo started %date% %time% > reports\eval_2021la_whisper_subset_retrain_300k\eval.status.txt
C:\Users\Administrator\anaconda3\envs\cqcc-gmm-ai-audio\python.exe scripts\evaluate_incremental.py --model models\cqcc_gmm_2019la_ljspeech_retrain_300k.joblib --manifest manifests\asv2021_la_eval_whisper_subset.csv --output reports\eval_2021la_whisper_subset_retrain_300k\predictions.csv --report-dir reports\eval_2021la_whisper_subset_retrain_300k --batch-size 3000 > reports\eval_2021la_whisper_subset_retrain_300k\eval.out.log 2> reports\eval_2021la_whisper_subset_retrain_300k\eval.err.log
echo %ERRORLEVEL% > reports\eval_2021la_whisper_subset_retrain_300k\eval.exitcode.txt
echo finished %date% %time% >> reports\eval_2021la_whisper_subset_retrain_300k\eval.status.txt

endlocal
