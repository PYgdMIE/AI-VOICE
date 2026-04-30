$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\mie\Downloads\AIVOICE"
$CondaExe = "C:\Users\mie\anaconda3\Scripts\conda.exe"
$TrainLog = Join-Path $RepoRoot "checkpoints\lcnn_asvspoof2019_ljspeech.log"
$EvalStdout = Join-Path $RepoRoot "checkpoints\lcnn_subset_eval.stdout.log"
$EvalStderr = Join-Path $RepoRoot "checkpoints\lcnn_subset_eval.stderr.log"
$OutCsv = "C:\Users\mie\Downloads\AIVOICE_datasets_baseline\lcnn_scores_subset_74432_ljspeechmix.csv"
$OutJson = "C:\Users\mie\Downloads\AIVOICE_datasets_baseline\lcnn_metrics_subset_74432_ljspeechmix.json"

Write-Host "[watcher] waiting for training completion..."
while ($true) {
    if (Test-Path $TrainLog) {
        $tail = Get-Content $TrainLog -Tail 10 -ErrorAction SilentlyContinue
        if ($tail -match "done\. best val acc=") {
            break
        }
    }
    Start-Sleep -Seconds 20
}

Write-Host "[watcher] training finished, starting subset evaluation..."
& $CondaExe run -n aivoice-lcnn python scripts\test_lcnn_2021_subset_from_csv.py `
  --eval-root C:\Users\mie\Downloads\AIVOICE\ASVspoof2021_LA_eval `
  --subset-csv C:\Users\mie\Downloads\AIVOICE_datasets_baseline\asvspoof2021_eval_whisper_merged.csv `
  --keys-zip C:\Users\mie\Downloads\AIVOICE_datasets_baseline\LA-keys-stage-1.zip `
  --ckpt C:\Users\mie\Downloads\AIVOICE\checkpoints\lcnn_asvspoof2019_ljspeech_best.pt `
  --out $OutCsv `
  --metrics-out $OutJson `
  --device auto 1>> $EvalStdout 2>> $EvalStderr

Write-Host "[watcher] subset evaluation finished."
