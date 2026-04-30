# CQCC-GMM 正式训练与评估命令

数据已按你的分配解压并整理：

- 2019 LA 训练：`data/LA/LA/ASVspoof2019_LA_train`
- 2019 LA 协议：`data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt`
- 2021 LA 评估音频：`data/ASVspoof2021_LA_eval`
- 2021 LA 标签：`data/LA-keys-stage-1/keys/CM/trial_metadata.txt`
- LJSpeech 人声支撑：`data/LJSpeech-1.1`

已生成 manifest：

- `manifests/asv2019_la_train.csv`
- `manifests/ljspeech_bonafide.csv`
- `manifests/train_2019_la_plus_ljspeech.csv`
- `manifests/asv2021_la_eval.csv`

## 1. 激活环境

```powershell
conda activate cqcc-gmm-ai-audio
```

## 2. 重新生成 manifest

只有在你移动或替换数据后才需要重新执行：

```powershell
python scripts/build_manifests.py --data-root data --output-dir manifests
```

## 3. 训练模型（2019 LA + LJSpeech）

推荐先用每类 300,000 帧训练一个可控规模版本：

```powershell
cqcc-gmm train `
  --manifest manifests/train_2019_la_plus_ljspeech.csv `
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib `
  --components 16 `
  --max-iter 100 `
  --max-frames-per-class 300000 `
  --report-dir reports/train_2019la_ljspeech_300k
```

训练过程会显示：

- CQCC 特征提取进度条
- GMM EM 迭代日志
- 训练可视化：`reports/train_2019la_ljspeech_300k/training_summary.png`
- 训练统计：`reports/train_2019la_ljspeech_300k/training_summary.csv`

如果机器内存充足、希望使用更多帧：

```powershell
cqcc-gmm train `
  --manifest manifests/train_2019_la_plus_ljspeech.csv `
  --model models/cqcc_gmm_2019la_ljspeech_800k.joblib `
  --components 32 `
  --max-iter 150 `
  --max-frames-per-class 800000 `
  --report-dir reports/train_2019la_ljspeech_800k
```

不建议一开始使用 `--max-frames-per-class 0` 全量帧，CQCC 帧数会很大，GMM 内存和训练时间都会明显上升。

## 4. 2021 LA 带标签评估

```powershell
cqcc-gmm evaluate `
  --model models/cqcc_gmm_2019la_ljspeech_300k.joblib `
  --manifest manifests/asv2021_la_eval.csv `
  --output reports/eval_2021la_300k/predictions.csv `
  --report-dir reports/eval_2021la_300k
```

评估过程会显示检测进度条，并生成：

- `reports/eval_2021la_300k/predictions.csv`
- `reports/eval_2021la_300k/evaluation_metrics.csv`，包含 `accuracy`、`eer`、`eer_percent`、`eer_threshold` 和 `roc_auc`
- `reports/eval_2021la_300k/roc_curve.png`
- `reports/eval_2021la_300k/score_histogram.png`

## 5. 只使用 2019 LA 训练的对照实验

```powershell
cqcc-gmm train `
  --manifest manifests/asv2019_la_train.csv `
  --model models/cqcc_gmm_2019la_only_300k.joblib `
  --components 16 `
  --max-iter 100 `
  --max-frames-per-class 300000 `
  --report-dir reports/train_2019la_only_300k
```

```powershell
cqcc-gmm evaluate `
  --model models/cqcc_gmm_2019la_only_300k.joblib `
  --manifest manifests/asv2021_la_eval.csv `
  --output reports/eval_2021la_2019only_300k/predictions.csv `
  --report-dir reports/eval_2021la_2019only_300k
```
