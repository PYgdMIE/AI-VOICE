# Evaluating AASIST on ASVspoof 2021

This guide explains how to evaluate your trained AASIST model on the ASVspoof 2021 dataset using the labeled CSV protocol file.

## Two Approaches

### Approach 1: Direct Evaluation (Recommended)

Use `evaluate_2021.py` to directly evaluate your model on ASVspoof 2021 data.

**Requirements:**
- Trained AASIST model (`.pth` file)
- ASVspoof 2021 audio files
- Protocol CSV file with labels

**Usage:**

```bash
python evaluate_2021.py \
    --config ./config/AASIST.conf \
    --model_path ./exp_result/LA_AASIST_ep100_bs24/weights/best.pth \
    --protocol_csv /Users/tdao/Downloads/asvspoof2021_eval_whisper_merged_with_labels.csv \
    --audio_dir /path/to/ASVspoof2021_LA_eval/flac \
    --output_score ./scores_2021.txt \
    --subset eval \
    --batch_size 24
```

**Parameters:**
- `--config`: Path to your model config file
- `--model_path`: Path to trained model weights (e.g., `best.pth`)
- `--protocol_csv`: Path to the CSV file with labels
- `--audio_dir`: Directory containing audio files (should contain `.flac` files)
- `--output_score`: Where to save the output scores
- `--subset`: Which subset to evaluate (`eval`, `progress`, or `all`)
- `--batch_size`: Batch size for inference (reduce if out of memory)
- `--num_workers`: Number of dataloader workers (default: 4)

**Output files:**
- `scores_2021.txt`: Utterance-level scores (format: `utterance_id\tlabel\tscore`)
- `scores_2021_metrics.txt`: Evaluation metrics (EER, threshold, etc.)

**Expected audio file locations:**
The script will look for audio files in these locations (in order):
1. `{audio_dir}/{utterance_id}.flac`
2. `{audio_dir}/{utterance_id}.wav`
3. `{audio_dir}/flac/{utterance_id}.flac`
4. `{audio_dir}/flac/{utterance_id}.wav`

---

### Approach 2: Match Pre-computed Scores

If you've already generated scores using existing code, use `match_and_evaluate.py` to match them with the labeled subset.

**Usage:**

```bash
python match_and_evaluate.py \
    --score_file ./existing_scores.txt \
    --protocol_csv /Users/tdao/Downloads/asvspoof2021_eval_whisper_merged_with_labels.csv \
    --output_file ./matched_results.txt \
    --subset eval
```

**Parameters:**
- `--score_file`: Pre-computed score file (any format with `utterance_id` and `score`)
- `--protocol_csv`: Path to the CSV file with labels
- `--output_file`: Where to save evaluation results
- `--subset`: Which subset to evaluate (`eval`, `progress`, or `all`)
- `--show_unmatched`: Show utterance IDs that couldn't be matched

**Score file format:**
The script accepts flexible formats. Examples:
```
# Format 1: ASVspoof 2019 format
LA_E_1234567 A07 spoof 0.123456

# Format 2: Simple format
LA_E_1234567 bonafide 0.876543

# Format 3: Tab-separated
LA_E_1234567\tbonafide\t0.876543
```

The script extracts the utterance ID (first column) and score (last column).

**Output files:**
- `matched_results.txt`: Evaluation metrics (EER, threshold, statistics)
- `matched_results_matched_scores.txt`: Matched scores with labels

---

## Dataset Information

**ASVspoof 2021 Protocol CSV Structure:**
```csv
utterance_id,text,label_str,label_num,subset
LA_E_9332881,It would be nice...,spoof,1,eval
LA_E_1234567,Hello world,bonafide,0,eval
```

**Columns:**
- `utterance_id`: Audio file identifier
- `text`: Transcribed text
- `label_str`: Label as string (`bonafide` or `spoof`)
- `label_num`: Numeric label (not used in evaluation)
- `subset`: Dataset subset (`eval` or `progress`)

**Statistics:**
- Total utterances: 74,432
- Bonafide: 7,457
- Spoof: 66,975
- Eval subset: 67,042
- Progress subset: 7,390

---

## Finding ASVspoof 2021 Audio Files

If you don't have the ASVspoof 2021 audio files yet:

1. **Download from official source:**
   - Visit: https://www.asvspoof.org/index2021.html
   - Look for "ASVspoof 2021 LA (Logical Access)" dataset
   - Download and extract the evaluation set

2. **Expected directory structure:**
   ```
   ASVspoof2021_LA_eval/
   ├── flac/
   │   ├── LA_E_1234567.flac
   │   ├── LA_E_2345678.flac
   │   └── ...
   ```

3. **If files are in a different format:**
   - Update the `Dataset_ASVspoof2021_eval` class in `evaluate_2021.py`
   - Modify the file extension checks in `__getitem__` method

---

## Example Workflow

### Full Evaluation Workflow

```bash
# 1. Train your AASIST model on ASVspoof 2019
python main.py --config ./config/AASIST.conf

# 2. Evaluate on ASVspoof 2021
python evaluate_2021.py \
    --config ./config/AASIST.conf \
    --model_path ./exp_result/LA_AASIST_ep100_bs24/weights/best.pth \
    --protocol_csv /Users/tdao/Downloads/asvspoof2021_eval_whisper_merged_with_labels.csv \
    --audio_dir /path/to/ASVspoof2021_LA_eval/flac \
    --output_score ./aasist_2021_scores.txt

# 3. Check results
cat ./aasist_2021_scores_metrics.txt
```

### Quick Score Matching Workflow

```bash
# If you already have scores from another evaluation
python match_and_evaluate.py \
    --score_file ./my_existing_scores.txt \
    --protocol_csv /Users/tdao/Downloads/asvspoof2021_eval_whisper_merged_with_labels.csv \
    --output_file ./my_2021_results.txt \
    --show_unmatched

# Check results
cat ./my_2021_results.txt
```

---

## Troubleshooting

### Audio files not found
**Error:** `FileNotFoundError: Audio file not found for LA_E_XXXXXXX`

**Solutions:**
1. Verify the `--audio_dir` path is correct
2. Check if files are in a subdirectory (e.g., `flac/`)
3. Verify file extensions (`.flac` vs `.wav`)
4. Ensure utterance IDs match between CSV and audio files

### Out of memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `--batch_size` (try 12, 8, or even 4)
2. Reduce `--num_workers` (try 2 or 0)
3. Use CPU if necessary (slower but works)

### No matched utterances
**Error:** `No matched utterances found!`

**Solutions:**
1. Check if utterance ID format matches (e.g., `LA_E_1234567` vs `LA_E_1234567.flac`)
2. Verify the score file format is correct
3. Try `--show_unmatched` to see which IDs failed to match

### Model loading fails
**Error:** `RuntimeError: Error(s) in loading state_dict`

**Solutions:**
1. Verify the config file matches the model you're loading
2. Check if the model path is correct
3. Ensure the model was trained with the same architecture

---

## Expected Results

AASIST models trained on ASVspoof 2019 should achieve reasonable performance on ASVspoof 2021, though there may be some domain mismatch.

**Baseline expectations:**
- ASVspoof 2019 (same domain): EER ~0.83-0.99%
- ASVspoof 2021 (different domain): EER will likely be higher due to:
  - Different recording conditions
  - Different spoofing attacks
  - Dataset distribution shift

Your actual results will depend on:
- How well your model generalized during training
- The specific attacks present in the 2021 dataset
- Whether the 2021 attacks are similar to 2019 training data

---

## Notes

1. **No ASV scores needed:** These scripts calculate EER (Countermeasure-only metric) without requiring ASV scores. The original t-DCF metric requires ASV scores which are typically not available for 2021 data.

2. **Subset selection:** Use `--subset eval` for the official evaluation set, or `--subset all` to evaluate on all labeled data.

3. **Score interpretation:** Higher scores indicate more bonafide-like. The model outputs logits for [spoof, bonafide] classes, and we use the bonafide class score (index 1).

4. **GPU usage:** The scripts automatically use GPU if available. For large datasets, GPU is highly recommended.

5. **Reproducibility:** Use the same `--seed` value for reproducible results.
