"""
Evaluation script for ASVspoof 2021 dataset using trained AASIST model.

This script:
1. Reads ASVspoof 2021 protocol CSV
2. Generates utterance-level scores using AASIST model
3. Evaluates EER (and optionally t-DCF if ASV scores available)

Usage:
    python evaluate_2021.py --config CONFIG_FILE --model_path MODEL_PATH \
                            --protocol_csv CSV_FILE --audio_dir AUDIO_DIR \
                            --output_score OUTPUT_SCORE_FILE

Example:
    python evaluate_2021.py --config ./config/AASIST.conf \
                            --model_path ./exp_result/LA_AASIST_ep100_bs24/weights/best.pth \
                            --protocol_csv /Users/tdao/Downloads/asvspoof2021_eval_whisper_merged_with_labels.csv \
                            --audio_dir /path/to/ASVspoof2021_LA_eval/flac \
                            --output_score ./scores_2021.txt
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_utils import pad
from utils import set_seed


class Dataset_ASVspoof2021_eval(Dataset):
    """Dataset class for ASVspoof 2021 evaluation data."""

    def __init__(self, utterance_ids, labels, base_dir, cut=64600):
        """
        Args:
            utterance_ids: List of utterance IDs
            labels: Dict mapping utterance_id to label (1=bonafide, 0=spoof)
            base_dir: Base directory containing audio files
            cut: Number of samples to pad/cut to
        """
        self.utterance_ids = utterance_ids
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.cut = cut

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, index):
        utt_id = self.utterance_ids[index]

        # Try different file extensions and locations
        audio_path = None
        for ext in ['.flac', '.wav']:
            # Try direct path
            path = self.base_dir / f"{utt_id}{ext}"
            if path.exists():
                audio_path = path
                break
            # Try with flac subdirectory
            path = self.base_dir / "flac" / f"{utt_id}{ext}"
            if path.exists():
                audio_path = path
                break

        if audio_path is None:
            # This should not happen if filtering was done properly
            # Return zeros as a fallback
            print(f"\n⚠ Warning: File not found for {utt_id}, returning zeros")
            X_pad = np.zeros(self.cut, dtype=np.float32)
            x_inp = Tensor(X_pad)
            label = self.labels.get(utt_id, -1)
            return x_inp, utt_id, label

        # Load audio with error handling
        try:
            X, _ = sf.read(str(audio_path))
            X_pad = pad(X, self.cut)
            x_inp = Tensor(X_pad)
        except Exception as e:
            # If file is corrupted, return zeros and warn
            print(f"\n⚠ Warning: Error reading {utt_id} ({audio_path}): {e}")
            print(f"   Skipping this file and returning zeros")
            X_pad = np.zeros(self.cut, dtype=np.float32)
            x_inp = Tensor(X_pad)

        label = self.labels.get(utt_id, -1)

        return x_inp, utt_id, label


def load_protocol_csv(csv_path, subset='eval'):
    """
    Load ASVspoof 2021 protocol CSV.

    Args:
        csv_path: Path to CSV file
        subset: Which subset to use ('eval' or 'progress' or 'all')

    Returns:
        utterance_ids: List of utterance IDs
        labels: Dict mapping utterance_id to label (1=bonafide, 0=spoof)
    """
    utterance_ids = []
    labels = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt_id = row['utterance_id']
            label_str = row['label_str']
            row_subset = row['subset']

            # Filter by subset if specified
            if subset != 'all' and row_subset != subset:
                continue

            utterance_ids.append(utt_id)
            # Convert label: bonafide=1, spoof=0
            labels[utt_id] = 1 if label_str == 'bonafide' else 0

    print(f"Loaded {len(utterance_ids)} utterances from CSV")
    print(f"  Bonafide: {sum(1 for l in labels.values() if l == 1)}")
    print(f"  Spoof: {sum(1 for l in labels.values() if l == 0)}")

    return utterance_ids, labels


def filter_existing_files(utterance_ids, labels, audio_dir, validate_audio=True):
    """
    Filter utterance IDs to only include files that actually exist and can be loaded.

    Args:
        utterance_ids: List of utterance IDs
        labels: Dict mapping utterance_id to label
        audio_dir: Directory containing audio files
        validate_audio: If True, also test if files can be loaded (slower but safer)

    Returns:
        filtered_utterance_ids: List of IDs with existing and valid files
        filtered_labels: Dict with only valid files
        missing_ids: List of IDs without files
        corrupted_ids: List of IDs with corrupted files
    """
    audio_dir = Path(audio_dir)
    filtered_utterance_ids = []
    filtered_labels = {}
    missing_ids = []
    corrupted_ids = []

    print("\nChecking which audio files exist and are readable...")
    if validate_audio:
        print("(This may take a while as we validate each file)")

    for utt_id in tqdm(utterance_ids, desc="Validating files"):
        found = False
        audio_path = None

        # Try different file extensions and locations
        for ext in ['.flac', '.wav']:
            for path in [
                audio_dir / f"{utt_id}{ext}",
                audio_dir / "flac" / f"{utt_id}{ext}"
            ]:
                if path.exists():
                    audio_path = path
                    found = True
                    break
            if found:
                break

        if not found:
            missing_ids.append(utt_id)
            continue

        # Validate that file can actually be read
        if validate_audio:
            try:
                # Try to read just the first few samples to check if file is valid
                data, sr = sf.read(str(audio_path), stop=100)
                # File is valid
                filtered_utterance_ids.append(utt_id)
                filtered_labels[utt_id] = labels[utt_id]
            except Exception as e:
                # File exists but can't be read (corrupted)
                corrupted_ids.append(utt_id)
        else:
            # Just check existence without validating
            filtered_utterance_ids.append(utt_id)
            filtered_labels[utt_id] = labels[utt_id]

    print(f"\nFile validation results:")
    print(f"  Valid files: {len(filtered_utterance_ids)}")
    print(f"  Missing files: {len(missing_ids)}")
    if validate_audio:
        print(f"  Corrupted/unreadable files: {len(corrupted_ids)}")
    print(f"  Coverage: {len(filtered_utterance_ids)/len(utterance_ids)*100:.1f}%")

    if filtered_labels:
        bonafide_count = sum(1 for l in filtered_labels.values() if l == 1)
        spoof_count = sum(1 for l in filtered_labels.values() if l == 0)
        print(f"  Valid - Bonafide: {bonafide_count}, Spoof: {spoof_count}")

    return filtered_utterance_ids, filtered_labels, missing_ids, corrupted_ids


def get_model(model_config, device):
    """Load model architecture."""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("Model params: {}".format(nb_params))
    return model


def produce_evaluation_scores(data_loader, model, device, save_path):
    """
    Generate utterance-level scores.

    Output format: utterance_id label score
    """
    model.eval()

    results = []
    skipped_batches = 0
    print("Generating scores...")

    with torch.no_grad():
        for batch_x, utt_ids, batch_labels in tqdm(data_loader):
            try:
                batch_x = batch_x.to(device)

                # Forward pass
                _, batch_out = model(batch_x)

                # Get scores for bonafide class (index 1)
                batch_scores = batch_out[:, 1].data.cpu().numpy()

                # Collect results
                for utt_id, label, score in zip(utt_ids, batch_labels.numpy(), batch_scores):
                    results.append({
                        'utterance_id': utt_id,
                        'label': 'bonafide' if label == 1 else 'spoof',
                        'score': float(score)
                    })
            except Exception as e:
                print(f"\n⚠ Warning: Skipping batch due to error: {e}")
                skipped_batches += 1
                continue

    if skipped_batches > 0:
        print(f"\n⚠ Skipped {skipped_batches} batches due to errors")

    # Save scores
    with open(save_path, 'w') as f:
        f.write("utterance_id\tlabel\tscore\n")
        for res in results:
            f.write(f"{res['utterance_id']}\t{res['label']}\t{res['score']:.6f}\n")

    print(f"Scores saved to {save_path}")
    return results


def calculate_eer(bonafide_scores, spoof_scores):
    """
    Calculate Equal Error Rate (EER).

    Args:
        bonafide_scores: Scores for bonafide samples (higher = more bonafide)
        spoof_scores: Scores for spoof samples (higher = more bonafide)

    Returns:
        eer: Equal Error Rate
        threshold: EER threshold
    """
    # Combine scores and labels
    y_true = np.concatenate([
        np.ones(len(bonafide_scores)),
        np.zeros(len(spoof_scores))
    ])
    y_score = np.concatenate([bonafide_scores, spoof_scores])

    # Sort by score
    sorted_indices = np.argsort(y_score)
    y_true_sorted = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]

    # Calculate FAR and FRR at each threshold
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)

    far = []
    frr = []
    thresholds = []

    for i in range(len(y_score_sorted)):
        threshold = y_score_sorted[i]

        # FRR: bonafide samples below threshold / total bonafide
        frr_val = np.sum((y_score < threshold) & (y_true == 1)) / n_bonafide

        # FAR: spoof samples above threshold / total spoof
        far_val = np.sum((y_score >= threshold) & (y_true == 0)) / n_spoof

        far.append(far_val)
        frr.append(frr_val)
        thresholds.append(threshold)

    far = np.array(far)
    frr = np.array(frr)
    thresholds = np.array(thresholds)

    # Find EER (where FAR ≈ FRR)
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    eer_threshold = thresholds[min_index]

    return eer, eer_threshold


def evaluate_scores(results, output_file):
    """
    Evaluate EER from score results.

    Args:
        results: List of dicts with keys: utterance_id, label, score
        output_file: Path to save evaluation metrics
    """
    # Separate bonafide and spoof scores
    bonafide_scores = [r['score'] for r in results if r['label'] == 'bonafide']
    spoof_scores = [r['score'] for r in results if r['label'] == 'spoof']

    print(f"\nEvaluation:")
    print(f"  Bonafide samples: {len(bonafide_scores)}")
    print(f"  Spoof samples: {len(spoof_scores)}")

    # Calculate EER
    eer, threshold = calculate_eer(
        np.array(bonafide_scores),
        np.array(spoof_scores)
    )

    print(f"\n  EER: {eer*100:.3f}%")
    print(f"  EER threshold: {threshold:.6f}")

    # Save results
    with open(output_file, 'w') as f:
        f.write("ASVspoof 2021 Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Bonafide samples: {len(bonafide_scores)}\n")
        f.write(f"Spoof samples: {len(spoof_scores)}\n")
        f.write(f"Total samples: {len(results)}\n\n")
        f.write(f"EER: {eer*100:.3f}%\n")
        f.write(f"EER threshold: {threshold:.6f}\n")

    print(f"\nResults saved to {output_file}")

    return eer, threshold


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_config = config['model_config']

    # Set seed for reproducibility
    set_seed(args.seed, config)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = get_model(model_config, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from: {args.model_path}")

    # Load protocol
    print("\nLoading protocol...")
    utterance_ids, labels = load_protocol_csv(args.protocol_csv, subset=args.subset)

    # Filter to only include files that actually exist and can be loaded
    utterance_ids, labels, missing_ids, corrupted_ids = filter_existing_files(
        utterance_ids, labels, args.audio_dir, validate_audio=args.validate_audio
    )

    if len(utterance_ids) == 0:
        print("\n⚠ ERROR: No valid audio files found!")
        print(f"Audio directory: {args.audio_dir}")
        print("\nPossible issues:")
        print("  1. Incorrect audio directory path")
        print("  2. Audio files don't exist for any utterances in the CSV")
        print("  3. All audio files are corrupted or unreadable")
        print("\nPlease run test_audio_loading.py to diagnose:")
        print(f"python test_audio_loading.py --protocol_csv {args.protocol_csv} --audio_dir {args.audio_dir}")
        sys.exit(1)

    # Save list of missing/corrupted files for reference
    if (missing_ids or corrupted_ids) and args.save_missing_list:
        problem_file = args.output_score.replace('.txt', '_problem_files.txt')
        with open(problem_file, 'w') as f:
            f.write(f"# Problem Files Report\n")
            f.write(f"# Missing files: {len(missing_ids)}\n")
            f.write(f"# Corrupted files: {len(corrupted_ids)}\n")
            f.write(f"# Total problem files: {len(missing_ids) + len(corrupted_ids)}\n\n")

            if missing_ids:
                f.write("=== MISSING FILES ===\n")
                for utt_id in missing_ids:
                    f.write(f"{utt_id}\n")
                f.write("\n")

            if corrupted_ids:
                f.write("=== CORRUPTED/UNREADABLE FILES ===\n")
                for utt_id in corrupted_ids:
                    f.write(f"{utt_id}\n")

        print(f"\nList of problem files saved to: {problem_file}")

    # Create dataset and dataloader
    print("\nCreating dataset...")
    eval_set = Dataset_ASVspoof2021_eval(
        utterance_ids=utterance_ids,
        labels=labels,
        base_dir=args.audio_dir,
        cut=model_config.get('nb_samp', 64600)
    )

    # Use num_workers=0 to avoid multiprocessing issues with soundfile
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid soundfile issues
        pin_memory=False  # Disable pin_memory to avoid threading issues
    )

    # Generate scores
    print("\nGenerating scores...")
    results = produce_evaluation_scores(eval_loader, model, device, args.output_score)

    # Evaluate EER
    print("\nEvaluating...")
    output_metrics = args.output_score.replace('.txt', '_metrics.txt')
    eer, threshold = evaluate_scores(results, output_metrics)

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate AASIST on ASVspoof 2021')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config file (e.g., ./config/AASIST.conf)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (e.g., best.pth)')
    parser.add_argument('--protocol_csv', type=str, required=True,
                        help='Path to ASVspoof 2021 protocol CSV file')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output_score', type=str, default='./scores_2021.txt',
                        help='Output file for scores')
    parser.add_argument('--subset', type=str, default='eval',
                        choices=['eval', 'progress', 'all'],
                        help='Which subset to evaluate (default: eval)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--save_missing_list', action='store_true',
                        help='Save list of missing/corrupted audio files')
    parser.add_argument('--validate_audio', action='store_true', default=True,
                        help='Validate that audio files can be loaded (default: True)')
    parser.add_argument('--skip_validation', dest='validate_audio', action='store_false',
                        help='Skip audio validation (faster but may crash on corrupted files)')
    # Note: num_workers is hardcoded to 0 to avoid soundfile multiprocessing issues

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.protocol_csv):
        print(f"Error: Protocol CSV not found: {args.protocol_csv}")
        sys.exit(1)
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found: {args.audio_dir}")
        sys.exit(1)

    main(args)
