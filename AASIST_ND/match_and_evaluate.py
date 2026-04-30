"""
Match pre-computed utterance-level scores with labeled ASVspoof 2021 subset
and evaluate EER.

This is useful if you've already generated scores using the existing code
and just want to match them with the 2021 labeled subset.

Usage:
    python match_and_evaluate.py --score_file SCORES.txt \
                                  --protocol_csv PROTOCOL.csv \
                                  --output_file RESULTS.txt

Example:
    python match_and_evaluate.py \
        --score_file ./eval_scores.txt \
        --protocol_csv /Users/tdao/Downloads/asvspoof2021_eval_whisper_merged_with_labels.csv \
        --output_file ./matched_results.txt
"""

import argparse
import csv
import numpy as np


def load_scores(score_file):
    """
    Load scores from file.

    Expected format (space or tab separated):
    utterance_id source key score
    OR
    utterance_id label score

    Returns:
        dict: {utterance_id: score}
    """
    scores = {}

    with open(score_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines, comments, and header lines
            if not line or line.startswith('#'):
                continue

            # Skip header line (contains words like "utterance_id", "score", etc.)
            if line_num == 1 and any(keyword in line.lower() for keyword in ['utterance_id', 'label', 'score']):
                continue

            # Split by whitespace or tab
            parts = line.split()

            if len(parts) >= 2:
                utt_id = parts[0]
                try:
                    score = float(parts[-1])  # Last column is score
                    scores[utt_id] = score
                except ValueError:
                    # Skip lines where last column isn't a number (likely another header)
                    continue

    print(f"Loaded scores for {len(scores)} utterances")
    return scores


def load_protocol_csv(csv_path, subset='eval'):
    """
    Load ASVspoof 2021 protocol CSV.

    Returns:
        dict: {utterance_id: label_str}
    """
    labels = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt_id = row['utterance_id']
            label_str = row['label_str']
            row_subset = row['subset']

            # Filter by subset
            if subset != 'all' and row_subset != subset:
                continue

            labels[utt_id] = label_str

    print(f"Loaded labels for {len(labels)} utterances from CSV")
    print(f"  Bonafide: {sum(1 for l in labels.values() if l == 'bonafide')}")
    print(f"  Spoof: {sum(1 for l in labels.values() if l == 'spoof')}")

    return labels


def match_scores_with_labels(scores, labels):
    """
    Match scores with labels.

    Returns:
        matched: List of dicts with keys: utterance_id, label, score
        unmatched_ids: List of utterance IDs without scores
    """
    matched = []
    unmatched_ids = []

    for utt_id, label in labels.items():
        if utt_id in scores:
            matched.append({
                'utterance_id': utt_id,
                'label': label,
                'score': scores[utt_id]
            })
        else:
            unmatched_ids.append(utt_id)

    print(f"\nMatching results:")
    print(f"  Matched: {len(matched)} utterances")
    print(f"  Unmatched: {len(unmatched_ids)} utterances")

    return matched, unmatched_ids


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


def evaluate_matched_scores(matched_results, output_file):
    """
    Evaluate EER from matched results.

    Args:
        matched_results: List of dicts with keys: utterance_id, label, score
        output_file: Path to save evaluation metrics
    """
    # Separate bonafide and spoof scores
    bonafide_scores = [r['score'] for r in matched_results if r['label'] == 'bonafide']
    spoof_scores = [r['score'] for r in matched_results if r['label'] == 'spoof']

    print(f"\nEvaluation:")
    print(f"  Bonafide samples: {len(bonafide_scores)}")
    print(f"  Spoof samples: {len(spoof_scores)}")

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        print("Error: Need both bonafide and spoof samples for evaluation!")
        return None, None

    # Calculate EER
    eer, threshold = calculate_eer(
        np.array(bonafide_scores),
        np.array(spoof_scores)
    )

    print(f"\n  EER: {eer*100:.3f}%")
    print(f"  EER threshold: {threshold:.6f}")

    # Calculate additional stats
    bonafide_mean = np.mean(bonafide_scores)
    bonafide_std = np.std(bonafide_scores)
    spoof_mean = np.mean(spoof_scores)
    spoof_std = np.std(spoof_scores)

    print(f"\n  Bonafide score: {bonafide_mean:.4f} ± {bonafide_std:.4f}")
    print(f"  Spoof score: {spoof_mean:.4f} ± {spoof_std:.4f}")

    # Save results
    with open(output_file, 'w') as f:
        f.write("ASVspoof 2021 Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Bonafide samples: {len(bonafide_scores)}\n")
        f.write(f"Spoof samples: {len(spoof_scores)}\n")
        f.write(f"Total samples: {len(matched_results)}\n\n")
        f.write(f"EER: {eer*100:.3f}%\n")
        f.write(f"EER threshold: {threshold:.6f}\n\n")
        f.write(f"Bonafide score (mean ± std): {bonafide_mean:.4f} ± {bonafide_std:.4f}\n")
        f.write(f"Spoof score (mean ± std): {spoof_mean:.4f} ± {spoof_std:.4f}\n")

    # Also save matched scores in standard format
    matched_score_file = output_file.replace('.txt', '_matched_scores.txt')
    with open(matched_score_file, 'w') as f:
        f.write("utterance_id\tlabel\tscore\n")
        for r in matched_results:
            f.write(f"{r['utterance_id']}\t{r['label']}\t{r['score']:.6f}\n")

    print(f"\nResults saved to {output_file}")
    print(f"Matched scores saved to {matched_score_file}")

    return eer, threshold


def main(args):
    print("Loading score file...")
    scores = load_scores(args.score_file)

    print("\nLoading protocol CSV...")
    labels = load_protocol_csv(args.protocol_csv, subset=args.subset)

    print("\nMatching scores with labels...")
    matched, unmatched = match_scores_with_labels(scores, labels)

    if unmatched and args.show_unmatched:
        print(f"\nFirst 10 unmatched IDs:")
        for utt_id in unmatched[:10]:
            print(f"  {utt_id}")

    if len(matched) == 0:
        print("\nError: No matched utterances found!")
        print("Make sure the utterance IDs in your score file match those in the CSV.")
        return

    print("\nEvaluating matched scores...")
    eer, threshold = evaluate_matched_scores(matched, args.output_file)

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Match scores with ASVspoof 2021 labels and evaluate EER'
    )

    parser.add_argument('--score_file', type=str, required=True,
                        help='Path to score file (format: utterance_id ... score)')
    parser.add_argument('--protocol_csv', type=str, required=True,
                        help='Path to ASVspoof 2021 protocol CSV file')
    parser.add_argument('--output_file', type=str, default='./matched_results.txt',
                        help='Output file for evaluation results')
    parser.add_argument('--subset', type=str, default='eval',
                        choices=['eval', 'progress', 'all'],
                        help='Which subset to evaluate (default: eval)')
    parser.add_argument('--show_unmatched', action='store_true',
                        help='Show unmatched utterance IDs')

    args = parser.parse_args()

    main(args)
