from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from cqcc_gmm.data import collect_audio_files, read_manifest, split_by_label
from cqcc_gmm.features import CqccConfig
from cqcc_gmm.model import CqccGmmDetector, GmmConfig
from cqcc_gmm.reporting import write_evaluation_report, write_training_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CQCC-GMM AI audio spoof detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train bonafide/spoof GMMs")
    train.add_argument("--manifest", type=Path, help="CSV/TSV with path,label columns")
    train.add_argument("--bonafide-dir", type=Path, help="Directory containing real audio")
    train.add_argument("--spoof-dir", type=Path, help="Directory containing spoof/AI audio")
    train.add_argument("--model", type=Path, required=True, help="Output .joblib model path")
    train.add_argument("--components", type=int, default=16, help="GMM components per class")
    train.add_argument("--sample-rate", type=int, default=16_000)
    train.add_argument("--max-iter", type=int, default=200)
    train.add_argument(
        "--max-frames-per-class",
        type=int,
        default=250_000,
        help="Random CQCC frame cap per class. Use 0 to keep all frames.",
    )
    train.add_argument("--report-dir", type=Path, help="Directory for training plots and CSV summary")

    predict = subparsers.add_parser("predict", help="Score audio with a trained model")
    predict.add_argument("--model", type=Path, required=True, help="Trained .joblib model")
    predict.add_argument("--input", type=Path, required=True, help="Audio file or directory")
    predict.add_argument("--output", type=Path, help="Optional CSV output path")
    predict.add_argument("--threshold", type=float, default=0.0)

    evaluate = subparsers.add_parser("evaluate", help="Score a labeled manifest and plot metrics")
    evaluate.add_argument("--model", type=Path, required=True, help="Trained .joblib model")
    evaluate.add_argument("--manifest", type=Path, required=True, help="CSV/TSV with path,label columns")
    evaluate.add_argument("--output", type=Path, required=True, help="Prediction CSV output path")
    evaluate.add_argument("--report-dir", type=Path, required=True, help="Directory for evaluation plots")
    evaluate.add_argument("--threshold", type=float, default=0.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return train_command(args)
    if args.command == "predict":
        return predict_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def train_command(args: argparse.Namespace) -> int:
    if args.manifest:
        files, labels = read_manifest(args.manifest)
        bonafide_files, spoof_files = split_by_label(files, labels)
    elif args.bonafide_dir and args.spoof_dir:
        bonafide_files = collect_audio_files(args.bonafide_dir)
        spoof_files = collect_audio_files(args.spoof_dir)
    else:
        raise SystemExit("Provide either --manifest or both --bonafide-dir and --spoof-dir.")

    detector = CqccGmmDetector(
        feature_config=CqccConfig(sample_rate=args.sample_rate),
        gmm_config=GmmConfig(
            n_components=args.components,
            max_iter=args.max_iter,
            max_frames_per_class=args.max_frames_per_class or None,
        ),
    )
    detector.fit(bonafide_files, spoof_files)
    detector.save(args.model)
    if args.report_dir:
        write_training_report(args.report_dir, detector.training_stats)
    print(f"Saved model: {args.model}")
    print(f"Bonafide files: {len(bonafide_files)}")
    print(f"Spoof files: {len(spoof_files)}")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    detector = CqccGmmDetector.load(args.model)
    files = collect_audio_files(args.input)
    if not files:
        raise SystemExit(f"No supported audio files found: {args.input}")

    rows = [detector.score_file(file, threshold=args.threshold) for file in files]
    fieldnames = ["path", "score", "prediction", "bonafide_loglike", "spoof_loglike"]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {args.output}")
    else:
        writer = csv.DictWriter(_StdoutProxy(), fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    detector = CqccGmmDetector.load(args.model)
    files, labels = read_manifest(args.manifest)
    rows = []
    for file, label in tqdm(list(zip(files, labels, strict=True)), desc="Evaluating", unit="file", ascii=True):
        scored = detector.score_file(file, threshold=args.threshold)
        rows.append(
            {
                "path": scored["path"],
                "label": label,
                "prediction": scored["prediction"],
                "score": scored["score"],
                "bonafide_loglike": scored["bonafide_loglike"],
                "spoof_loglike": scored["spoof_loglike"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "prediction", "score", "bonafide_loglike", "spoof_loglike"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = write_evaluation_report(args.report_dir, rows)
    print(f"Saved predictions: {args.output}")
    print(f"Saved evaluation report: {args.report_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"EER: {metrics['eer'] * 100.0:.2f}%")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    return 0


class _StdoutProxy:
    def write(self, value: str) -> int:
        print(value, end="")
        return len(value)

    def flush(self) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
