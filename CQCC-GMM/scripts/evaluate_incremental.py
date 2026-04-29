from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from cqcc_gmm.data import read_manifest
from cqcc_gmm.model import CqccGmmDetector
from cqcc_gmm.reporting import write_evaluation_report


FIELDNAMES = ["id", "path", "label", "prediction", "score", "bonafide_loglike", "spoof_loglike"]
ERROR_FIELDNAMES = ["id", "path", "label", "error"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally evaluate a labeled manifest.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    files, labels = read_manifest(args.manifest)
    ids = read_manifest_ids(args.manifest)
    existing_rows = read_existing_rows(args.output)
    error_path = args.report_dir / "errors.csv"
    successful_ids = {row["id"] for row in existing_rows}
    existing_errors = {
        row["id"]: row for row in read_existing_errors(error_path) if row["id"] not in successful_ids
    }
    done_error_ids = {utt_id for utt_id, row in existing_errors.items() if not is_retryable_error(row)}
    done_ids = successful_ids | done_error_ids
    pending = [(utt_id, file, label) for utt_id, file, label in zip(ids, files, labels, strict=True) if utt_id not in done_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if pending:
        detector = CqccGmmDetector.load(args.model)
        batch = pending[: args.batch_size]
        append_writer = PredictionAppender(args.output)
        for utt_id, file, label in tqdm(batch, desc="Evaluating batch", unit="file", ascii=True):
            try:
                scored = detector.score_file(file, threshold=args.threshold)
                row = {
                    "id": utt_id,
                    "path": scored["path"],
                    "label": label,
                    "prediction": scored["prediction"],
                    "score": scored["score"],
                    "bonafide_loglike": scored["bonafide_loglike"],
                    "spoof_loglike": scored["spoof_loglike"],
                }
                append_writer.write(row)
                existing_rows.append(row)
                existing_errors.pop(utt_id, None)
            except Exception as exc:  # Keep long evaluations moving past corrupt audio.
                error_row = {"id": utt_id, "path": str(file), "label": label, "error": repr(exc)}
                existing_errors[utt_id] = error_row
            completed = completed_count(existing_rows, existing_errors)
            write_progress(args.report_dir / "incremental_progress.csv", len(files), completed, len(files) - completed)
            write_errors(error_path, existing_errors.values())
        append_writer.close()

    total = len(files)
    completed = completed_count(existing_rows, existing_errors)
    remaining = total - completed
    write_progress(args.report_dir / "incremental_progress.csv", total, completed, remaining)
    print(f"total={total}")
    print(f"completed={completed}")
    print(f"remaining={remaining}")
    print(f"errors={len(existing_errors)}")
    print(f"output={args.output}")

    if remaining == 0:
        metrics = write_evaluation_report(args.report_dir, existing_rows)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"EER: {metrics['eer'] * 100.0:.2f}%")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        return 0
    return 2


def read_manifest_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an id column")
        return [row["id"] for row in reader]


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def read_existing_errors(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def is_retryable_error(row: dict[str, str]) -> bool:
    return "LibsndfileError" in row.get("error", "")


def completed_count(rows: list[dict[str, str]], errors: dict[str, dict[str, str]]) -> int:
    return len(rows) + sum(1 for row in errors.values() if not is_retryable_error(row))


def write_errors(path: Path, rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class PredictionAppender:
    def __init__(self, path: Path) -> None:
        write_header = not path.exists() or path.stat().st_size == 0
        self.handle = path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDNAMES)
        if write_header:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def write_progress(path: Path, total: int, completed: int, remaining: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerow(["total", total])
        writer.writerow(["completed", completed])
        writer.writerow(["remaining", remaining])


if __name__ == "__main__":
    raise SystemExit(main())
