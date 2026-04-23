from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a labeled eval manifest by filtering an existing manifest with utterance IDs."
    )
    parser.add_argument("--ids-csv", type=Path, required=True, help="CSV containing an utterance_id column")
    parser.add_argument("--base-manifest", type=Path, required=True, help="Manifest with path,label,source,id columns")
    parser.add_argument("--output", type=Path, required=True, help="Filtered manifest output path")
    args = parser.parse_args()

    wanted_ids = read_ids(args.ids_csv)
    base_rows = read_base_manifest(args.base_manifest)

    output_rows = []
    missing = []
    seen = set()
    duplicate_rows = 0
    for utt_id in wanted_ids:
        if utt_id in seen:
            duplicate_rows += 1
            continue
        seen.add(utt_id)
        row = base_rows.get(utt_id)
        if row is None:
            missing.append(utt_id)
            continue
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["path", "label", "source", "id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    bonafide = sum(1 for row in output_rows if row["label"] == "bonafide")
    spoof = sum(1 for row in output_rows if row["label"] == "spoof")
    print(f"ids_csv_rows={len(wanted_ids)}")
    print(f"unique_ids={len(seen)}")
    print(f"duplicate_rows_skipped={duplicate_rows}")
    print(f"matched={len(output_rows)}")
    print(f"missing={len(missing)}")
    print(f"bonafide={bonafide}")
    print(f"spoof={spoof}")
    print(f"output={args.output}")
    if missing:
        print("first_missing=" + ",".join(missing[:10]))
    return 0 if not missing else 1


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "utterance_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an utterance_id column")
        return [row["utterance_id"].strip() for row in reader if row.get("utterance_id", "").strip()]


def read_base_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "source", "id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            rows[row["id"]] = {
                "path": row["path"],
                "label": row["label"],
                "source": row["source"],
                "id": row["id"],
            }
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
