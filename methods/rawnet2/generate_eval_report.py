import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OFFICIAL_REFERENCE_BASELINES = [
    {
        "system": "LFCC-LCNN official baseline",
        "subset": "ASVspoof2021 LA evaluation phase subset",
        "eer_percent": 9.26,
        "min_tDCF": 0.3445,
        "source_title": "Das et al. 2021, Table 8",
        "source_url": "https://www.isca-archive.org/asvspoof_2021/das21_asvspoof.pdf",
        "reference_kind": "official",
    },
    {
        "system": "RawNet2 official baseline",
        "subset": "ASVspoof2021 LA evaluation phase subset",
        "eer_percent": 9.50,
        "min_tDCF": 0.4257,
        "source_title": "Das et al. 2021, Table 8",
        "source_url": "https://www.isca-archive.org/asvspoof_2021/das21_asvspoof.pdf",
        "reference_kind": "official",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an evaluation comparison report for ASVspoof 2021 LA."
    )
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--comparison-plot", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--model-label", type=str, default="RawNet2 local model")
    parser.add_argument("--scored-csv", type=Path, default=None)
    parser.add_argument("--reference-html", type=Path, default=None)
    return parser.parse_args()


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_metrics(metrics_json_path):
    with open(metrics_json_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    overall = metrics.get("overall") or {}
    if not overall:
        raise ValueError(f"No overall metrics found in {metrics_json_path}")
    return metrics, overall


def _parse_float(text):
    if text is None:
        return None
    cleaned = text.replace(",", "").strip()
    if not cleaned:
        return None
    return float(cleaned)


def _to_percent_if_fraction(value):
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


def load_reference_baselines(reference_html_path=None):
    notes = []
    if reference_html_path is not None and Path(reference_html_path).exists():
        reference = parse_project_report_lcnn(reference_html_path)
        notes.append(
            "Primary comparison uses the LCNN baseline reported in PROJECT_REPORT.html."
        )
        if reference.get("num_rows") is not None:
            notes.append(
                f"PROJECT_REPORT.html reports LCNN results on {reference['num_rows']} samples."
            )
        return [reference], notes

    notes.append(
        "PROJECT_REPORT.html was not provided or not found, so the report falls back to official ASVspoof2021 baselines."
    )
    return OFFICIAL_REFERENCE_BASELINES, notes


def parse_project_report_lcnn(reference_html_path):
    reference_html_path = Path(reference_html_path)
    text = reference_html_path.read_text(encoding="utf-8", errors="ignore")

    card_match = re.search(
        r"LCNN.*?<div class=\"stats\">(?P<stats>.*?)</div>\s*</div>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if card_match is None:
        raise ValueError(f"Could not locate LCNN stats in {reference_html_path}")

    stats = {}
    for label, value in re.findall(
        r"<div class=\"row\"><span>([^<]+)</span><span>([^<]+)</span></div>",
        card_match.group("stats"),
        flags=re.IGNORECASE,
    ):
        stats[label.strip()] = _parse_float(value)

    sample_count = None
    sample_match = re.search(
        r"ASVspoof2021\s+LA\s+eval\s*\(([\d,]+)\s*",
        text,
        flags=re.IGNORECASE,
    )
    if sample_match is not None:
        sample_count = int(sample_match.group(1).replace(",", ""))

    eer_value = stats.get("EER")
    if eer_value is None:
        raise ValueError(f"Could not parse LCNN EER from {reference_html_path}")

    return {
        "system": "LCNN baseline from PROJECT_REPORT.html",
        "subset": (
            f"PROJECT_REPORT.html ASVspoof2021 LA eval ({sample_count} samples)"
            if sample_count is not None
            else "PROJECT_REPORT.html ASVspoof2021 LA eval"
        ),
        "eer_percent": _to_percent_if_fraction(eer_value),
        "min_tDCF": None,
        "accuracy": stats.get("Acc @ 0.5"),
        "balanced_accuracy": None,
        "auc": stats.get("AUC"),
        "f1_spoof": stats.get("F1-spoof @ EER"),
        "num_rows": sample_count,
        "num_rows_with_truth": sample_count,
        "source_title": "PROJECT_REPORT.html LCNN branch metrics",
        "source_url": str(reference_html_path),
        "reference_kind": "project_report",
    }


def build_summary(metrics, overall, model_label, scored_csv, references, notes):
    model_row = {
        "system": model_label,
        "subset": "User-selected CSV aligned with ASVspoof2021 LA CM keys",
        "eer_percent": overall.get("eer_percent"),
        "min_tDCF": None,
        "accuracy": overall.get("accuracy"),
        "balanced_accuracy": overall.get("balanced_accuracy"),
        "auc": None,
        "f1_spoof": (overall.get("spoof_metrics") or {}).get("f1"),
        "num_rows": overall.get("num_rows"),
        "num_rows_with_truth": metrics.get("num_rows_with_truth"),
        "source_title": "Local evaluation",
        "source_url": str(scored_csv) if scored_csv is not None else "",
        "reference_kind": "local",
    }

    comparisons = []
    model_eer = model_row["eer_percent"]
    for reference in references:
        delta_eer = None
        relative_gain = None
        caveat = None
        if model_eer is not None and reference.get("eer_percent") is not None:
            delta_eer = model_eer - reference["eer_percent"]
            if reference["eer_percent"]:
                relative_gain = (
                    (reference["eer_percent"] - model_eer) / reference["eer_percent"] * 100.0
                )
        if (
            model_row.get("num_rows_with_truth") is not None
            and reference.get("num_rows") is not None
            and model_row["num_rows_with_truth"] != reference["num_rows"]
        ):
            caveat = (
                f"Sample count differs: local rows={model_row['num_rows_with_truth']}, "
                f"reference rows={reference['num_rows']}."
            )
        comparisons.append(
            {
                "reference_system": reference["system"],
                "reference_subset": reference["subset"],
                "reference_eer_percent": reference.get("eer_percent"),
                "reference_min_tDCF": reference.get("min_tDCF"),
                "reference_accuracy": reference.get("accuracy"),
                "reference_auc": reference.get("auc"),
                "reference_f1_spoof": reference.get("f1_spoof"),
                "delta_eer_percent": delta_eer,
                "relative_eer_improvement_percent": relative_gain,
                "source_title": reference["source_title"],
                "source_url": reference["source_url"],
                "caveat": caveat,
            }
        )
        if caveat is not None:
            notes.append(caveat)

    return {
        "model": model_row,
        "references": references,
        "comparisons": comparisons,
        "metrics": metrics,
        "notes": list(dict.fromkeys(notes)),
    }


def write_comparison_csv(summary, output_path):
    ensure_parent(output_path)
    rows = [
        {
            "system": summary["model"]["system"],
            "subset": summary["model"]["subset"],
            "eer_percent": summary["model"]["eer_percent"],
            "min_tDCF": summary["model"]["min_tDCF"],
            "accuracy": summary["model"]["accuracy"],
            "balanced_accuracy": summary["model"]["balanced_accuracy"],
            "auc": summary["model"]["auc"],
            "f1_spoof": summary["model"]["f1_spoof"],
            "num_rows": summary["model"]["num_rows"],
            "num_rows_with_truth": summary["model"]["num_rows_with_truth"],
            "source_title": summary["model"]["source_title"],
            "source_url": summary["model"]["source_url"],
        }
    ]
    for reference in summary["references"]:
        rows.append(
            {
                "system": reference["system"],
                "subset": reference["subset"],
                "eer_percent": reference.get("eer_percent"),
                "min_tDCF": reference.get("min_tDCF"),
                "accuracy": reference.get("accuracy"),
                "balanced_accuracy": reference.get("balanced_accuracy"),
                "auc": reference.get("auc"),
                "f1_spoof": reference.get("f1_spoof"),
                "num_rows": reference.get("num_rows"),
                "num_rows_with_truth": reference.get("num_rows_with_truth"),
                "source_title": reference["source_title"],
                "source_url": reference["source_url"],
            }
        )

    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_plot(summary, output_path):
    ensure_parent(output_path)
    systems = [summary["model"]["system"]] + [ref["system"] for ref in summary["references"]]
    eer_values = [summary["model"]["eer_percent"]] + [
        ref.get("eer_percent") for ref in summary["references"]
    ]
    plot_values = [0.0 if value is None else value for value in eer_values]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors = [color_cycle[index % len(color_cycle)] for index in range(len(systems))]
    bars = ax.bar(systems, plot_values, color=colors)
    ax.set_title("EER Comparison")
    ax.set_ylabel("EER (%)")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, eer_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            "n/a" if value is None else f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report_md(summary, output_path):
    ensure_parent(output_path)
    model = summary["model"]
    comparisons = summary["comparisons"]
    references = summary["references"]

    lines = [
        "# RawNet2 Evaluation Summary",
        "",
        "## Local Evaluation",
        "",
        f"- Model: {model['system']}",
        f"- Evaluated subset: {model['subset']}",
        f"- Rows with truth: {model['num_rows_with_truth']}",
        f"- Accuracy: {model['accuracy']:.4f}" if model["accuracy"] is not None else "- Accuracy: n/a",
        (
            f"- Balanced accuracy: {model['balanced_accuracy']:.4f}"
            if model["balanced_accuracy"] is not None
            else "- Balanced accuracy: n/a"
        ),
        f"- F1-spoof: {model['f1_spoof']:.4f}" if model["f1_spoof"] is not None else "- F1-spoof: n/a",
        f"- EER: {model['eer_percent']:.4f}%" if model["eer_percent"] is not None else "- EER: n/a",
        "",
        "## Reference Baselines",
        "",
    ]

    for reference in references:
        metric_bits = []
        if reference.get("eer_percent") is not None:
            metric_bits.append(f"EER {reference['eer_percent']:.2f}%")
        if reference.get("accuracy") is not None:
            metric_bits.append(f"Accuracy {reference['accuracy']:.4f}")
        if reference.get("auc") is not None:
            metric_bits.append(f"AUC {reference['auc']:.4f}")
        if reference.get("f1_spoof") is not None:
            metric_bits.append(f"F1-spoof {reference['f1_spoof']:.4f}")
        if reference.get("min_tDCF") is not None:
            metric_bits.append(f"min t-DCF {reference['min_tDCF']:.4f}")
        if reference.get("num_rows") is not None:
            metric_bits.append(f"rows {reference['num_rows']}")
        lines.extend(
            [
                f"- {reference['system']}: {', '.join(metric_bits) if metric_bits else 'n/a'}",
                f"  Source: {reference['source_title']} ({reference['source_url']})",
            ]
        )

    lines.extend(["", "## Delta vs References", ""])
    for comparison in comparisons:
        delta = comparison["delta_eer_percent"]
        relative = comparison["relative_eer_improvement_percent"]
        delta_text = "n/a" if delta is None else f"{delta:+.4f} percentage points"
        relative_text = "n/a" if relative is None else f"{relative:+.2f}%"
        line = (
            f"- Against {comparison['reference_system']}: delta EER {delta_text}, "
            f"relative improvement {relative_text}"
        )
        if comparison.get("caveat"):
            line += f". Caveat: {comparison['caveat']}"
        lines.append(line)

    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(summary, output_path):
    ensure_parent(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    metrics, overall = load_metrics(args.metrics_json)
    references, notes = load_reference_baselines(args.reference_html)
    summary = build_summary(
        metrics=metrics,
        overall=overall,
        model_label=args.model_label,
        scored_csv=args.scored_csv,
        references=references,
        notes=notes,
    )
    write_comparison_csv(summary, args.comparison_csv)
    write_comparison_plot(summary, args.comparison_plot)
    write_report_md(summary, args.report_md)
    write_summary_json(summary, args.summary_json)
    print(f"Saved comparison CSV to {args.comparison_csv.resolve()}")
    print(f"Saved comparison plot to {args.comparison_plot.resolve()}")
    print(f"Saved markdown report to {args.report_md.resolve()}")
    print(f"Saved summary JSON to {args.summary_json.resolve()}")


if __name__ == "__main__":
    main()
