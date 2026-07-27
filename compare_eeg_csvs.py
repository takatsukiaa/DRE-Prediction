"""Compare human EEG timeline CSVs with parser-generated EEG timelines.

The comparison normalizes both layouts to patient-month-feature cells:
    (patient_id, month, feature) -> value

Human timelines use wide CSV chunks under ../csv_temp. Parser timelines use
one long CSV per patient under new_csvs/eeg_timeline. For EEG feature rows,
blank and 0 are both treated as feature absence.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


EEG_FEATURES = [
    "Spike ED",
    "Sharp ED",
    "Spike-and-wave ED",
    "Periodic Discharge",
    "TLED",
    "TRED",
    "FLED",
    "FRED",
    "OLED",
    "ORED",
    "PLED",
    "PRED",
    "CLED",
    "CRED",
    "Delta waves",
    "Theta waves",
    "TL",
    "TR",
    "FL",
    "FR",
    "OL",
    "OR",
    "PL",
    "PR",
    "CL",
    "CR",
]

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

DIFF_FIELDNAMES = [
    "patient_id",
    "month",
    "feature",
    "human_value",
    "parser_value",
    "difference_type",
    "human_file",
    "parser_file",
]

SUMMARY_FIELDNAMES = ["section", "metric", "value"]


@dataclass
class NormalizedCells:
    cells: dict[tuple[str, str, str], str]
    sources: dict[tuple[str, str, str], str]
    patient_months: set[tuple[str, str]]
    file_count: int


def normalize_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""

    try:
        number = Decimal(text)
    except InvalidOperation:
        return text

    if number == 0:
        return ""
    if number == number.to_integral():
        return str(int(number))
    return format(number.normalize(), "f")


def parse_month_label(value: str) -> str | None:
    text = (value or "").strip()
    match = re.fullmatch(r"(\d{2})-([A-Za-z]{3})", text)
    if match:
        month = MONTHS.get(match.group(2).lower())
        if month:
            return f"20{match.group(1)}-{month:02d}"

    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"

    match = re.fullmatch(r"(\d{6})", text)
    if match:
        return f"{text[:4]}-{text[4:]}"

    return None


def store_max_cell(
    cells: dict[tuple[str, str, str], str],
    sources: dict[tuple[str, str, str], str],
    key: tuple[str, str, str],
    value: str,
    source: str,
) -> None:
    old_value = cells.get(key, "")
    if not value:
        cells.setdefault(key, "")
        sources.setdefault(key, source)
        return

    try:
        if not old_value or Decimal(value) > Decimal(old_value):
            cells[key] = value
            sources[key] = source
    except InvalidOperation:
        cells[key] = value
        sources[key] = source


def load_human_cells(human_dir: Path) -> NormalizedCells:
    feature_set = set(EEG_FEATURES)
    cells: dict[tuple[str, str, str], str] = {}
    sources: dict[tuple[str, str, str], str] = {}
    patient_months: set[tuple[str, str]] = set()

    files = sorted(human_dir.glob("Pt*.csv"))
    for filepath in files:
        patient_match = re.search(r"Pt(\d{4})_", filepath.name)
        if not patient_match:
            continue
        patient_id = patient_match.group(1)

        with filepath.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))

        time_row = next((row for row in rows if row and row[0].strip() == "Time"), None)
        if not time_row:
            continue

        months = [parse_month_label(value) for value in time_row[1:]]
        for month in months:
            if month:
                patient_months.add((patient_id, month))

        for row in rows:
            if not row:
                continue
            feature = row[0].strip()
            if feature not in feature_set:
                continue

            for index, month in enumerate(months, start=1):
                if not month:
                    continue
                raw_value = row[index] if index < len(row) else ""
                key = (patient_id, month, feature)
                store_max_cell(cells, sources, key, normalize_value(raw_value), filepath.name)

    return NormalizedCells(cells, sources, patient_months, len(files))


def load_parser_cells(parser_dir: Path) -> NormalizedCells:
    cells: dict[tuple[str, str, str], str] = {}
    sources: dict[tuple[str, str, str], str] = {}
    patient_months: set[tuple[str, str]] = set()

    files = sorted(parser_dir.glob("*_eeg_timeline.csv"))
    for filepath in files:
        patient_match = re.fullmatch(r"(\d{4})(?:DRE|NDRE)_eeg_timeline\.csv", filepath.name)
        if not patient_match:
            continue
        patient_id = patient_match.group(1)

        with filepath.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                month = (row.get("month") or "").strip()
                if not month:
                    continue

                patient_months.add((patient_id, month))
                for feature in EEG_FEATURES:
                    key = (patient_id, month, feature)
                    store_max_cell(
                        cells,
                        sources,
                        key,
                        normalize_value(row.get(feature, "")),
                        filepath.name,
                    )

    return NormalizedCells(cells, sources, patient_months, len(files))


def build_diff_rows(
    human: NormalizedCells, parser: NormalizedCells, include_outside_human_range: bool = False
) -> list[dict[str, str]]:
    diff_rows: list[dict[str, str]] = []

    for patient_id, month, feature in sorted(set(human.cells) | set(parser.cells)):
        key = (patient_id, month, feature)
        human_value = human.cells.get(key, "")
        parser_value = parser.cells.get(key, "")
        if human_value == parser_value:
            continue
        if (patient_id, month) not in human.patient_months and not include_outside_human_range:
            continue

        if human_value and not parser_value:
            difference_type = "missing_in_parser"
        elif parser_value and not human_value and (patient_id, month) in human.patient_months:
            difference_type = "extra_in_parser"
        elif parser_value and not human_value and (patient_id, month) not in human.patient_months:
            difference_type = "parser_month_outside_human_csv"
        elif human_value and parser_value:
            difference_type = "value_mismatch"
        else:
            difference_type = "other"

        diff_rows.append(
            {
                "patient_id": patient_id,
                "month": month,
                "feature": feature,
                "human_value": human_value,
                "parser_value": parser_value,
                "difference_type": difference_type,
                "human_file": human.sources.get(key, ""),
                "parser_file": parser.sources.get(key, ""),
            }
        )

    return diff_rows


def build_summary_rows(
    human: NormalizedCells, parser: NormalizedCells, diff_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    inside_rows = [
        row for row in diff_rows if row["difference_type"] != "parser_month_outside_human_csv"
    ]
    rows: list[dict[str, str]] = [
        {"section": "overall", "metric": "human_csv_files", "value": str(human.file_count)},
        {
            "section": "overall",
            "metric": "parser_timeline_files",
            "value": str(parser.file_count),
        },
        {
            "section": "overall",
            "metric": "human_patient_months",
            "value": str(len(human.patient_months)),
        },
        {
            "section": "overall",
            "metric": "parser_patient_months",
            "value": str(len(parser.patient_months)),
        },
        {
            "section": "overall",
            "metric": "different_cells_total",
            "value": str(len(diff_rows)),
        },
        {
            "section": "overall",
            "metric": "inside_human_months_total",
            "value": str(len(inside_rows)),
        },
    ]

    diff_counts = Counter(row["difference_type"] for row in diff_rows)
    for difference_type in [
        "missing_in_parser",
        "extra_in_parser",
        "value_mismatch",
        "parser_month_outside_human_csv",
        "other",
    ]:
        rows.append(
            {
                "section": "difference_type",
                "metric": difference_type,
                "value": str(diff_counts.get(difference_type, 0)),
            }
        )

    for feature, count in Counter(row["feature"] for row in inside_rows).most_common():
        rows.append({"section": "inside_by_feature", "metric": feature, "value": str(count)})

    for patient_id, count in Counter(row["patient_id"] for row in inside_rows).most_common():
        rows.append({"section": "inside_by_patient", "metric": patient_id, "value": str(count)})

    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    human_dir: Path,
    parser_dir: Path,
    output_dir: Path,
    include_outside_human_range: bool = False,
) -> tuple[Path, Path]:
    human = load_human_cells(human_dir)
    parser = load_parser_cells(parser_dir)
    diff_rows = build_diff_rows(human, parser, include_outside_human_range)
    summary_rows = build_summary_rows(human, parser, diff_rows)

    diff_path = output_dir / "eeg_comparison_diff.csv"
    summary_path = output_dir / "eeg_comparison_summary.csv"
    write_csv(diff_path, DIFF_FIELDNAMES, diff_rows)
    write_csv(summary_path, SUMMARY_FIELDNAMES, summary_rows)
    return diff_path, summary_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-dir",
        type=Path,
        default=repo_root.parent / "csv_temp",
        help="Directory containing human Pt*.csv timeline chunks.",
    )
    parser.add_argument(
        "--parser-dir",
        type=Path,
        default=repo_root / "new_csvs" / "eeg_timeline",
        help="Directory containing parser *_eeg_timeline.csv files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=repo_root / "new_csvs",
        help="Directory where comparison CSVs will be written.",
    )
    parser.add_argument(
        "--include-outside-human-range",
        action="store_true",
        help="Include parser-only patient-months not covered by the human CSV timelines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diff_path, summary_path = run_comparison(
        args.human_dir,
        args.parser_dir,
        args.output_dir,
        include_outside_human_range=args.include_outside_human_range,
    )
    print(f"Wrote {diff_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
