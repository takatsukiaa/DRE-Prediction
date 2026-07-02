import argparse
import csv
import hashlib
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from docx_parser_seizure import Evidence, aggregate_evidence, safe_filename


DEFAULT_EVIDENCE_FILE = "new_csvs/seizure_extraction_evidence.csv"
DEFAULT_OUTPUT_DIR = "new_csvs/reviewed_seizure_outputs"
LOW_CONFIDENCE_VALUES = {"low", "medium"}

CORRECTION_FIELDS = [
    "row_key",
    "review_status",
    "human_count",
    "reviewed_at",
    "patient_id",
    "cohort",
    "source_file",
    "month",
    "original_count_min",
    "original_count_max",
    "original_count_estimate",
    "original_confidence",
    "source_section",
    "method",
    "evidence",
]

EVIDENCE_FIELDS = [
    "patient_id",
    "cohort",
    "source_file",
    "month",
    "count_min",
    "count_max",
    "count_estimate",
    "confidence",
    "source_section",
    "method",
    "evidence",
]

SUMMARY_FIELDS = [
    "patient_id",
    "cohort",
    "source_file",
    "eligible_rows",
    "reviewed_rows",
    "skipped_rows",
    "remaining_low_medium_rows",
    "corrected_monthly_rows",
    "timeline_csv",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row):
    parts = [
        row.get("patient_id", ""),
        row.get("cohort", ""),
        row.get("source_file", ""),
        row.get("month", ""),
        row.get("method", ""),
        row.get("evidence", ""),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def is_eligible(row):
    return row.get("confidence", "").strip().lower() in LOW_CONFIDENCE_VALUES


def load_corrections(path):
    if not os.path.exists(path):
        return []
    return read_csv(path)


def correction_map(corrections):
    return {row["row_key"]: row for row in corrections if row.get("row_key")}


def format_row_for_prompt(row, index, total):
    lines = [
        "",
        "=" * 88,
        f"Review {index}/{total}",
        f"Patient: {row.get('patient_id', '')}    Cohort: {row.get('cohort', '')}    Month: {row.get('month', '')}",
        f"Source: {row.get('source_file', '')}",
        f"Confidence: {row.get('confidence', '')}    Method: {row.get('method', '')}",
        (
            "Parser count: "
            f"min={row.get('count_min', '')}, "
            f"max={row.get('count_max', '')}, "
            f"estimate={row.get('count_estimate', '')}"
        ),
        "-" * 88,
        row.get("evidence", ""),
        "=" * 88,
    ]
    return "\n".join(lines)


def parse_human_count(value):
    value = value.strip()
    if not value:
        return None
    count = float(value)
    if count < 0:
        raise ValueError("count must be non-negative")
    return count


def correction_from_input(row, status, human_count):
    return {
        "row_key": row_key(row),
        "review_status": status,
        "human_count": "" if human_count is None else format_number(human_count),
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "patient_id": row.get("patient_id", ""),
        "cohort": row.get("cohort", ""),
        "source_file": row.get("source_file", ""),
        "month": row.get("month", ""),
        "original_count_min": row.get("count_min", ""),
        "original_count_max": row.get("count_max", ""),
        "original_count_estimate": row.get("count_estimate", ""),
        "original_confidence": row.get("confidence", ""),
        "source_section": row.get("source_section", ""),
        "method": row.get("method", ""),
        "evidence": row.get("evidence", ""),
    }


def format_number(value):
    value = round(float(value), 6)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def review_rows(evidence_rows, existing_corrections, limit=None):
    reviewed_keys = set(correction_map(existing_corrections))
    eligible = [row for row in evidence_rows if is_eligible(row)]
    pending = [row for row in eligible if row_key(row) not in reviewed_keys]
    if limit is not None:
        pending = pending[:limit]

    new_corrections = []
    for index, row in enumerate(pending, start=1):
        print(format_row_for_prompt(row, index, len(pending)))
        while True:
            answer = input("Actual seizure count [number / blank=skip / q=save+quit]: ").strip()
            if answer.lower() == "q":
                return new_corrections, True
            if answer == "":
                new_corrections.append(correction_from_input(row, "skipped", None))
                break
            try:
                human_count = parse_human_count(answer)
            except ValueError:
                print("Invalid input. Enter a non-negative number, blank to skip, or q to save+quit.")
                continue
            new_corrections.append(correction_from_input(row, "reviewed", human_count))
            break
    return new_corrections, False


def apply_corrections(evidence_rows, corrections):
    corrections_by_key = correction_map(corrections)
    corrected = []
    for row in evidence_rows:
        out = {field: row.get(field, "") for field in EVIDENCE_FIELDS}
        correction = corrections_by_key.get(row_key(row))
        if correction and correction.get("review_status") == "reviewed":
            human_count = correction.get("human_count", "")
            out["count_min"] = human_count
            out["count_max"] = human_count
            out["count_estimate"] = human_count
            out["confidence"] = "human_reviewed"
        corrected.append(out)
    return corrected


def evidence_objects(rows):
    objects = []
    for row in rows:
        try:
            count_min = float(row.get("count_min", "") or 0)
            count_max = float(row.get("count_max", "") or 0)
            count_estimate = float(row.get("count_estimate", "") or 0)
        except ValueError:
            continue
        objects.append(
            Evidence(
                patient_id=row.get("patient_id", ""),
                cohort=row.get("cohort", ""),
                source_file=row.get("source_file", ""),
                month=row.get("month", ""),
                count_min=count_min,
                count_max=count_max,
                count_estimate=count_estimate,
                confidence=row.get("confidence", ""),
                source_section=row.get("source_section", ""),
                method=row.get("method", ""),
                evidence=row.get("evidence", ""),
            )
        )
    return objects


def write_corrected_timelines(corrected_evidence_rows, output_dir):
    monthly_rows = aggregate_evidence(evidence_objects(corrected_evidence_rows))
    timeline_dir = os.path.join(output_dir, "reviewed_seizure_timeline")
    grouped = defaultdict(list)
    for row in monthly_rows:
        grouped[row["source_file"]].append(row)

    fieldnames = [
        "patient_id",
        "cohort",
        "source_file",
        "month",
        "seizure_count_min",
        "seizure_count_max",
        "seizure_count_estimate",
        "confidence",
        "evidence_count",
        "methods",
        "evidence",
    ]
    paths = {}
    for source_file, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["month"])
        first = rows[0]
        filename = f"{safe_filename(first['patient_id'])}{safe_filename(first['cohort'])}_seizure_timeline.csv"
        out_path = os.path.join(timeline_dir, filename)
        write_csv(out_path, rows, fieldnames)
        paths[source_file] = out_path
    return monthly_rows, paths


def build_summary(evidence_rows, corrections, monthly_rows, patient_paths):
    corrections_by_key = correction_map(corrections)
    by_source = defaultdict(
        lambda: {
            "patient_id": "",
            "cohort": "",
            "source_file": "",
            "eligible_rows": 0,
            "reviewed_rows": 0,
            "skipped_rows": 0,
            "remaining_low_medium_rows": 0,
            "corrected_monthly_rows": 0,
            "timeline_csv": "",
        }
    )

    for row in evidence_rows:
        source = row.get("source_file", "")
        entry = by_source[source]
        entry["patient_id"] = row.get("patient_id", "")
        entry["cohort"] = row.get("cohort", "")
        entry["source_file"] = source
        if is_eligible(row):
            entry["eligible_rows"] += 1
            correction = corrections_by_key.get(row_key(row))
            if not correction:
                entry["remaining_low_medium_rows"] += 1
            elif correction.get("review_status") == "reviewed":
                entry["reviewed_rows"] += 1
            elif correction.get("review_status") == "skipped":
                entry["skipped_rows"] += 1

    for row in monthly_rows:
        source = row.get("source_file", "")
        entry = by_source[source]
        entry["patient_id"] = row.get("patient_id", "")
        entry["cohort"] = row.get("cohort", "")
        entry["source_file"] = source
        entry["corrected_monthly_rows"] += 1
        entry["timeline_csv"] = patient_paths.get(source, "")

    return sorted(by_source.values(), key=lambda row: (row["cohort"], row["patient_id"], row["source_file"]))


def print_summary(eligible_count, corrections, new_count, quit_requested, output_paths):
    reviewed_count = sum(1 for row in corrections if row.get("review_status") == "reviewed")
    skipped_count = sum(1 for row in corrections if row.get("review_status") == "skipped")
    remaining = eligible_count - reviewed_count - skipped_count

    print("")
    print("Review summary")
    print("-" * 80)
    print(f"Total eligible rows: {eligible_count}")
    print(f"Reviewed rows: {reviewed_count}")
    print(f"Skipped rows: {skipped_count}")
    print(f"New rows handled this run: {new_count}")
    print(f"Remaining unreviewed rows: {max(remaining, 0)}")
    print(f"Quit requested: {'yes' if quit_requested else 'no'}")
    print("")
    for label, path in output_paths.items():
        print(f"{label}: {path}")


def main():
    parser = argparse.ArgumentParser(description="Interactively review low/medium-confidence seizure extractions.")
    parser.add_argument("--evidence-file", default=DEFAULT_EVIDENCE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Review at most N pending rows in this run.")
    args = parser.parse_args()

    if not os.path.exists(args.evidence_file):
        raise SystemExit(f"Evidence CSV not found: {args.evidence_file}")

    evidence_rows = read_csv(args.evidence_file)
    correction_path = os.path.join(args.output_dir, "reviewed_evidence_corrections.csv")
    corrected_evidence_path = os.path.join(args.output_dir, "reviewed_seizure_extraction_evidence.csv")
    summary_path = os.path.join(args.output_dir, "review_summary.csv")

    corrections = load_corrections(correction_path)
    new_corrections, quit_requested = review_rows(evidence_rows, corrections, args.limit)
    all_corrections = corrections + new_corrections

    write_csv(correction_path, all_corrections, CORRECTION_FIELDS)
    corrected_evidence = apply_corrections(evidence_rows, all_corrections)
    write_csv(corrected_evidence_path, corrected_evidence, EVIDENCE_FIELDS)
    monthly_rows, patient_paths = write_corrected_timelines(corrected_evidence, args.output_dir)
    summary = build_summary(evidence_rows, all_corrections, monthly_rows, patient_paths)
    write_csv(summary_path, summary, SUMMARY_FIELDS)

    output_paths = {
        "Corrections": correction_path,
        "Corrected evidence": corrected_evidence_path,
        "Corrected timelines": os.path.join(args.output_dir, "reviewed_seizure_timeline"),
        "Review summary": summary_path,
    }
    eligible_count = sum(1 for row in evidence_rows if is_eligible(row))
    print_summary(eligible_count, all_corrections, len(new_corrections), quit_requested, output_paths)


if __name__ == "__main__":
    main()
