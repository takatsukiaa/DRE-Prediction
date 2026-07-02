import argparse
import csv
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, asksaveasfilename

from docx_parser_static import DEFAULT_PARSED_DATA, initial_table_selection


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "new_csvs" / "static_data"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "static_data_batch_table.csv"


def safe_print(text=""):
    encoding = sys.stdout.encoding or "utf-8"
    print(str(text).encode(encoding, errors="backslashreplace").decode(encoding))


def collect_docx_files(input_dir):
    folder = Path(input_dir)
    if folder.is_file() and folder.suffix.lower() == ".docx":
        return [folder]
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob("*.docx")
        if not path.name.startswith("~$")
    )


def parse_static_data(file_path, verbose=False):
    if verbose:
        return initial_table_selection(file_path)

    # The individual parser prints each field. Suppress that here so batch output
    # stays readable while still reusing the same extraction logic.
    with redirect_stdout(io.StringIO()):
        return initial_table_selection(file_path)


def build_batch_rows(docx_files, verbose=False):
    rows = []
    errors = []

    for index, file_path in enumerate(docx_files, start=1):
        safe_print(f"Processing {index}/{len(docx_files)}: {file_path.name}")
        try:
            data = parse_static_data(str(file_path), verbose=verbose)
        except Exception as exc:
            errors.append((file_path.name, str(exc)))
            row = {
                **DEFAULT_PARSED_DATA,
                "parse_error": str(exc),
            }
            rows.append(row)
            safe_print(f"ERROR {file_path.name}: {exc}")
            continue

        row = {
            **{field: data.get(field, "") for field in DEFAULT_PARSED_DATA},
            "parse_error": "",
        }
        rows.append(row)

    return rows, errors


def save_batch_table(rows, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        *DEFAULT_PARSED_DATA.keys(),
        "parse_error",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def create_batch_static_table(input_dir, output_file=DEFAULT_OUTPUT_FILE, verbose=False):
    docx_files = collect_docx_files(input_dir)
    if not docx_files:
        safe_print(f"No .docx files found in {input_dir}")
        return {
            "files": 0,
            "rows": 0,
            "errors": 0,
            "output_file": "",
        }

    rows, errors = build_batch_rows(docx_files, verbose=verbose)
    output_path = save_batch_table(rows, output_file)

    safe_print(f"\nSaved batch static-data table to {output_path}")
    safe_print(f"Done: {len(rows)} row(s), {len(errors)} error(s).")
    if errors:
        safe_print("Errors:")
        for fname, err in errors:
            safe_print(f"  {fname}: {err}")

    return {
        "files": len(docx_files),
        "rows": len(rows),
        "errors": len(errors),
        "output_file": str(output_path),
    }


def choose_input_dir():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    input_dir = askdirectory(
        title="Select Folder Containing Patient DOCX Files"
    )
    root.destroy()
    return input_dir


def choose_output_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    output_file = asksaveasfilename(
        title="Save Batch Static-Data Table",
        defaultextension=".csv",
        initialdir=str(DEFAULT_OUTPUT_DIR),
        initialfile=DEFAULT_OUTPUT_FILE.name,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Create one combined static-data CSV table from DOCX files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Folder containing patient DOCX files. If omitted, a folder selector opens.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT_FILE}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each individual parser field while processing.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir or choose_input_dir()
    if not input_dir:
        safe_print("No folder selected.")
        return

    output_file = args.output
    if not output_file and not args.input_dir:
        output_file = choose_output_file()
        if not output_file:
            safe_print("No output file selected.")
            return

    create_batch_static_table(
        input_dir=input_dir,
        output_file=output_file or DEFAULT_OUTPUT_FILE,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
