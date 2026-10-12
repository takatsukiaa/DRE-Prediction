import argparse
import os
import re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory

import pandas as pd
from docx import Document


DEFAULT_INPUT_DIRS = ("DRE", "NDRE")
DEFAULT_OUTPUT_DIR = "./new_csvs/cognitive_timeline"
MENTAL_TABLE_TITLE = "Mental Retardation"

TEST_COLUMNS = {
    "MMSE": ["MMSE"] + [f"MMSE_Q{i}" for i in range(1, 12)],
    "CASI": ["CASI"] + [f"CASI_Q{i}" for i in range(1, 10)],
    "WAIS": ["WAIS"] + [f"WAIS_Q{i}" for i in range(1, 15)],
}
OUTPUT_COLUMNS = (
    TEST_COLUMNS["MMSE"]
    + TEST_COLUMNS["CASI"]
    + TEST_COLUMNS["WAIS"]
)


def normalize_text(text):
    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_cell_text(cell):
    clean_text = normalize_text(cell.text.replace("\n", " "))
    all_text_nodes = cell._tc.xpath(".//w:t")
    xpath_text = normalize_text("".join(node.text for node in all_text_nodes if node.text))
    if len(xpath_text) > len(clean_text):
        return xpath_text
    return clean_text


def get_unique_row_values(row):
    row_data = []
    visited_cells = set()
    for cell in row.cells:
        if cell._tc in visited_cells:
            continue
        visited_cells.add(cell._tc)
        row_data.append(get_cell_text(cell))
    return row_data


def identify_test_type(row_data):
    joined = " ".join(row_data).upper()
    for test_name in TEST_COLUMNS:
        if test_name in joined:
            return test_name
    return None


def is_empty_value(value):
    text = normalize_text(value)
    return text == "" or text in {"無", "NA", "N/A", "nan", "None"}


def parse_date(value):
    text = normalize_text(value)
    if is_empty_value(text):
        return pd.NaT

    compact = re.fullmatch(r"\d{8}", text)
    if compact:
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")

    return pd.to_datetime(text, errors="coerce")


def normalize_score(value):
    text = normalize_text(value)
    if is_empty_value(text):
        return pd.NA
    return text


def patient_id_from_path(file_path):
    match = re.search(r"\d+", Path(file_path).stem)
    return match.group(0) if match else Path(file_path).stem


def cohort_from_path(file_path):
    path_text = str(file_path).upper()
    if "NDRE" in path_text:
        return "NDRE"
    if "DRE" in path_text:
        return "DRE"
    return ""


def find_mental_retardation_table(doc):
    for table in doc.tables:
        if not table.rows:
            continue
        first_row = get_unique_row_values(table.rows[0])
        if first_row and first_row[0] == MENTAL_TABLE_TITLE:
            return table
    return None


def extract_cognitive_timeline_from_table(table):
    records_by_date = {}
    current_test = None

    for row in table.rows:
        row_data = get_unique_row_values(row)
        if not row_data:
            continue

        first_cell = normalize_text(row_data[0])
        if first_cell == MENTAL_TABLE_TITLE:
            continue

        if first_cell == "Date":
            current_test = identify_test_type(row_data)
            continue

        if current_test is None:
            continue

        date = parse_date(first_cell)
        if pd.isna(date):
            continue

        date_key = date.normalize()
        row_record = records_by_date.setdefault(
            date_key,
            {column: pd.NA for column in OUTPUT_COLUMNS},
        )
        section_columns = TEST_COLUMNS[current_test]
        values = row_data[1:]
        for column, value in zip(section_columns, values):
            row_record[column] = normalize_score(value)

    if not records_by_date:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame.from_dict(records_by_date, orient="index")
    df.index.name = "Date"
    df = df.reindex(columns=OUTPUT_COLUMNS)
    df.sort_index(inplace=True)
    df.dropna(how="all", inplace=True)
    return df


def table_selection(file_path):
    doc = Document(file_path)
    table = find_mental_retardation_table(doc)
    if table is None:
        return None
    return extract_cognitive_timeline_from_table(table)


def extract_cognitive_timeline_from_docx(file_path):
    return table_selection(file_path)


def save_cognitive_output(file_path, output_dir=DEFAULT_OUTPUT_DIR):
    df = table_selection(file_path)
    if df is None or df.empty:
        return None

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.basename(file_path)
    base_name = fname[:-5] if fname.lower().endswith(".docx") else fname
    output_file = os.path.join(output_dir, f"{base_name}_cognitive_timeline.csv")
    df.to_csv(output_file, index=True, encoding="utf-8-sig")
    print(f"Saved to {output_file}")
    return output_file


def collect_docx_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        folder = Path(input_dir)
        if not folder.exists():
            continue
        files.extend(
            sorted(
                path
                for path in folder.glob("*.docx")
                if not path.name.startswith("~$")
            )
        )
    return files


def build_combined_rows(saved_files):
    combined_frames = []
    for source_docx, csv_path in saved_files:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if df.empty:
            continue
        df.insert(0, "source_file", str(source_docx))
        df.insert(0, "cohort", cohort_from_path(source_docx))
        df.insert(0, "patient_id", patient_id_from_path(source_docx))
        combined_frames.append(df)
    if not combined_frames:
        return pd.DataFrame()
    return pd.concat(combined_frames, ignore_index=True)


def process_directories(input_dirs=DEFAULT_INPUT_DIRS, output_dir=DEFAULT_OUTPUT_DIR):
    files = collect_docx_files(input_dirs)
    if not files:
        print("No .docx files found.")
        return {
            "files": 0,
            "saved_files": 0,
            "skipped_files": 0,
            "errors": 0,
            "output_dir": output_dir,
            "combined_path": "",
        }

    print(f"Processing {len(files)} file(s)")
    saved, skipped, errors = [], [], []
    for file_path in files:
        try:
            output_file = save_cognitive_output(file_path, output_dir)
        except Exception as exc:
            errors.append((str(file_path), str(exc)))
            print(f"ERROR {file_path.name}: {exc}")
            continue

        if output_file:
            saved.append((file_path, output_file))
            print(f"OK {file_path.name}: saved {output_file}")
        else:
            skipped.append(str(file_path))
            print(f"SKIP {file_path.name}: no cognitive scores found")

    combined_path = os.path.join(output_dir, "cognitive_all_records.csv")
    combined_df = build_combined_rows(saved)
    if not combined_df.empty:
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"Saved combined records to {combined_path}")
    else:
        combined_path = ""

    print(f"\nDone: {len(saved)} saved, {len(skipped)} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")

    return {
        "files": len(files),
        "saved_files": len(saved),
        "skipped_files": len(skipped),
        "errors": len(errors),
        "output_dir": output_dir,
        "combined_path": combined_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract MMSE, CASI, and WAIS scores from Mental Retardation DOCX tables."
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        default=None,
        help="Directories containing patient .docx files. If omitted, a folder picker opens.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cognitive timeline CSV outputs.",
    )
    args = parser.parse_args()

    input_dirs = args.input_dirs
    if not input_dirs:
        root = Tk()
        root.withdraw()
        selected_dir = askdirectory(
            title="Select Folder Containing Patient DOCX Files (Cancel to Exit)"
        )
        root.destroy()
        if not selected_dir:
            print("No folder selected.")
            return
        input_dirs = [selected_dir]

    result = process_directories(input_dirs, args.output_dir)
    print(
        "Processed {files} DOCX files; saved {saved_files} cognitive timeline CSVs; "
        "skipped {skipped_files}; errors {errors}.".format(**result)
    )
    print(f"Output directory: {result['output_dir']}")
    if result["combined_path"]:
        print(f"Combined records: {result['combined_path']}")


if __name__ == "__main__":
    main()
