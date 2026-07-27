import os

import pandas as pd
from docx import Document
import re
from tkinter.filedialog import askdirectory
from tkinter import Tk
import numpy as np
from difflib import get_close_matches


def normalize_row(row_data, width):
    if len(row_data) < width:
        row_data.extend([np.nan] * (width - len(row_data)))
    return row_data[:width]


LAB_DATE_PATTERN = re.compile(
    r"^\s*(?P<year>\d{4})(?:(?P<compact_month>\d{2})(?P<compact_day>\d{2})"
    r"|[-/](?P<separated_month>\d{1,2})[-/](?P<separated_day>\d{1,2}))(?!\d)"
)


def parse_lab_date(value):
    if pd.isna(value):
        return pd.NaT

    match = LAB_DATE_PATTERN.match(str(value))
    if not match:
        return pd.NaT

    month = match.group("compact_month") or match.group("separated_month")
    day = match.group("compact_day") or match.group("separated_day")
    try:
        return pd.Timestamp(year=int(match.group("year")), month=int(month), day=int(day))
    except ValueError:
        return pd.NaT


def prepare_date_index(df):
    df = df.copy()
    df['Date'] = df['Date'].map(parse_lab_date)
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    return df


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-"}:
        return np.nan
    text = re.sub(r"^[<>]=?", "", text)
    text = re.sub(r"(\d+)\.\.+(\d+)", r"\1.\2", text)
    return pd.to_numeric(text, errors='coerce')


def coerce_numeric_frame(df):
    return df.map(clean_numeric_value)


GENERAL_REPORT = "\u4e00\u822c\u751f\u5316\u6aa2\u9a57\u5831\u544a"
ROUTINE_REPORT = "\u5e38\u898f\u751f\u5316\u6aa2\u9a57\u5831\u544a"
ANTIGEN_REPORT = "\u6297\u539f\u6aa2\u9a57\u5831\u544a"
IMMUNE_REPORT = "\u514d\u75ab\u751f\u5316\u6aa2\u9a57\u5831\u544a"
DRUG_REPORT = "\u85e5\u7269\u6fc3\u5ea6\uff08\u7279\u6b8a\u751f\u5316\u6aa2\u9a57\u5831\u544a\uff09"
SECTION_TITLES = {
    GENERAL_REPORT,
    ROUTINE_REPORT,
    ANTIGEN_REPORT,
    IMMUNE_REPORT,
    DRUG_REPORT,
}


def get_cell_text(cell):
    """
    Robustly extracts text from a docx cell, matching the logic 
    used in docx_parser_static.py.
    """
    clean_text = cell.text.strip().replace('\n', ' ')
    all_text_nodes = cell._tc.xpath('.//w:t')
    xpath_text = "".join([node.text for node in all_text_nodes if node.text]).strip()
    if len(xpath_text) > len(clean_text):
        return xpath_text
    return clean_text


def extract_blood_test_timeline_legacy(target_table):
    header_row_idx = -1
    visited_cells = set()
    for i, row in enumerate(target_table.rows):
        first_cell_text = get_cell_text(row.cells[0])
        if first_cell_text == "Date":
            header_row_idx = i
            break
    if header_row_idx == -1:
        print("Could not find the 'Date' header row in the ASM table.")
        exit(1)
    headers = []
    header_row = target_table.rows[header_row_idx]
    for cell in header_row.cells:
        if cell._tc not in visited_cells:
            visited_cells.add(cell._tc)
            headers.append(get_cell_text(cell))
        else:
            continue

    data = []
    splitting_point = []
    i = 0
    # df_routine = None
    # df_norm = None
    # df_drug = None
    for row in target_table.rows[header_row_idx + 1:]:
        row_data = []
        j = 0
        for cell in row.cells:
            if cell._tc not in visited_cells:
                visited_cells.add(cell._tc)
                text = get_cell_text(cell)
                if "常規生化檢驗報告" in text or "抗原檢驗報告" in text or "免疫生化檢驗報告" in text or "藥物濃度（特殊生化檢驗報告）" in text:
                    splitting_point.append(i)
                    row_data.append(text)
                    continue
                if text == '':
                    row_data.append(np.nan)
                    continue
                row_data.append(text)
                j += 1
        # Skip rows that are entirely empty or don't have a date
        if not any(row_data) or not row_data[0]:
            continue
        row_data = normalize_row(row_data, len(headers))
        data.append(row_data)
        i += 1

    df_raw = pd.DataFrame(data, columns=headers)
    df_1 = df_raw.iloc[:splitting_point[0], :]
    df_1 = prepare_date_index(df_1)
    df_2 = df_raw.iloc[splitting_point[0] + 1:splitting_point[1], :]
    df_2.reset_index(drop=True, inplace=True)
    df_2.columns = df_2.iloc[0]
    df_2 = df_2[1:]
    df_2.reset_index(drop=True, inplace=True)
    df_2 = prepare_date_index(df_2)
    df_3 = df_raw.iloc[splitting_point[3] + 1:, :]
    df_3.reset_index(drop=True, inplace=True)
    df_3.columns = df_3.iloc[0]
    df_3 = df_3[1:]
    df_3.reset_index(drop=True, inplace=True)
    df_3 = prepare_date_index(df_3)

    df_1 = coerce_numeric_frame(df_1)
    # df_2 = df_2.astype(float)
    df_3 = coerce_numeric_frame(df_3)
    df_1 = df_1.resample('ME').mean()
    df_3 = df_3.resample('ME').mean()
    # df_2 = df_2.resample('ME').sum()

    df_1.dropna(how='all', axis=0, inplace=True)
    df_1.dropna(how='all', axis=1, inplace=True)

    df_2.dropna(how='all', axis=0, inplace=True)
    df_2.dropna(how='all', axis=1, inplace=True)

    df_3.dropna(how='all', axis=0, inplace=True)
    df_3.dropna(how='all', axis=1, inplace=True)
    # df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
    # df_raw = df_raw.set_index('Date')
    # df_raw = df_raw.resample('ME').mean()
    # df_raw.dropna(how="all", axis=0, inplace=True)
    return df_1, df_2, df_3


def get_unique_row_values(row):
    row_data = []
    visited_cells = set()
    for cell in row.cells:
        if cell._tc not in visited_cells:
            visited_cells.add(cell._tc)
            text = get_cell_text(cell)
            row_data.append(np.nan if text == '' else text)
    return row_data


def build_section_df(headers, rows):
    if not headers or not rows:
        return None
    normalized_rows = [normalize_row(row, len(headers)) for row in rows]
    df = pd.DataFrame(normalized_rows, columns=headers)
    df = prepare_date_index(df)
    df.dropna(how='all', axis=0, inplace=True)
    return df


def extract_lab_sections(target_table):
    sections = {}
    current_section = None
    current_headers = None
    current_rows = []

    def flush_section():
        nonlocal current_headers, current_rows
        df = build_section_df(current_headers, current_rows)
        if current_section and df is not None:
            sections[current_section] = df
        current_headers = None
        current_rows = []

    for row in target_table.rows:
        row_data = get_unique_row_values(row)
        if not row_data or pd.isna(row_data[0]):
            continue

        first_cell_text = str(row_data[0]).strip()
        if first_cell_text in SECTION_TITLES:
            flush_section()
            current_section = first_cell_text
            continue

        if current_section and first_cell_text == "Date":
            current_headers = row_data
            current_rows = []
            continue

        if current_section and current_headers:
            current_rows.append(row_data)

    flush_section()
    return sections


def resample_numeric_section(df):
    if df is None:
        return None
    df = coerce_numeric_frame(df)
    df = df.resample('ME').mean()
    df.dropna(how='all', axis=0, inplace=True)
    return df


def extract_blood_test_timeline(target_table):
    sections = extract_lab_sections(target_table)

    df_norm = resample_numeric_section(sections.get(GENERAL_REPORT))
    df_routine = resample_numeric_section(sections.get(ROUTINE_REPORT))
    df_drug = resample_numeric_section(sections.get(DRUG_REPORT))

    return df_norm, df_routine, df_drug


def table_selection(filepath):
    try:
        doc = Document(filepath)
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None
    for table in doc.tables:
        if table.rows:
            # Check the first cell's text to identify the table
            first_cell_text = get_cell_text(table.rows[0].cells[0])
            DISPATCH_MAP = {
                "Lab": extract_blood_test_timeline,
            }
        handler = DISPATCH_MAP.get(first_cell_text)
        if handler:
            df_1, df_2, df_3 = handler(table)
            return df_1, df_2, df_3
        else:
            continue


def extract_lab_timeline_from_docx(filepath):
    return table_selection(filepath)


def save_blood_test_outputs(file_path, output_dir="./new_csvs/lab_timeline"):
    result = table_selection(file_path)
    if result is None:
        return []

    df_norm, df_routine, df_drug = result
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.basename(file_path)
    base_name = fname[:-5] if fname.lower().endswith(".docx") else fname
    saved = []

    if df_norm is not None:
        print(df_norm.columns.tolist())
        out_path = os.path.join(output_dir, f"{base_name}_norm_blood_timeline.csv")
        df_norm.to_csv(out_path, index=True, encoding="utf-8-sig")
        saved.append(out_path)
        print(f"Saved to {out_path}")

    if df_routine is not None:
        print(df_routine.columns.tolist())
        out_path = os.path.join(output_dir, f"{base_name}_routine_blood_timeline.csv")
        df_routine.to_csv(out_path, index=True, encoding="utf-8-sig")
        saved.append(out_path)
        print(f"Saved to {out_path}")

    if df_drug is not None:
        print(df_drug.columns.tolist())
        out_path = os.path.join(output_dir, f"{base_name}_drug_blood_timeline.csv")
        df_drug.to_csv(out_path, index=True, encoding="utf-8-sig")
        saved.append(out_path)
        print(f"Saved to {out_path}")

    return saved


def process_directory(input_dir, output_dir="./new_csvs/lab_timeline"):
    docx_files = [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(".docx") and not fname.startswith("~$")
    ]

    if not docx_files:
        print(f"No .docx files found in {input_dir}")
        return

    print(f"Processing {len(docx_files)} file(s) from {input_dir}")
    success, skipped, errors = [], [], []

    for file_path in sorted(docx_files):
        fname = os.path.basename(file_path)
        try:
            saved = save_blood_test_outputs(file_path, output_dir)
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"ERROR {fname}: {e}")
            continue

        if saved:
            success.append((fname, saved))
            print(f"OK {fname}: saved {len(saved)} file(s)")
        else:
            skipped.append(fname)
            print(f"SKIP {fname}: no Lab table found")

    print(f"\nDone: {len(success)} succeeded, {len(skipped)} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for fname, err in errors:
            print(f"  {fname}: {err}")


# def process_directory(input_dir, output_dir=None):
#     import glob

#     if output_dir is None:
#         output_dir = "./new_csvs"
#     os.makedirs(output_dir, exist_ok=True)

#     docx_files = glob.glob(os.path.join(input_dir, "*.docx"))
#     if not docx_files:
#         print(f"No .docx files found in {input_dir}")
#         return

#     print(f"Processing {len(docx_files)} file(s) from {input_dir}")
#     success, skipped, errors = [], [], []

#     for filepath in sorted(docx_files):
#         fname = os.path.basename(filepath)
#         try:
#             df = table_selection(filepath)
#         except Exception as e:
#             errors.append((fname, str(e)))
#             print(f"  ERROR {fname}: {e}")
#             continue

#         if df is None or df.empty:
#             skipped.append(fname)
#             print(f"  SKIP  {fname}: no ASM table found")
#             continue

#         out_path = os.path.join(output_dir, fname.replace(".docx", "_asm_timeline.csv"))
#         df.to_csv(out_path, index=True, encoding="utf-8-sig")
#         success.append((fname, list(df.columns)))
#         print(f"  OK    {fname} -> {out_path}")

#     print(f"\nDone: {len(success)} succeeded, {len(skipped)} skipped, {len(errors)} errors")
#     if errors:
#         print("Errors:")
#         for fname, err in errors:
#             print(f"  {fname}: {err}")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    input_dir = askdirectory(
        title="Select Folder Containing Patient Docx Files (Cancel to Exit)"
    )
    if not input_dir:
        print("No folder selected.")
    else:
        process_directory(input_dir)
