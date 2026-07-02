import argparse
import re
import sys
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename

from docx import Document


BRAIN_IMAGE_TABLE_TITLE = "Brain Image"


def safe_print(text=""):
    encoding = sys.stdout.encoding or "utf-8"
    print(str(text).encode(encoding, errors="backslashreplace").decode(encoding))


def normalize_text(text):
    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
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


def find_brain_image_table(doc):
    for table in doc.tables:
        if not table.rows:
            continue
        first_row = get_unique_row_values(table.rows[0])
        if first_row and first_row[0] == BRAIN_IMAGE_TABLE_TITLE:
            return table
    return None


def extract_brain_image_table(table):
    rows = []
    for row in table.rows:
        row_data = get_unique_row_values(row)
        if not row_data:
            continue
        if not any(value.strip() for value in row_data):
            continue
        rows.append(row_data)
    return rows


def table_selection(file_path):
    doc = Document(file_path)
    table = find_brain_image_table(doc)
    if table is None:
        return []
    return extract_brain_image_table(table)


def print_brain_image_rows(file_path, rows):
    safe_print(f"\n{file_path}")
    if not rows:
        safe_print("No Brain Image table found.")
        return

    for row in rows:
        safe_print(" | ".join(row))


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


def process_paths(paths):
    files = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            files.extend(collect_docx_files(path_obj))
        elif path_obj.suffix.lower() == ".docx":
            files.append(path_obj)

    if not files:
        safe_print("No .docx files found.")
        return {
            "files": 0,
            "with_brain_image_table": 0,
            "errors": 0,
        }

    with_brain_image_table = 0
    errors = []
    for file_path in files:
        try:
            rows = table_selection(file_path)
        except Exception as exc:
            errors.append((str(file_path), str(exc)))
            safe_print(f"ERROR {file_path}: {exc}")
            continue

        if rows:
            with_brain_image_table += 1
        print_brain_image_rows(file_path, rows)

    safe_print(
        f"\nDone: {with_brain_image_table}/{len(files)} file(s) had a Brain Image table; "
        f"{len(errors)} errors."
    )
    if errors:
        safe_print("Errors:")
        for file_path, error in errors:
            safe_print(f"  {file_path}: {error}")

    return {
        "files": len(files),
        "with_brain_image_table": with_brain_image_table,
        "errors": len(errors),
    }


def choose_input_path():
    root = Tk()
    root.withdraw()

    file_path = askopenfilename(
        title="Select Patient DOCX File (Cancel to Select a Folder)",
        filetypes=[("DOCX files", "*.docx"), ("All files", "*.*")],
    )
    if file_path:
        root.destroy()
        return file_path

    folder_path = askdirectory(
        title="Select Folder Containing Patient DOCX Files (Cancel to Exit)"
    )
    root.destroy()
    return folder_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract and print the complete Brain Image table contents from DOCX files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="DOCX files or folders containing DOCX files. If omitted, a selector opens.",
    )
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        selected_path = choose_input_path()
        if not selected_path:
            safe_print("No file or folder selected.")
            return
        paths = [selected_path]

    process_paths(paths)


if __name__ == "__main__":
    main()
