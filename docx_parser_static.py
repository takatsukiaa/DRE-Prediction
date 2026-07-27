import os
import re
import sys
import io
from contextlib import redirect_stdout
from tkinter import Tk

from docx import Document
from tkinter.filedialog import askdirectory
import csv

DEFAULT_PARSED_DATA = {
    "Patient ID": "",
    "Birthday": "",
    "Gender": "",
    "1st Seizure Age": "",
    "Drug-refractory epilepsy": "",
    "Systemic disease": "",
    "Adverse of ASM": "",
    "Etiology": "",
    "EEG Focality": "",
    "Frequency (n/month)": "",
    "Focal": "",
    "Unknown whether focal or generalized (U)": "",
    "Generalized (G)": "",
    "Unclassified": "",
    "Elementary motor (Motor)": "",
    "Complex  hypermotor (automatism)":"",
    "Sensory":"",
    "Cognitive and language": "",
    "Autonomic (e.g. palpitation, sweating, hot flush, epigastric or abdominal raising)": "",
    "Affective (emotional)": "",
    "Aura (Indescribable)": "",
    "Descriptor": "",
    "Asymptomatic IED (EEG IED)": "",
    "Provoking factor": "",
    "Dx": "",
    "Mental Retardation": "",
    # other fields
}

parsed_data = DEFAULT_PARSED_DATA.copy()

EEG_ELECTRODES = [
    "Fp1", "F3", "F7", "Fp2", "F4", "F8", "Fz",
    "T3", "T4", "T5", "T6", "S1", "S2",
    "C3", "C4", "Cz", "P3", "P4", "Pz", "O1", "O2",
]
EEG_ELECTRODE_BY_CASEFOLD = {electrode.casefold(): electrode for electrode in EEG_ELECTRODES}
EEG_ELECTRODE_PATTERN = re.compile(
    "|".join(re.escape(electrode) for electrode in sorted(EEG_ELECTRODES, key=len, reverse=True)),
    flags=re.IGNORECASE,
)

def reset_parsed_data():
    parsed_data.clear()
    parsed_data.update(DEFAULT_PARSED_DATA.copy())

def safe_print(text=""):
    encoding = sys.stdout.encoding or "utf-8"
    print(str(text).encode(encoding, errors="backslashreplace").decode(encoding))

def baseline_parsing(table):
    flat_cells = []
    visited_cells = set()
    for row in table.rows:
        for cell in row.cells:
            if cell._tc not in visited_cells:
                visited_cells.add(cell._tc)
                clean_text = cell.text.strip().replace('\n', ' ')
                all_text_nodes = cell._tc.xpath('.//w:t')
                xpath_text = "".join([node.text for node in all_text_nodes if node.text]).strip()
                if len(xpath_text) > len(clean_text):
                    parsed_text = xpath_text
                else:
                    parsed_text = clean_text
                flat_cells.append(parsed_text)
            else:
                continue

    for i, content in enumerate(flat_cells):
        for target in parsed_data:
            if target in content:
                # 假設 Value 就在下一個儲存格
                if i + 1 < len(flat_cells):
                    if target == "Patient ID":
                        flat_cells[i + 1] = re.sub(r'\D', '', flat_cells[i + 1])
                    parsed_data[target] = flat_cells[i + 1]
            else:
                continue

def adverse_parsing(table):
    flat_cells = []
    visited_cells = set()
    for row in table.rows:
        cells = row.cells
        for i, cell in enumerate(cells):
            if cell._tc not in visited_cells:
                visited_cells.add(cell._tc)
                clean_text = cell.text.strip().replace('\n', ' ')
                all_text_nodes = cell._tc.xpath('.//w:t')
                xpath_text = "".join([node.text for node in all_text_nodes if node.text]).strip()
                if len(xpath_text) > len(clean_text):
                    parsed_text = xpath_text
                else:
                    parsed_text = clean_text
                if parsed_text == "Yes":
                    if cells[i+1].text == "0":
                        parsed_data["Adverse of ASM"] = "0"
                        return
                    else:
                        flat_cells.append(parsed_text)
                        continue
                flat_cells.append(parsed_text)
            else:
                continue
    parsed_data["Adverse of ASM"] = flat_cells[1]+","+flat_cells[4]+","+flat_cells[8]

def etiology_parsing(table):
    row = table.rows[0]
    cell = row.cells[1]
    clean_text = cell.text.strip().replace('\n', ' ')
    parsed_data["Etiology"] = clean_text
    temp = row.cells[3].text.strip().replace('\n', ' ')
    parsed_data["Mental Retardation"] = temp

def extract_eeg_electrodes(text):
    """Return canonical electrode names, including codes joined as T4T6 or T3T5."""
    electrodes = []
    seen = set()
    for match in EEG_ELECTRODE_PATTERN.findall(str(text or "")):
        electrode = EEG_ELECTRODE_BY_CASEFOLD[match.casefold()]
        if electrode not in seen:
            electrodes.append(electrode)
            seen.add(electrode)
    return electrodes

def eeg_parsing(table):
    selected_electrodes = []
    seen = set()
    for row in table.rows[1:]:
        value_cell = row.cells[-1]
        clean_text = value_cell.text.strip().replace('\n', ' ')
        all_text_nodes = value_cell._tc.xpath('.//w:t')
        xpath_text = "".join(node.text for node in all_text_nodes if node.text).strip()
        value_text = xpath_text if len(xpath_text) > len(clean_text) else clean_text

        for electrode in extract_eeg_electrodes(value_text):
            if electrode not in seen:
                selected_electrodes.append(electrode)
                seen.add(electrode)

    parsed_data["EEG Focality"] = ", ".join(selected_electrodes)


def initial_table_selection(file_path):
    reset_parsed_data()
    doc = Document(file_path)
    # List to store flattened data for CSV
    for table in doc.tables:
        row = table.rows[0]
        # Scanning cell by cell within the row
        cell = row.cells[0]
        # Clean text: remove extra spaces and newline characters
        clean_text = cell.text.strip().replace('\n', ' ')
        all_text_nodes = cell._tc.xpath('.//w:t')
        xpath_text = "".join([node.text for node in all_text_nodes if node.text]).strip()
        if len(xpath_text) > len(clean_text):
            parsed_text = xpath_text
        else:
            parsed_text = clean_text
        # Table Selection Logic
        DISPATCH_MAP = {
            "Baseline 基本資料": baseline_parsing,
            "Adverse of ASM": adverse_parsing,
            "EEG focality": eeg_parsing,
            "Etiology of epilepsy★" : etiology_parsing,
            "Semiology" : baseline_parsing,
            # 之後新增表格只需加在這邊
        }
        handler = DISPATCH_MAP.get(parsed_text)

        if handler:
            handler(table)
        else:
            continue
    print_parsed_data()
    return parsed_data.copy()

def extract_static_data_from_docx(file_path, quiet=True):
    if not quiet:
        return initial_table_selection(file_path)
    with redirect_stdout(io.StringIO()):
        return initial_table_selection(file_path)

def save_to_csv(data, output_file):
    # Since the source is a Key-Value pair table,
    # we can pair them into a dictionary for a standard CSV format
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        row_to_write = [item for pair in data.items() for item in pair]
        for i in range(0, len(row_to_write), 2):
            writer.writerow([row_to_write[i], row_to_write[i + 1]])

def save_static_output(file_path, output_dir="./new_csvs/static_data"):
    data = initial_table_selection(file_path)
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.basename(file_path)
    base_name = fname[:-5] if fname.lower().endswith(".docx") else fname
    output_file = os.path.join(output_dir, f"{base_name}_static_data.csv")
    save_to_csv(data, output_file)
    print(f"Saved to {output_file}")
    return output_file

def print_parsed_data():
    for key, value in parsed_data.items():
        safe_print(f"{key}: {value}")
    safe_print("\n")

def process_directory(input_dir, output_dir="./new_csvs/static_data"):
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
            out_path = save_static_output(file_path, output_dir)
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"ERROR {fname}: {e}")
            continue

        if out_path:
            success.append((fname, out_path))
            print(f"OK {fname}: saved {out_path}")
        else:
            skipped.append(fname)
            print(f"SKIP {fname}: no static data saved")

    print(f"\nDone: {len(success)} succeeded, {len(skipped)} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for fname, err in errors:
            print(f"  {fname}: {err}")

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
