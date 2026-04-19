import re

from docx import Document
from tkinter.filedialog import askopenfilename
import csv

parsed_data = {
    "Patient ID": "",
    "Birthday": "",
    "Gender": "",
    "1st Seizure Age": "",
    "DRE": "",
    "Drug-refractory epilepsy": "",
    "Systemic disease": "",
    "Adverse of ASM": "",
    "Etiology": "",
    "Focal": "",
    "EEG Focality": "",
    # other fields
}
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

def eeg_parsing(table):
    flat_cells = []
    visited_cells = set()
    pattern = r'\b[A-Z]+[0-9z][A-Z0-9]*\b'
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

    start_idx = 2
    for i in enumerate(flat_cells):
        text = flat_cells[start_idx]
        start_idx += 2
        if start_idx >= len(flat_cells):
            break
        matches = re.findall(pattern, text)
        if matches:
            matches = ", ".join(matches)
            parsed_data["EEG Focality"] = matches


def initial_table_selection(file_path):
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
    return parsed_data
def save_to_csv(output_file):
    # Since the source is a Key-Value pair table,
    # we can pair them into a dictionary for a standard CSV format
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["# Metadata"])
        row_to_write = [item for pair in parsed_data.items() for item in pair]
        for i in range(0, len(row_to_write), 2):
            writer.writerow([row_to_write[i], row_to_write[i + 1]])

def print_parsed_data():
    for key, value in parsed_data.items():
        print(f"{key}: {value}")