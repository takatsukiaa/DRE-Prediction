from docx import Document
from tkinter.filedialog import askopenfilename
import csv
parsed_data = {
    "Patient ID": "",
    "Birthday": "",
    "Gender": "",
    "Age of 1st NCKU visit (ymd)": "",
    "Drug-refractory epilepsy (≥3ASM & Sz in recent 1 yr)": "",
    "Systemic disease": "",
    # other fields

}
def baseline_parsing(table):
    flat_cells = []
    for row in table.rows:
        visited_cells = set()
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
            # "Adverse of ASM": outpatient_parsing,
            # "EEG focality": baseline_parsing,
            # "Etiology of epilepsy" : baseline_parsing,
            # "Semiology" : baseline_parsing,
            # 之後新增表格只需加在這邊
        }
        handler = DISPATCH_MAP.get(parsed_text)

        if handler:
            handler(table)
        else:
            print("No handler found. Exiting.")
            exit(0)

# def save_to_csv(data_list, output_file):
#     # Since the source is a Key-Value pair table,
#     # we can pair them into a dictionary for a standard CSV format
#     with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
#         writer = csv.writer(f)
#         # Writing as a simple key-value list for this example
#         # Depending on your CSV structure, you might need to reshape the list
#         for i in range(0, len(data_list), 2):
#             if i + 1 < len(data_list):
#                 writer.writerow([data_list[i], data_list[i+1]])
# Example usage:
if __name__ == "__main__":
    # Prompt the user to select a .docx file
    input_docx = askopenfilename(title="Select a .docx file", filetypes=[("Word Documents", "*.docx")])
    output_csv = "formatted_data.csv"  # You can also prompt for this if needed
    if input_docx:
        initial_table_selection(input_docx)
        # save_to_csv(data, output_csv)
    else:
        print("No file selected. Exiting.")
        exit(0)