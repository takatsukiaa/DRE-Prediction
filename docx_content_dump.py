from tkinter.filedialog import askopenfilename

from docx import Document
import csv


def dump_table_structure(file_path):
    """
    Dumps the underlying grid structure of all tables in a document for debugging.
    """
    doc = Document(file_path)
    for t_idx, table in enumerate(doc.tables):
        print(f"\n--- Table {t_idx} Structure ---")
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                # Use the underlying XML element ID to identify unique cells
                cell_id = id(cell._tc)
                content = cell.text.strip().replace('\n', '\\n')[:20]
                print(f"Row {r_idx}, Col {c_idx} | ID: {cell_id} | Text: {content}...")


def save_to_csv(data_list, output_file):
    # Since the source is a Key-Value pair table,
    # we can pair them into a dictionary for a standard CSV format
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # Writing as a simple key-value list for this example
        # Depending on your CSV structure, you might need to reshape the list
        for i in range(0, len(data_list), 2):
            if i + 1 < len(data_list):
                writer.writerow([data_list[i], data_list[i + 1]])

# Execution
input_docx = askopenfilename(title="Select a .docx file", filetypes=[("Word Documents", "*.docx")])
data = dump_table_structure(input_docx)
# save_to_csv(data, "output.csv")