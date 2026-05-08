import sys
from docx import Document

def inspect_tables(filepath, output_file):
    try:
        doc = Document(filepath)
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for i, table in enumerate(doc.tables):
            f.write(f"--- Table {i} ---\n")
            f.write(f"Rows: {len(table.rows)}\n")
            
            # Print first 5 rows to identify the table
            for r_idx, row in enumerate(table.rows[:5]):
                row_data = []
                for cell in row.cells:
                    text = cell.text.strip().replace('\n', ' ')
                    row_data.append(text)
                f.write(f"Row {r_idx}: {row_data}\n")
            f.write("\n")

if __name__ == "__main__":
    inspect_tables(r"C:\Users\89665\Desktop\Epilepsy\DRE_Prediction\DRE\0003DRE.docx", "inspect_output.txt")
