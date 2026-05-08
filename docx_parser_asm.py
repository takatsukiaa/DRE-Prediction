import pandas as pd
from docx import Document

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

def extract_asm_timeline(target_table):
    """
    Extracts the ASM Timeline data from the 'ASM狀況' table in a docx file 
    and returns it as a pandas DataFrame.
    """
    # Find the header row (starts with 'Date')
    header_row_idx = -1
    for i, row in enumerate(target_table.rows):
        first_cell_text = get_cell_text(row.cells[0])
        if first_cell_text == "Date":
            header_row_idx = i
            break
            
    if header_row_idx == -1:
        print("Could not find the 'Date' header row in the ASM table.")
        return None

    # Extract headers
    headers = []
    header_row = target_table.rows[header_row_idx]
    for cell in header_row.cells:
        headers.append(get_cell_text(cell))
        
    # Extract data rows
    data = []
    for row in target_table.rows[header_row_idx + 1:]:
        row_data = []
        for cell in row.cells:
            row_data.append(get_cell_text(cell))
        
        # Skip rows that are entirely empty or don't have a date
        if not any(row_data) or not row_data[0]:
            continue
            
        data.append(row_data)

    df = pd.DataFrame(data, columns=headers)
    return df


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
            "ASM狀況" : extract_asm_timeline,
            }
        handler = DISPATCH_MAP.get(first_cell_text)
        if handler:
            df = handler(table)
            return df
        else:
            continue


if __name__ == "__main__":
    # Test the function with the provided file
    filepath = r"C:\Users\89665\Desktop\Epilepsy\DRE_Prediction\DRE\0003DRE.docx"
    df = table_selection(filepath)
    if df is not None:
        print("DataFrame Shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
    # df.to_csv("asm_timeline.csv", index=False)
 
