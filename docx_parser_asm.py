import os
import pandas as pd
from docx import Document
import re
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import numpy as np

DRUG_FREQUENCY_MAP = [
    ("BIDPC (7 days)", 2),
    ("BIDPC", 2),
    ("BID", 2),
    ("TIDPC", 3),
    ("TID", 3),
    ("QDPC", 1),
    ("QNPC", 1),
    ("QDLUNCH", 1),
    ("QO HS", 0.5),
    ("QOHS", 0.5),
    ("QOD", 0.5),
    ("HS", 1),
    ("QD", 1),
    ("QN", 1),
    ("as order", 1),
]


def lookup_freq(freq_str):
    for keyword, multiplier in DRUG_FREQUENCY_MAP:
        if keyword in freq_str:
            return multiplier
    return 1

DDD_MG = {
        "Lamotrigine": 300,
        "Pregabalin": 300,
        "Topiramate": 300,
        "Perampanel": 8,
        "Levetiracetam": 1500,
        "Oxcarbazepine": 1000,
        "Carbamazepine": 1000,
        "Valproate": 1500,
        "Clobazam": 20,
        "Lacosamide": 300,
        "Zonisamide": 200,
        "Clonazepam": 8,
        "Phenytoin": 300,
        "Phenobarbital": 100,
        "Risperidone": 5,
        "Gabapentin" : 1800,
        "Diazepam" : 10,
        "Brivaracetam": 100,
}

# MAP CSV ROW → (drug_name, mg_per_unit)
ASM_ROW_MAP = {
    "Lamictal (Lamotrigine) 5mg/tab": ("Lamotrigine", 50),
    "Lamictal (Lamotrigine) 50mg/tab": ("Lamotrigine", 50),
    "Lamictal 5mg/tab (Lamotrigine)": ("Lamotrigine", 50),
    "Lamictal 5mg/tab(Lamotrigine)": ("Lamotrigine", 50),
    "Lamictal 50mg/tab (Lamotrigine)": ("Lamotrigine", 50),
    "Lamictal 50mg/tab(Lamotrigine)": ("Lamotrigine", 50),
    "Lyrica 75mg/cap (Pregabalin)": ("Pregabalin", 75),
    "Lyrica 75mg/cap(Pregabalin)": ("Pregabalin", 75),
    "Topamax 100mg/tab (Topiramate)": ("Topiramate", 100),
    "Topamax 25mg/cap (Topiramate)": ("Topiramate", 25),
    "Topamax 100mg/tab(Topiramate)": ("Topiramate", 100),
    "Topamax 25mg/cap(Topiramate)": ("Topiramate", 25),
    "Topamax 25mg/tab (Topiramate)": ("Topiramate", 25),
    "Fycompa 2mg/tab (Perampanel)": ("Perampanel", 1),
    "Fycompa 2mg (Perampanel)": ("Perampanel", 1),
    "Fycompa 4mg/tab (Perampanel)": ("Perampanel", 1),
    "Keppra 500mg/tab (Levetiracetam)": ("Levetiracetam", 500),
    "Keppra 500mg/tab(Levetiracetam)": ("Levetiracetam", 500),
    "Keppra (Levetiracetam) 500mg/tab": ("Levetiracetam", 500),
    "Trileptal 300mg/tab (Oxcarbazepine)": ("Oxcarbazepine", 300),
    "Trileptal 300mg/tab(Oxcarbazepine)": ("Oxcarbazepine", 300),
    "Trileptal 300mg (Oxcarbazepine)": ("Oxcarbazepine", 300),
    "Depakine 500mg/tab (Valproate)": ("Valproate", 500),
    "Valproate (Depakine)": ("Valproate", 500),
    "Frisium 10mg/tab (Clobazam)": ("Clobazam", 10),
    "Frisium tab 10mg/tab (Clobazam)": ("Clobazam", 10),
    "Clobazam tab 10mg/tab (Frisium)": ("Clobazam", 10),
    "Vimpat 100mg/tab (Lacosamide)": ("Lacosamide", 100),
    "Vimpet 100mg/tab (Lacosamide)": ("Lacosamide", 100),
    "Zonegran 100mg/tab (Zonisamide)": ("Zonisamide", 100),
    "Zonegran tab 100mg/tab (Zonisamide)": ("Zonisamide", 100),
    "Carpine 200mg/tab (Carbamazepine)": ("Carbamazepine", 200),
    "Aclonax 0.5mg/tab (Clonazepam)": ("Clonazepam", 0.5),
    "Rivotril 0.5mg/tab (Clonazepam)": ("Clonazepam", 0.5),
    "Rivotril tab 0.5mg/tab (Clonazepam)": ("Clonazepam", 0.5),
    "Carbamazepine 200mg/tab (Carbamazepine)": ("Carbamazepine", 200),
    "Carbamazepine 200mg/tab (Carpine)": ("Carbamazepine", 200),
    "Carbamazepine C.R 200mg/tab (tegretol)": ("Carbamazepine", 200),
    "Carbamazepine C.R": ("Carbamazepine", 200),
    "Tegretol CR 200mg/tab (Carbamazepine)": ("Carbamazepine", 200),
    "Tegretol CR 200mg/tab(Carbamazepine)": ("Carbamazepine", 200),
    "Tegretol CR (Carbamazepine) 200mg/tab": ("Carbamazepine", 200),
    "Dilantin 100mg/cap (Phenytoin)": ("Phenytoin", 100),
    "Dilantin cap 100mg/cap (Phenytoin)": ("Phenytoin", 100),
    "Dilantin cap 100mg/cap(Phenytoin)": ("Phenytoin", 100),
    "Dilantin (Phenytoin) 100mg/cap": ("Phenytoin", 100),
    "Phenytoin 100mg/cap": ("Phenytoin", 100),
    "Fycompa 2mg": ("Perampanel", 1),
    "Topiramate 25mg/cap": ("Topiramate", 25),
    "Topiramate 100mg/tab": ("Topiramate", 100),
    "Topiramate (topamax) 25mg/cap": ("Topiramate", 25),
    "Toporamate (topamax) 25mg/cap": ("Topiramate", 25),
    "Topiramate (topamax) 25mg/tab": ("Topiramate", 25),
    "Topiramate (topamax) 100mg/tab": ("Topiramate", 100),
    "Toporamate (topamax) 100mg/tab": ("Topiramate", 100),
    "Toporamate (topamax) 25mg/tab": ("Topiramate", 25),
    "Toporamate (topamax) 25cap/tab": ("Topiramate", 25),
    "Phenobarbital 30mg/tab": ("Phenobarbital", 30),
    "Depakine 200mg/mL oral sol": ("Valproate", 200),
    "Depakine Chrono 500mg/tab (Valproate)": ("Valproate", 500),
    "Depakine Chrono 200mg/tab (Valproate)": ("Valproate", 200),
    "Depakine Chrono 500mg/tab(Valproate)": ("Valproate", 500),
    "Depakine Chrono 200mg/tab(Valproate)": ("Valproate", 200),
    "Depakine Chrono 500mg/tab": ("Valproate", 500),
    "Depakine Chrono 200mg/tab": ("Valproate", 200),
    "Risperdal 1 mg/tab": ("Risperidone", 1),
    "Risperdal 1 mg/tab (Risperidone)": ("Risperidone", 1),
    "Tegretol 200mg/tab":("Carbamazepine", 200),
    "Depakine 200mg/mL oral sol, 40 mL/btl (Valproate)": ("Valproate", 1),
    "Depakine oral sol 200mg/mL, 40mL/btl(Valproate)": ("Valproate", 1),
    "neurontin 300mg/cap (Gabapentin)" :("Gabapentin", 300),
    "Neurontin cap 300mg/cap(Gabapentin)" :("Gabapentin", 300),
    "Tegretol/Carpine 200mg/tab (Carbamazepine)" : ("Carbamazepine", 200),
    "UFree ER 500mg/tab (Levetiracetam)" : ("Levetiracetam", 500),
    "Dupin 5mg/tab (Diazepam)" : ("Diazepam", 5),
    "Briviact 100mg/tab (Brivaracetam)": ("Brivaracetam", 100),

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
        exit(1)

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

    df_raw = pd.DataFrame(data, columns=headers)
    df_ddd = df_raw.copy()
    cols_to_clear = df_raw.columns != 'Date'
    df_ddd.loc[:, cols_to_clear] = None
    df_ddd = df_ddd.set_index('Date')
    df_ddd = df_ddd.astype(float)
    df_ddd.reset_index(inplace=True)
    df_ddd.drop("備註", axis=1, inplace=True)
    for label, data in df_raw.items():
        i = 0
        if label not in ASM_ROW_MAP:
            continue
        else:
            drug, mg_strength = ASM_ROW_MAP[label]
            if drug not in df_ddd.columns:
                df_ddd.rename({label: drug}, axis=1, inplace=True)
            else:
                df_ddd.drop(label, axis=1, inplace=True)
            ddd_den = DDD_MG[drug]
            for value in data:
                if value:
                    total_dose = 0
                    items = value.split(',')
                    for item in items:
                        item = re.sub(r'[^a-zA-Z0-9./ ]', '', item)
                        if not item.strip():
                            continue
                        units = re.findall(r"(\d+.\d+|\d+)", item)
                        freq = re.findall(r"(?<=/).*?(?=,|$)", item)
                        freq = re.sub(r'[\(\[\{].*?[\)\]\}]', '', freq[0]) if freq else []
                        if len(units) == 0 or len(freq) == 0:
                            continue
                        if "PRN" in freq:
                            continue
                        if "多開" in freq:
                            continue
                        freq_multiplier = lookup_freq(freq)
                        total_dose += float(units[0]) * freq_multiplier * mg_strength

                    ddd = total_dose / ddd_den
                    try:
                        if not np.isnan(df_ddd.loc[i, drug]):
                            df_ddd.loc[i, drug] += ddd
                        else:
                            df_ddd.loc[i, drug] = ddd
                        i+=1
                    except Exception as e:
                        print(e)
                else:
                    i += 1
                    continue

    # fill in empty months!
    df_ddd['Date'] = pd.to_datetime(df_ddd['Date'])
    try:
        df_ddd.drop('是否為吳醫師看診(不是：0；是：1；無看診紀錄：-)', axis=1, inplace=True)
    except:
        pass
    # df_ddd.dropna(how='all', axis=1, inplace=True)
    df_ddd = df_ddd.set_index('Date')
    df_ddd = df_ddd.resample('ME').mean().ffill()
    df_ddd = df_ddd.round(2)


    return df_ddd


def save_ddd_csv(df, output_path):
    """Save the DDD DataFrame to CSV with proper encoding."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=True, encoding='utf-8-sig')
    print(f"Saved ASM timeline to {output_path}")
    return output_path

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
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Select Patient Docx File (Cancel to Exit)",
        filetypes=[("Docx files", "*.Docx"), ("All files", "*.*")]
    )
    df = table_selection(file_path)
    if df is not None:
        print(df.columns)
    file_path = os.path.basename(file_path)
    file_path = os.path.join("./new_csvs",file_path)
    df.to_csv(file_path.replace(".docx", "_asm_timeline.csv"), index=True)
 
