# Parsing Logic #

## 1. 基本流程 ##

撰寫``docx_parser_static.py``擷取靜態資料，並撰寫GUI介面，方便操作和展示進度

## 2. 細部套件和資料結構

### parsed_data (dict)
```python

parsed_data = {
    "Patient ID": ""
    "Date of Birth": ""
    "Gender": ""
    # other fields

}
```
### initial_table_selection: 透過第一行第一格選擇關鍵字

```python
def initial_table_selection(file_path):
    doc = Document(file_path)
    # List to store flattened data for CSV
    data_points = []

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
        match parsed_text.casefold():
            case "Baseline 基本資料":
                baseline_parsing(table)
        # Other cases ...

            case _:
                print("Unknown Key word")
    
```
### 各Table獨立的Parsing Logic (以baseline為例)

```python
def baseline_parsing(table):
    baseline_targets = ["Patient ID", "1st Seizure Age (ymd)", "Birthday", "Gender","Drug-refractory epilepsy (≥3ASM & Sz in recent 1 yr)", "Systemic disease"] # Alias in .docx file
    for row in table.rows:
        for cell in row.cells:
            clean_text = cell.text.strip().replace('\n', ' ')
            all_text_nodes = cell._tc.xpath('.//w:t')
            xpath_text = "".join([node.text for node in all_text_nodes if node.text]).strip()
            if len(xpath_text) > len(clean_text):
                parsed_text = xpath_text
            else:
                parsed_text = clean_text
            if parsed_text not in baseline_targets:
                continue
            else:
            # Put contents inside parsed_data dict
```
其他Table皆使用相似的內容

### save_to_csv
此套件是將擷取出的資料以特定格式存入指定csv

