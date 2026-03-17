import pandas as pd
import numpy as np
import sqlite3
import glob
import os
import csv
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 1. 藥物與 DDD 標準配置 (參考自 PCA.py)
# ==========================================
DRUG_MAP = {
    "Lamictal 50mg/tab": ("Lamotrigine", 50), "Lyrica 75mg/cap": ("Pregabalin", 75),
    "Topamax 100mg/tab": ("Topiramate", 100), "Fycompa 2mg/tab": ("Perampanel", 2),
    "Keppra 500mg/tab": ("Levetiracetam", 500), "Trileptal 300mg/tab": ("Oxcarbazepine", 300),
    "Depakine 500mg/tab": ("Valproate", 500), "Frisium 10mg/tab": ("Clobazam", 10),
    "Vimpat 100mg/tab": ("Lacosamide", 100), "Zonegran 100mg/tab": ("Zonisamide", 100),
    "Carpine 200mg/tab": ("Carbamazepine", 200), "Aclonax 0.5mg/tab": ("Clonazepam", 0.5),
    "Rivotril 0.5mg/tab": ("Clonazepam", 0.5), "Carbamazepine 200mg/tab": ("Carbamazepine", 200),
    "Tegretol CR 200mg/tab": ("Carbamazepine", 200), "Dilantin 100mg/cap": ("Phenytoin", 100),
    "Phenytoin 100mg/cap": ("Phenytoin", 100), "Fycompa 2mg": ("Perampanel", 2),
    "Topiramate 25mg/cap": ("Topiramate", 25), "Topiramate 100mg/tab": ("Topiramate", 100),
    "Phenobarbital 30mg/tab": ("Phenobarbital", 30), "Depakine 200mg/mL oral sol": ("Valproate", 200),
    "Depakine Chrono 500mg/tab": ("Valproate", 500), "Depakine Chrono 200mg/tab": ("Valproate", 200),
    "Risperdal 1 mg/tab": ("Risperidone", 1)
}

DDD_TABLE = {
    "Lamotrigine": 300, "Pregabalin": 300, "Topiramate": 300, "Perampanel": 8,
    "Levetiracetam": 1500, "Oxcarbazepine": 1000, "Carbamazepine": 1000,
    "Valproate": 1500, "Clobazam": 20, "Lacosamide": 300, "Zonisamide": 200,
    "Clonazepam": 8, "Phenytoin": 300, "Phenobarbital": 100, "Risperidone": 5
}

# ==========================================
# 2. 檔案初始化邏輯 (維持不變)
# ==========================================
db_name = "PatientClinicalData.db"
if os.path.exists(db_name):
    try:
        os.remove(db_name)
    except PermissionError:
        print(f"警告：無法刪除 {db_name}，請關閉相關檢視軟體。")


class ClinicalDBManager:
    def __init__(self, db_name="PatientClinicalData.db"):
        self.conn = sqlite3.connect(db_name)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS patients
                       (
                           patient_id
                           TEXT
                           PRIMARY
                           KEY,
                           gender
                           TEXT,
                           dob
                           TEXT,
                           age_of_onset
                           TEXT,
                           dre
                           TEXT,
                           time_interval
                           TEXT,
                           asm_adverse
                           TEXT
                       )
                       ''')
        cursor.execute('CREATE TABLE IF NOT EXISTS patient_tags (patient_id TEXT, category TEXT, value TEXT)')
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS clinical_features (patient_id TEXT, feature_name TEXT, time_point TEXT, value REAL)')
        self.conn.commit()

    def _parse_list(self, text):
        if not text or str(text).lower() in ['no', 'nan', 'none', '0']: return []
        return [i.strip() for i in str(text).replace('"', '').split(',') if i.strip()]

    def import_patient_csv(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                lines = list(reader)

            meta = {}
            timeline_idx = 0
            for i, parts in enumerate(lines):
                if not parts: continue
                key = parts[0].strip()
                val = parts[1].strip() if len(parts) > 1 else ""
                if "Patient ID" in key:
                    meta['id'] = val
                elif "Gender" in key:
                    meta['gender'] = val
                elif "Date of Birth" in key:
                    meta['dob'] = val
                elif "Time interval" in key:
                    meta['interval'] = val
                elif "Age of onset" in key:
                    meta['onset'] = val
                elif "DRE" in key:
                    meta['dre'] = val
                elif "Systemic disease" in key:
                    meta['systemic'] = self._parse_list(val)
                elif "Comorbidities" in key:
                    meta['comorb'] = self._parse_list(val)
                elif "EEG Focality" in key:
                    meta['eeg'] = self._parse_list(val)
                elif "Focal" in key:
                    meta['focal'] = self._parse_list(val)
                elif "Adverse of ASM" in key:
                    meta['asm_adv'] = val
                elif "Etiology" in key:
                    meta['etio'] = self._parse_list(val)
                if "# Timeline" in key:
                    timeline_idx = i + 1
                    break

            cursor = self.conn.cursor()
            # 檢查 Metadata 是否已存在 (只記錄一次)
            cursor.execute("SELECT 1 FROM patients WHERE patient_id = ?", (meta['id'],))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (meta['id'], meta.get('gender'), meta.get('dob'), meta.get('onset'),
                                meta.get('dre'), meta.get('interval'), meta.get('asm_adv')))
                tag_map = {'Etiology': meta.get('etio', []), 'Systemic disease': meta.get('systemic', []),
                           'Comorbidities': meta.get('comorb', []), 'Focal': meta.get('focal', []), "EEG Focality": meta.get('eeg', []),}
                for cat, items in tag_map.items():
                    for item in items:
                        cursor.execute('INSERT INTO patient_tags VALUES (?, ?, ?)', (meta['id'], cat, item))

            # --- Timeline 解析與 DDD 計算 ---
            df_timeline = pd.read_csv(file_path, skiprows=timeline_idx)
            df_timeline = df_timeline.dropna(subset=[df_timeline.columns[0]])
            time_cols = df_timeline.columns[1:]

            # 初始化每個月的總 DDD 與 用藥數
            ddd_totals = pd.Series(0.0, index=time_cols)
            drug_counts = pd.Series(0, index=time_cols)
            features_to_db = []

            for _, row in df_timeline.iterrows():
                feat_name = str(row[0]).strip()
                if feat_name.startswith('#'): continue

                # 判斷是否為藥物欄位，進行計算
                if feat_name in DRUG_MAP:
                    comp, strength = DRUG_MAP[feat_name]
                    ddd_std = DDD_TABLE.get(comp, 1.0)
                    counts = pd.to_numeric(row[1:], errors='coerce').fillna(0)
                    # 累加 DDD 負荷與計算用藥種類
                    ddd_totals += (counts * strength) / ddd_std
                    drug_counts += (counts > 0).astype(int)

                # 原始特徵寫入
                for t in time_cols:
                    if pd.notna(row[t]) and str(row[t]).strip() != "":
                        try:
                            features_to_db.append((meta['id'], feat_name, t, float(row[t])))
                        except:
                            continue

            # 將計算出的 ddd_load 與 drug_count 加入匯入清單
            for t in time_cols:
                features_to_db.append((meta['id'], 'ddd_load', t, float(ddd_totals[t])))
                features_to_db.append((meta['id'], 'drug_count', t, int(drug_counts[t])))

            cursor.executemany('INSERT INTO clinical_features VALUES (?, ?, ?, ?)', features_to_db)
            self.conn.commit()
            return meta['id']
        except Exception as e:
            print(f"錯誤 {file_path}: {e}")
            return None


manager = ClinicalDBManager(db_name)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    selected_path = filedialog.askdirectory(title="選擇病人資料夾")
    if selected_path:
        csv_files = glob.glob(os.path.join(selected_path, "Pt*.csv"))
        for f in csv_files:
            pid = manager.import_patient_csv(f)
            if pid: print(f"成功匯入並計算 DDD: {pid}")