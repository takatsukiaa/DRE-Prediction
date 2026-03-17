import pandas as pd
import glob
import os

# 1. 定義哪些項目中的 0 其實是「沒做測試」 (請根據您的 CSV 第一欄名稱調整)
# ⚠️ 注意：不要把 Seizure 放進來，因為發作 0 次是有意義的數據
MISSING_IS_ZERO_LIST = [
    "Delta waves",
    "Theta waves",
    "Temporal Slow-wave",
    "Frontal Slow-wave",
    "Occipital Slow-wave",
    "Parietal Slow-wave",
    "Central Slow-wave",
    "Spike ED",
    "Sharp ED",
    "Spike-and-wave ED",
    "Periodic Discharge",
    "Temporal ED",
    "Frontal ED",
    "Occipital ED",
    "Parietal ED",
    "Central ED",
    "TLED",
    "TRED",
    "FLED",
    "FRED",
    "OLED",
    "ORED",
    "PLED",
    "PRED",
    "CLED",
    "CRED",
    "TL",
    "TR",
    "FL",
    "FR",
    "PL",
    "PR",
    "OL",
    "OR",
    "CL",
    "CR",
    "MMSE (30)",
    "CASI (100)",
    "CASI Q1 (10)",
    "CASI Q2 (12)",
    "CASI Q3 (8)",
    "CASI Q4 (10)",
    "CASI Q5 (18)",
    "CASI Q6 (12)",
    "CASI Q7 (10)",
    "CASI Q8 (10)",
    "CASI Q9 (10)",
    "WAIS (100)",
    "WBC(10^3/uL)",
    "RBC(10^3/uL)",
    "Hb(g/uL)",
    "Na(mmol/L)",
    "K(mmol/L)",
    "BUN(mg/dL)",
    "CREA(mg/dL)",
    "eGFR",
    "AST(U/L)",
    "ALT(U/L)",
    "CHOL(mg/dL)",
    "LDL-C(mg/dL)",
    "HDL-C(mg/dL)",
    "HbA1c(%)",
    "CBZ(ug/mL)",
    "PTN(ug/mL)",
    "VALP(ug/mL)",
]


def batch_clean_csv(folder_path):
    # 建立一個備份資料夾，避免改錯
    backup_folder = os.path.join(folder_path, "backup_original")
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    csv_files = glob.glob(os.path.join(folder_path, "Pt*.csv"))

    print(f"開始處理 {len(csv_files)} 個檔案...")

    for file_path in csv_files:
        # 讀取檔案
        df = pd.read_csv(file_path, header=None)  # 使用 header=None 完整保留結構

        # 備份原始檔案
        file_name = os.path.basename(file_path)
        df.to_csv(os.path.join(backup_folder, file_name), index=False, header=False)

        # 遍歷每一列進行檢查
        for idx in range(len(df)):
            row_label = str(df.iloc[idx, 0]).strip()

            # 如果這一列的名字在我們的「缺失清單」中
            if any(target in row_label for target in MISSING_IS_ZERO_LIST):
                # 將該列中除了第一欄以外的 0 全部替換為空值
                # 我們先把資料轉成數字，如果是 0 就變為 None
                row_values = df.iloc[idx, 1:]

                # 邏輯：如果是數字 0 或字串 '0'，就換成空字串 ''
                df.iloc[idx, 1:] = row_values.replace({0: '', '0': '', '0.0': ''})

        # 寫回原始檔案
        df.to_csv(file_path, index=False, header=False)
        print(f"已更新: {file_name}")

    print("\n--- 批量處理完成！原始檔案已備份至 'backup_original' 資料夾 ---")


# 執行修改 (請確保您在正確的目錄下)
if __name__ == "__main__":
    batch_clean_csv("../")