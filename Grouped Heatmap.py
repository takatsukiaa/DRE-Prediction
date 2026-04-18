import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

DPI = 600
# =============================================================================
# 1. CONSTANTS & MAPPING
# =============================================================================
ETIOLOGY_LIST = [
    "Hypoxia at birth", "Developmental delay", "Cerebral palsy", "Family history of epilepsy",
    "Cerebral structure lesion", "Cerebral trauma", "Cerebral tumor", "Cerebral infection",
    "Cerebrovascular disorders", "Cerebral immunologic disorders", "Degenerative and other neurological conditions",
    "Hydrocephalus", "Brain Surgery", "Encephalitis"
]

EEG_GROUPS = {
    "Frontal": ["Fp1", "F3", "F7", "Fp2", "F4", "F8", "Fz"],
    "Temporal": ["T3", "T4", "T5", "T6", "S1", "S2"],
    "Central": ["C3", "C4", "Cz"],
    "Parietal": ["P3", "P4", "Pz"],
    "Occipital": ["O1", "O2"]
}
SEIZURE_TYPES = {
    "1.1": "FPC", "1.2": "FIC", "1.3": "FBTC",
    "2.1": "PC", "2.2": "IC", "2.3": "BTC",
    "3.1": "AS", "3.2": "GTC", "3.3": "Other G",
    "4": "Unclassified"
}


# =============================================================================
# 2. PLOTTING FUNCTIONS (STATIC COMPONENTS)
# =============================================================================
def plot_etiology_checklist_horizontal(ax, active_codes):
    """Plots the horizontal etiology checklist with purple indicators."""
    ax.axis("off")
    ax.text(0, 0.98, "Etiology", fontsize=20, fontweight="bold")
    num_items = len(ETIOLOGY_LIST)
    num_cols = 2
    items_per_col = (num_items + num_cols - 1) // num_cols
    for i, name in enumerate(ETIOLOGY_LIST):
        col, row = i // items_per_col, i % items_per_col
        x_base, y_pos = col * 0.45, 0.90 - (row * 0.06)
        code = i + 1
        is_active = code in active_codes
        color = "mediumpurple" if is_active else "lightgray"
        circ = patches.Circle((x_base + 0.05, y_pos), 0.015, color=color, fill=is_active, alpha=0.8,
                              linewidth=1 if not is_active else 0)
        ax.add_patch(circ)
        if is_active:
            ax.text(x_base + 0.05, y_pos, "V", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x_base + 0.10, y_pos, f"{code}. {name}", fontsize=11, va="center",
                fontweight="bold" if is_active else "normal")


def plot_eeg_focality_checklist(ax, eeg_info_str):
    """Plots detailed electrode checkboxes grouped by region with orange indicators."""
    ax.axis("off")
    ax.text(0, 1.0, "EEG Focality", fontsize=20, fontweight="bold")
    y_ptr = 0.93
    for region, electrodes in EEG_GROUPS.items():
        ax.text(0, y_ptr, region, fontsize=13, fontweight="bold", color="darkorange")
        y_ptr -= 0.05
        for idx, elec in enumerate(electrodes):
            col, row = idx % 4, idx // 4
            x_pos, curr_y = col * 0.24, y_ptr - (row * 0.07)
            is_active = elec.upper() in eeg_info_str.upper()
            color = "orange" if is_active else "lightgray"
            circ = patches.Circle((x_pos + 0.02, curr_y), 0.015, color=color, fill=is_active, alpha=0.8,
                                  linewidth=1 if not is_active else 0)
            ax.add_patch(circ)
            if is_active:
                ax.text(x_pos + 0.02, curr_y, "V", color="white", ha="center", va="center", fontsize=8,
                        fontweight="bold")
            ax.text(x_pos + 0.05, curr_y, elec, fontsize=11, va="center", fontweight="bold" if is_active else "normal")
        num_rows = (len(electrodes) + 3) // 4
        y_ptr -= (num_rows * 0.07) + 0.04


def plot_seizure_type_checklist(ax, focal_info_str):
    """
    Plots a checklist for Seizure Types showing only the numeric codes.
    Checks boxes based on codes found in the 'Focal' metadata field.
    """
    ax.axis("off")
    ax.text(0, 1.0, "Seizure Type (Focal/Gen)", fontsize=20, fontweight="bold")

    # Parse numeric codes from string (e.g., "1.2, 2.3" -> ['1.2', '2.3'])
    active_codes = re.findall(r'\d+\.\d+|\d+', str(focal_info_str))

    codes_list = list(SEIZURE_TYPES.keys())
    num_cols = 5  # Display in 5 columns to keep it compact

    for i, code in enumerate(codes_list):
        col = i % num_cols
        row = i // num_cols

        x_pos = col * 0.18
        y_pos = 0.85 - (row * 0.25)

        is_active = code in active_codes
        color = "crimson" if is_active else "lightgray"

        # Draw Circle
        circ = patches.Circle((x_pos + 0.02, y_pos), 0.02, color=color, fill=is_active, alpha=0.8)
        ax.add_patch(circ)
        if is_active:
            ax.text(x_pos + 0.02, y_pos, "V", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

        # Display ONLY the Number
        ax.text(x_pos + 0.05, y_pos, code, fontsize=12, va="center",
                fontweight="bold" if is_active else "normal")


# =============================================================================
# TL/TR proportional bar (no text)
# =============================================================================
def draw_lr_proportion(ax, cx, cy, left_val, right_val, colors, i, j):
    """Fill the entire grid cell with TL/TR proportional coloring."""
    total = left_val + right_val
    if total <= 0:
        return

    left_ratio = left_val / total
    right_ratio = right_val / total

    # Draw full-width left proportion
    ax.add_patch(
        patches.Rectangle(
            (j, i),  # top-left of cell
            left_ratio,  # width
            1.0,  # height
            color=colors,
            alpha=0.88
        )
    )

    # Draw full-width right proportion
    ax.add_patch(
        patches.Rectangle(
            (j + left_ratio, i),  # start where left ended
            right_ratio,
            1.0,
            color=colors,
            alpha=0.88
        )
    )

    # Add L and R labels
    if left_ratio != 0:
        ax.text(j + left_ratio * 0.25, i + 0.5, "L",
                ha="center", va="center", fontsize=8, color="black")
        sep_x = j + left_ratio  # boundary between L and R
        ax.plot(
            [sep_x, sep_x],  # vertical line
            [i, i + 1],  # full cell height
            color="black",
            linewidth=0.8,
            alpha=0.9
        )
    if right_ratio != 0:
        ax.text(j + left_ratio + right_ratio * 0.25, i + 0.5, "R",
                ha="center", va="center", fontsize=8, color="black")


# =============================================================================
# 主程式迴圈 (MAIN EXECUTION LOOP)
# =============================================================================
print("程式已啟動。請選擇檔案以產生圖表。關閉檔案選擇視窗即可退出。")

while True:
    try:
        # =============================================================================
        # 1. FILE SELECTION
        # =============================================================================
        root = Tk()
        root.withdraw()
        file_path = askopenfilename(
            title="Select Patient CSV File (Cancel to Exit)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        # 如果使用者點擊取消或關閉視窗，跳出迴圈結束程式
        if not file_path:
            print("\n未選擇檔案，程式結束執行...")
            root.destroy()
            break

        root.destroy()
        print("\nSelected file:", file_path)

        # =============================================================================
        # 2. LOAD METADATA
        # =============================================================================
        metadata, timeline_start = {}, None
        with open(file_path, 'r', encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row: continue
                if row[0].startswith("# Timeline"):
                    timeline_start = i + 1;
                    break
                metadata[row[0]] = row[1]

        active_eti_codes = list(
            set([int(x) for key, val in metadata.items() if "Etiology" in key for x in re.findall(r'\d+', val)]))
        eeg_raw_text = metadata.get("EEG Focality", "No")

        # =============================================================================
        # 3. LOAD TIMELINE DATA
        # =============================================================================
        df = pd.read_csv(file_path, skiprows=timeline_start, encoding="utf-8-sig")
        df = df.set_index("Time")

        # Trim empty columns
        nonempty = df.sum(axis=0)
        df = df.loc[:, nonempty > 0]
        # =============================================================================
        # Build merged TL/TR slow-wave row
        # =============================================================================
        if "TL" in df.index and "TR" in df.index:
            tl_values = df.loc["TL"]
            tr_values = df.loc["TR"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(tl_values.iloc[j], tr_values.iloc[j]))
            df.loc["Temporal Slow-wave"] = combined

        if "FL" in df.index and "FR" in df.index:
            fl_values = df.loc["FL"]
            fr_values = df.loc["FR"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(fl_values.iloc[j], fr_values.iloc[j]))
            df.loc["Frontal Slow-wave"] = combined

        if "OL" in df.index and "OR" in df.index:
            ol_values = df.loc["OL"]
            or_values = df.loc["OR"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(ol_values.iloc[j], or_values.iloc[j]))
            df.loc["Occipital Slow-wave"] = combined

        if "PL" in df.index and "PR" in df.index:
            pl_values = df.loc["PL"]
            pr_values = df.loc["PR"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(pl_values.iloc[j], pr_values.iloc[j]))
            df.loc["Parietal Slow-wave"] = combined
        if "CL" in df.index and "CR" in df.index:
            cl_values = df.loc["CL"]
            cr_values = df.loc["CR"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(cl_values.iloc[j], cr_values.iloc[j]))
            df.loc["Central Slow-wave"] = combined
        if "TLED" in df.index and "TRED" in df.index:
            tled_values = df.loc["TLED"]
            tred_values = df.loc["TRED"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(tled_values.iloc[j], tred_values.iloc[j]))
            df.loc["Temporal ED"] = combined

        if "FLED" in df.index and "FRED" in df.index:
            fled_values = df.loc["FLED"]
            fred_values = df.loc["FRED"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(fled_values.iloc[j], fred_values.iloc[j]))
            df.loc["Frontal ED"] = combined

        if "OLED" in df.index and "ORED" in df.index:
            oled_values = df.loc["OLED"]
            ored_values = df.loc["ORED"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(oled_values.iloc[j], ored_values.iloc[j]))
            df.loc["Occipital ED"] = combined

        if "PLED" in df.index and "PRED" in df.index:
            pled_values = df.loc["PLED"]
            pred_values = df.loc["PRED"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(pled_values.iloc[j], pred_values.iloc[j]))
            df.loc["Parietal ED"] = combined
        if "CLED" in df.index and "CRED" in df.index:
            cled_values = df.loc["CLED"]
            cred_values = df.loc["CRED"]
            combined = []
            for j in range(len(df.columns)):
                combined.append(max(cled_values.iloc[j], cred_values.iloc[j]))
            df.loc["Central ED"] = combined
        # =============================================================================
        # 3B. CONVERT ASM FREQUENCIES → DDD UNITS
        # =============================================================================

        # WHO DDD REFERENCE VALUES (mg/day)
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
        }

        # MAP CSV ROW → (drug_name, mg_per_unit)
        ASM_ROW_MAP = {
            "Lamictal 50mg/tab": ("Lamotrigine", 50),
            "Lamictal 5mg/tab": ("Lamotrigine", 5),
            "Lyrica 75mg/cap": ("Pregabalin", 75),
            "Topamax 100mg/tab": ("Topiramate", 100),
            "Topamax 25mg/cap": ("Topiramate", 25),
            "Fycompa 2mg/tab": ("Perampanel", 2),
            "Keppra 500mg/tab": ("Levetiracetam", 500),
            "Trileptal 300mg/tab": ("Oxcarbazepine", 300),
            "Depakine 500mg/tab": ("Valproate", 500),
            "Frisium 10mg/tab": ("Clobazam", 10),
            "Vimpat 100mg/tab": ("Lacosamide", 100),
            "Zonegran 100mg/tab": ("Zonisamide", 100),
            "Carpine 200mg/tab": ("Carbamazepine", 200),
            "Aclonax 0.5mg/tab": ("Clonazepam", 0.5),
            "Rivotril 0.5mg/tab": ("Clonazepam", 0.5),
            "Carbamazepine 200mg/tab": ("Carbamazepine", 200),
            "Tegretol CR 200mg/tab": ("Carbamazepine", 200),
            "Dilantin 100mg/cap": ("Phenytoin", 100),
            "Phenytoin 100mg/cap": ("Phenytoin", 100),
            "Fycompa 2mg": ("Perampanel", 2),
            "Topiramate 25mg/cap": ("Topiramate", 25),
            "Topiramate 100mg/tab": ("Topiramate", 100),
            "Phenobarbital 30mg/tab": ("Phenobarbital", 30),
            "Depakine 200mg/mL oral sol": ("Valproate", 200),
            "Depakine Chrono 500mg/tab": ("Valproate", 500),
            "Depakine Chrono 200mg/tab": ("Valproate", 200),
            "Risperdal 1 mg/tab": ("Risperidone", 1),
            "Tegretol 200mg/tab":("Carbamazepine", 200),

        }

        # Only include ASMs that appear in the CSV (i.e., the patient actually used)
        asm_drugs = sorted({
            ASM_ROW_MAP[row][0]
            for row in df.index
            if row in ASM_ROW_MAP
        })

        # ASM table in DDD units (same columns as df)
        df_ddd = pd.DataFrame(0.0, index=asm_drugs, columns=df.columns)

        # Compute DDD values
        for row_name, row in df.iterrows():
            if row_name not in ASM_ROW_MAP:
                continue

            drug, mg_strength = ASM_ROW_MAP[row_name]
            ddd_den = DDD_MG[drug]

            mg_per_day = row.values * mg_strength
            ddd_units = mg_per_day / ddd_den

            df_ddd.loc[drug] += ddd_units  # accumulate multiple strengths
        df_ddd.loc["Total DDD"] = df_ddd.sum(axis=0)
        # =============================================================================
        # 4. FEATURE GROUPS
        # =============================================================================
        feature_groups = {
            "Seizure": [
                "Seizure",
                "Seizures in the past 1 year",
                ">= 3 ASMs",
            ],

            "Slow-Wave and ED": [
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
                "Central ED", ],

            "ASM (DDD Units)": asm_drugs + ["Total DDD"],
            "Cognitive Tests": [
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
                "WAIS (100)"],
            "Blood Tests": [
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
                "VALP(ug/mL)", ],
        }


        # =============================================================================
        # 6. GROUP PLOTTING FUNCTION
        # =============================================================================
        def plot_feature_group(ax, df_source, features, is_ddd=False):
            rows = [f for f in features if f in df_source.index]
            subdf = df_source.loc[rows]

            n_rows = len(subdf)
            n_cols = df_source.shape[1]

            # --- GRID ---
            for r in range(n_rows + 1):
                ax.plot([0, n_cols], [r, r], color="lightgray", linewidth=0.8)
            for c in range(n_cols + 1):
                ax.plot([c, c], [0, n_rows], color="lightgray", linewidth=0.8)

            # --- DRAW CIRCLES ---
            for i, (feat, row) in enumerate(subdf.iterrows()):
                for j, value in enumerate(row):
                    cx = j + 0.5
                    cy = i + 0.5
                    # --- NEW: Render TL/TR proportional bar ---
                    if feat in ["Frontal Slow-wave", "Temporal Slow-wave", "Occipital Slow-wave", "Parietal Slow-wave",
                                "Central Slow-wave"]:
                        if feat == "Temporal Slow-wave":
                            L = df.loc["TL", df.columns[j]]
                            R = df.loc["TR", df.columns[j]]
                        elif feat == "Frontal Slow-wave":
                            L = df.loc["FL", df.columns[j]]
                            R = df.loc["FR", df.columns[j]]
                        elif feat == "Occipital Slow-wave":
                            L = df.loc["OL", df.columns[j]]
                            R = df.loc["OR", df.columns[j]]
                        elif feat == "Parietal Slow-wave":
                            L = df.loc["PL", df.columns[j]]
                            R = df.loc["PR", df.columns[j]]
                        elif feat == "Central Slow-wave":
                            L = df.loc["CL", df.columns[j]]
                            R = df.loc["CR", df.columns[j]]
                        else:  # placeholder
                            L = 0
                            R = 0
                        draw_lr_proportion(ax, cx, cy, L, R, "gold", i, j)

                    elif feat in ["Temporal ED", "Frontal ED", "Occipital ED", "Parietal ED", "Central ED"]:
                        if feat == "Temporal ED":
                            L = df.loc["TLED", df.columns[j]]
                            R = df.loc["TRED", df.columns[j]]
                        elif feat == "Frontal ED":
                            L = df.loc["FLED", df.columns[j]]
                            R = df.loc["FRED", df.columns[j]]
                        elif feat == "Occipital ED":
                            L = df.loc["OLED", df.columns[j]]
                            R = df.loc["ORED", df.columns[j]]
                        elif feat == "Parietal ED":
                            L = df.loc["PLED", df.columns[j]]
                            R = df.loc["PRED", df.columns[j]]
                        elif feat == "Central ED":
                            L = df.loc["CLED", df.columns[j]]
                            R = df.loc["CRED", df.columns[j]]
                        else:  # placeholder
                            L = 0
                            R = 0
                        draw_lr_proportion(ax, cx, cy, L, R, "limegreen", i, j)

                    elif feat in ["Spike ED",
                                  "Sharp ED",
                                  "Spike-and-wave ED",
                                  "Periodic Discharge",
                                  "Delta waves",
                                  "Theta waves",
                                  "Alpha waves" ] and value > 0:
                        radius = 0.12 + 0.25 * (value / 3)
                        if feat in ["Spike ED",
                                      "Sharp ED",
                                      "Spike-and-wave ED",
                                      "Periodic Discharge"]:
                            circ = patches.Circle((cx, cy), radius=radius,
                                                  color="limegreen", alpha=0.85)
                        else:
                            circ = patches.Circle((cx, cy), radius=radius,
                                                  color="gold", alpha=0.85)
                        ax.add_patch(circ)
                        if 0 < value <= 1:
                            ax.text(cx, cy, "R", ha='center', va='center',
                                    color="black", fontsize=9)
                        elif 1 < value <= 2:
                            ax.text(cx, cy, "In", ha='center', va='center',
                                    color="black", fontsize=9)
                        elif 2 < value <= 3:
                            ax.text(cx, cy, "F", ha='center', va='center',
                                    color="black", fontsize=9)
                        else:
                            ax.text(cx, cy, "C", ha='center', va='center',
                                    color="black", fontsize=9)
                    else:
                        # Numeric features → includes DDD rows
                        if value > 0:
                            radius = 0.12 + 0.25 * (value / 6)
                            radius_ASM = 0.12 + 0.25 * (value / 2) if feat != "Total DDD" else 0.12 + 0.25 * (value / 4)
                            radius_CHOL = 0.12 + 0.25 * (value / 150)
                            radius_HDL = 0.12 + 0.25 * (value / 50)
                            radius_CREA = 0.12 + 0.3 * value
                            radius_HbA1c = 0.12 + 0.25 * (value / 10)
                            if is_ddd:
                                circ = patches.Circle((cx, cy), radius=radius_ASM,
                                                      color="royalblue", alpha=0.85)
                                label = f"{value:.2f}"
                            elif feat in ["CHOL(mg/dL)", "LDL-C(mg/dL)", "eGFR", "Na(mmol/L)", "CASI (100)"]:
                                if feat == "eGFR" and value == 1:
                                    label = "≧90"
                                    value = 100
                                    radius_eGFR = 0.12 + 0.25 * (value / 130)
                                    circ = patches.Circle((cx, cy), radius=radius_eGFR,
                                                          color="royalblue", alpha=0.85)
                                elif feat == "CASI (100)":
                                    circ = patches.Circle((cx, cy), radius=radius_CHOL,
                                                          color="blueviolet", alpha=0.85)
                                    label = f"{value:.0f}"
                                else:
                                    circ = patches.Circle((cx, cy), radius=radius_CHOL,
                                                          color="crimson" if (feat == "CHOL(mg/dL)" and value > 200)
                                                                             or (feat == "LDL-C(mg/dL)" and value > 130)
                                                                             or (feat == "eGFR" and value < 60)
                                                                             or (feat == "Na(mmol/L)" and (
                                                                      value > 145 or value < 136))
                                                          else "royalblue", alpha=0.85)
                                    label = f"{value:.0f}"
                            elif feat in ["HDL-C(mg/dL)", "ALT(U/L)", "AST(U/L)", "Hb(g/uL)", "CBZ(ug/mL)",
                                          "PTN(ug/mL)", "VALP(ug/mL)"]:
                                circ = patches.Circle((cx, cy), radius=radius_HDL,
                                                      color="crimson" if (feat == "HDL-C(mg/dL" and value < 40)
                                                                         or (feat == "ALT(U/L)" and value >= 40)
                                                                         or (feat == "AST(U/L)" and value >= 31)
                                                                         or (feat == "Hb(g/uL)" and (
                                                                  value < 11.0 or value > 17.2))
                                                                         or (feat == "CBZ(ug/mL)" and (
                                                                  value > 12 or value < 4))
                                                                         or (feat == "PTN(ug/mL)" and (
                                                                  value > 20 or value < 10))
                                                                         or (feat == "VALP(ug/mL)" and (
                                                                  value > 100 or value < 50))
                                                      else "royalblue", alpha=0.85)
                                label = f"{value:.0f}" if feat not in ["CBZ(ug/mL)", "PTN(ug/mL)",
                                                                       "VALP(ug/mL)"] else f"{value:.2f}"
                            elif feat in ["HbA1c(%)", "BUN(mg/dL)", "K(mmol/L)"]:
                                circ = patches.Circle((cx, cy), radius=radius_HbA1c,
                                                      color="crimson" if (feat == "HbA1c(%)" and (
                                                                  value > 6 or value < 4))
                                                                         or (feat == "BUN(mg/dL)" and (
                                                                  value < 7 or value > 20))
                                                                         or (feat == "K(mmol/L)" and (
                                                                  value > 5.1 or value < 3.5))
                                                      else "royalblue", alpha=0.85)
                                label = f"{value:.1f}"
                            elif feat in ["CREA(mg/dL)"]:
                                circ = patches.Circle((cx, cy), radius=radius_CREA,
                                                      color="crimson" if value < 0.7 or value > 1.4 else "royalblue",
                                                      alpha=0.85)
                                label = f"{value:.2f}"
                            elif feat in ["WBC(10^3/uL)", "RBC(10^3/uL)"]:
                                circ = patches.Circle((cx, cy), radius=radius,
                                                      color="crimson" if (feat == "WBC(10^3/uL)" and (
                                                                  value < 3.25 or value > 9.16))
                                                                         or (feat == "RBC(10^3/uL)" and (
                                                                  value < 3.78 or value > 5.9))
                                                      else "royalblue", alpha=0.85)
                                label = f"{value:.1f}"
                            elif feat in ["MMSE (30)"]:
                                temp = 0.12 + 0.25 * (value / 30)
                                circ = patches.Circle((cx, cy), radius=temp,
                                                      color="blueviolet", alpha=0.85)
                                label = f"{value:.0f}"
                            elif feat in ["CASI Q1 (10)",
                                          "CASI Q2 (12)",
                                          "CASI Q3 (8)",
                                          "CASI Q4 (10)",
                                          "CASI Q6 (12)",
                                          "CASI Q7 (10)",
                                          "CASI Q8 (10)",
                                          "CASI Q9 (10)", ]:
                                temp = 0.12 + 0.25 * (value / 10)
                                circ = patches.Circle((cx, cy), radius=temp,
                                                      color="blueviolet", alpha=0.85)
                                label = f"{value:.0f}"
                            elif feat in ["CASI Q5 (18)", ]:
                                temp = 0.12 + 0.25 * (value / 15)
                                circ = patches.Circle((cx, cy), radius=temp,
                                                      color="blueviolet", alpha=0.85)
                            elif feat in ["WAIS (100)"]:
                                temp = 0.12 + 0.25 * (value / 100)
                                circ = patches.Circle((cx, cy), radius=temp,
                                                      color="blueviolet", alpha=0.85)
                                label = f"{value:.0f}"
                            else:
                                circ = patches.Circle((cx, cy), radius=radius,
                                                      color="orange", alpha=0.85)
                                label = f"{value:.0f}"
                            ax.add_patch(circ)
                            ax.text(cx, cy, label, ha='center', va='center',
                                    fontsize=8, color="black")
                        else:
                            continue

            # --- LABELS ---
            ax.set_yticks(np.arange(n_rows) + 0.5)
            ax.set_yticklabels(rows, fontsize=11)

            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_xticklabels(df_source.columns, rotation=45, ha='right', fontsize=10)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(n_rows, 0)
            ax.set_aspect("equal")


        # =============================================================================
        # 7. CREATE FIGURE LAYOUT
        # =============================================================================
        num_groups = len(feature_groups)

        fig = plt.figure(figsize=(28, 14 + num_groups * 1.8), dpi=DPI)
        gs = fig.add_gridspec(
            nrows=num_groups,
            ncols=2,
            width_ratios=[1.7, 3.5],
            height_ratios=[3, 16, 5, 11, 15],
            wspace=0.2,
            hspace=0.45
        )

        # =============================================================================
        # 8. METADATA PANEL
        # =============================================================================
        # 使用子佈局切割左側欄位
        num_groups = len(feature_groups)
        fig = plt.figure(figsize=(28, 14 + num_groups * 1.8), dpi=DPI)
        gs = fig.add_gridspec(nrows=num_groups, ncols=2, width_ratios=[1.7, 3.5], wspace=0.2, hspace=0.30)

        # REVISED: 3-Part Left Panel Layout
        gs_left = gs[:, 0].subgridspec(4, 1, height_ratios=[1, 1.5, 0.8, 1.8], hspace=0.2)

        # Tier 1: Patient Information (Excluding headers starting with #)
        ax_meta = fig.add_subplot(gs_left[0])
        ax_meta.axis("off")
        ypos = 0.98
        ax_meta.text(0, ypos, "Patient Information", fontsize=26, fontweight="bold")
        for k, v in metadata.items():
            # Filter out #Metadata headers and fields visualized in checklists
            if k.startswith("#") or any(x in k for x in ["Etiology", "EEG", "Focal"]):
                continue
            ypos -= 0.07
            ax_meta.text(0, ypos, f"{k}:  {v}", fontsize=16)

        # Tier 2: Etiology Checklist
        ax_eti = fig.add_subplot(gs_left[1])
        plot_etiology_checklist_horizontal(ax_eti, active_eti_codes)

        # Tier 3: Seizure Type (Focal/Generalized)
        focal_raw_text = metadata.get("Focal", "No")
        ax_focal = fig.add_subplot(gs_left[2])
        plot_seizure_type_checklist(ax_focal, focal_raw_text)

        # Tier 4: EEG Focality Checklist (Detailed Electrodes)
        eeg_raw_text = metadata.get("EEG Focality", "No")
        ax_eeg_main = fig.add_subplot(gs_left[3])
        plot_eeg_focality_checklist(ax_eeg_main, eeg_raw_text)
        # =============================================================================
        # 9. PLOT EACH GROUP (USE df OR df_ddd DEPENDING ON GROUP)
        # =============================================================================
        for gi, (group_name, features) in enumerate(feature_groups.items()):
            ax = fig.add_subplot(gs[gi, 1])
            if group_name.startswith("ASM"):
                plot_feature_group(ax, df_ddd, features, True)
            elif group_name.startswith("Blood"):
                plot_feature_group(ax, df, features)
            else:
                plot_feature_group(ax, df, features, False)
            if group_name == "Blood Tests":
                ax.set_title("Blood Tests", fontsize=20)
                ax.title.set_y(0.95)
            else:
                ax.set_title(group_name, fontsize=20, pad=10)

        # =============================================================================
        # REVISED STATIC INFORMATION FIGURE (NON-TEMPORAL)
        # =============================================================================
        # 1. Increase figure height to 24 to fit the 4th section (Seizure Type)
        fig_static = plt.figure(figsize=(10, 24), dpi=DPI)

        # 2. Section 1: Patient Metadata (Top 20%)
        ax_meta_static = fig_static.add_axes([0.1, 0.78, 0.8, 0.18])  # [left, bottom, width, height]
        ax_meta_static.axis("off")
        ax_meta_static.set_xlim(0, 1)

        y_ptr = 0.95
        ax_meta_static.text(0, y_ptr, "Patient Information", fontsize=24, fontweight="bold")
        y_ptr -= 0.12
        for k, v in metadata.items():
            # Filter out headers and checklist-visualized fields
            if k.startswith("#") or any(x in k for x in ["Etiology", "EEG", "Focal"]):
                continue
            ax_meta_static.text(0, y_ptr, f"{k}:  {v}", fontsize=15)
            y_ptr -= 0.08

        # 3. Section 2: Etiology Checklist (25%)
        ax_eti_static = fig_static.add_axes([0.1, 0.50, 0.8, 0.25])
        plot_etiology_checklist_horizontal(ax_eti_static, active_eti_codes)

        # 4. NEW Section 3: Seizure Type Checklist (Focal/Gen) (10%)
        # Positioned between Etiology and EEG Focality
        focal_data = metadata.get("Focal", "No")
        ax_focal_static = fig_static.add_axes([0.1, 0.38, 0.8, 0.10])
        plot_seizure_type_checklist(ax_focal_static, focal_data)

        # 5. Section 4: EEG Focality Checklist (Bottom 33%)
        eeg_data = metadata.get("EEG Focality", "No")
        ax_eeg_static = fig_static.add_axes([0.1, 0.05, 0.8, 0.33])
        plot_eeg_focality_checklist(ax_eeg_static, eeg_data)

        # Save the integrated static report
        static_out = file_path.replace(".csv", "_Static_Information.png")
        plt.savefig(static_out, bbox_inches="tight", pad_inches=0.3, dpi = DPI)
        plt.close(fig_static)

        print(f"Static Report Saved: {static_out}\n")

        for group_name, features in feature_groups.items():
            fig_g, ax_g = plt.subplots(figsize=(20, 4), dpi=DPI)

            if group_name.startswith("ASM"):
                plot_feature_group(ax_g, df_ddd, features, True)
            elif group_name.startswith("Blood"):
                plot_feature_group(ax_g, df, features)
            else:
                plot_feature_group(ax_g, df, features, False)

            ax_g.set_title(group_name, fontsize=20, pad=10)

            filename = file_path.replace(".csv", f"_{group_name.replace(' ', '_')}.png")
            fig_g.savefig(filename, bbox_inches="tight", dpi = DPI)
            print(f"Saved: {filename}\n")
            plt.close(fig_g)

        combined_name = file_path.replace(".csv", "_Combined_timeline.png")
        plt.savefig(combined_name, dpi = 1000)
        print("Saved:\n", combined_name)

        # 釋放記憶體，確保執行多次不會卡頓
        plt.close('all')

    except Exception as e:
        print(f"處理檔案時發生錯誤: {e}")

print("程式已成功結束。")