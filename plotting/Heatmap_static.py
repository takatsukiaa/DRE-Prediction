import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from tkinter.filedialog import askopenfilename
from tkinter import Tk

DPI = 300
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

def main():
    parser_dir = str(Path(__file__).resolve().parents[1] / "docx_parsers")
    if parser_dir not in sys.path:
        sys.path.insert(0, parser_dir)
    from docx_parser_static import initial_table_selection

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Select Patient Docx File (Cancel to Exit)",
        filetypes=[("Docx files", "*.Docx"), ("All files", "*.*")]
    )
    root.destroy()
    if not file_path:
        return
    static_data = initial_table_selection(file_path)

    fig_static = plt.figure(figsize=(10, 24), dpi=DPI)

    # 2. Section 1: Patient Metadata (Top 20%)
    ax_meta_static = fig_static.add_axes([0.1, 0.78, 0.8, 0.18])  # [left, bottom, width, height]
    ax_meta_static.axis("off")
    ax_meta_static.set_xlim(0, 1)

    y_ptr = 0.95
    ax_meta_static.text(0, y_ptr, "Patient Information", fontsize=24, fontweight="bold")
    y_ptr -= 0.12

    for k, v in static_data.items():
        if k!= "Etiology" and k!= "Focal" and k!= "EEG Focality":
            ax_meta_static.text(0, y_ptr, f"{k}:  {v}", fontsize=15)
            y_ptr -= 0.08

    ax_eti_static = fig_static.add_axes([0.1, 0.50, 0.8, 0.25])
    active_eti_codes = static_data["Etiology"].split(',')
    for i in range(len(active_eti_codes)):
        active_eti_codes[i] = int(active_eti_codes[i])
    plot_etiology_checklist_horizontal(ax_eti_static, active_eti_codes)

    ax_focal_static = fig_static.add_axes([0.1, 0.38, 0.8, 0.10])
    plot_seizure_type_checklist(ax_focal_static, static_data["Focal"])

    ax_eeg_static = fig_static.add_axes([0.1, 0.05, 0.8, 0.33])
    plot_eeg_focality_checklist(ax_eeg_static, static_data["EEG Focality"])

    file_path = os.path.basename(file_path)
    static_out = file_path.replace(".docx", "_Static_Information.png")
    plt.savefig(static_out, bbox_inches="tight", pad_inches=0.3, dpi = DPI)
    plt.close(fig_static)

    print(f"Static Report Saved: {static_out}\n")


if __name__ == "__main__":
    main()
