import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import docx_parser_asm
from tkinter.filedialog import askopenfilename
from tkinter import Tk

DPI = 300


def plot_asm_heatmap(ax, df):
    """
    Plots the ASM DDD timeline as a heatmap grid with proportionally-sized circles.

    Parameters
    ----------
    ax : matplotlib Axes
    df : DataFrame with dates as rows and drug names as columns, values in DDD units.
         Expected to also have a 'Total DDD' column.
    """
    drugs = list(df.columns)
    dates = [d.strftime("%Y-%m") for d in df.index]

    n_rows = len(drugs)
    n_cols = len(dates)

    # Grid lines
    for r in range(n_rows + 1):
        ax.plot([0, n_cols], [r, r], color="lightgray", linewidth=0.8)
    for c in range(n_cols + 1):
        ax.plot([c, c], [0, n_rows], color="lightgray", linewidth=0.8)

    # Draw circles — dates along x-axis, drugs along y-axis
    for i, drug in enumerate(drugs):
        for j, (_, row) in enumerate(df.iterrows()):
            value = row[drug]
            if pd.isna(value) or value <= 0:
                continue
            cx = j + 0.5
            cy = i + 0.5
            radius = 0.12 + 0.25 * (value / 2) if drug != "Total DDD" else 0.12 + 0.25 * (value / 4)
            circ = patches.Circle((cx, cy), radius=radius, color="royalblue", alpha=0.85)
            ax.add_patch(circ)
            ax.text(cx, cy, f"{value:.2f}", ha='center', va='center', fontsize=8, color="black")

    # Axis labels
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(drugs, fontsize=11)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=10)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title("ASM Timeline (DDD Units)", fontsize=20, pad=10)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Select Patient Docx File",
        filetypes=[("Docx files", "*.docx"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        root.destroy()
        exit()

    root.destroy()
    print(f"Selected file: {file_path}")

    df = docx_parser_asm.table_selection(file_path)
    if df is None:
        print("No ASM table found in the document. Exiting.")
        exit(1)

    # Add Total DDD column
    df["Total DDD"] = df.sum(axis=1)

    # Figure sizing
    n_drugs = len(df.columns)
    n_dates = len(df)
    fig_h = max(6, n_drugs * 0.6)
    fig_w = max(12, n_dates * 0.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    plot_asm_heatmap(ax, df)

    base_name = os.path.basename(file_path)
    out_path = base_name.replace(".docx", "_ASM_Heatmap.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    print(f"Saved: {out_path}")

    plt.close("all")
    print("Done.")
