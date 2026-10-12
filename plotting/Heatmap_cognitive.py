import re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilenames

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 300
COGNITIVE_COLOR = "blueviolet"
FLAGGED_COLOR = "crimson"


MAX_SCORES = {
    "MMSE": 30,
    "CASI": 100,
    "CASI_Q1": 10,
    "CASI_Q2": 12,
    "CASI_Q3": 8,
    "CASI_Q4": 10,
    "CASI_Q5": 18,
    "CASI_Q6": 12,
    "CASI_Q7": 10,
    "CASI_Q8": 10,
    "CASI_Q9": 10,
    "WAIS": 100,
}


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "NA", "N/A"}:
        return np.nan
    text = text.replace("*", "")
    return pd.to_numeric(text, errors="coerce")


def was_flagged(value):
    return "*" in str(value)


def display_label(feature_name):
    return feature_name.replace("_", " ")


def max_score_for(feature_name):
    return MAX_SCORES.get(feature_name, 100)


def format_value(value):
    if pd.isna(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def load_cognitive_csv(csv_path):
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    date_column = "Date" if "Date" in raw.columns else raw.columns[0]
    raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce")
    raw = raw.dropna(subset=[date_column])
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw = raw.set_index(date_column)
    raw.index = raw.index.strftime("%Y-%m-%d")

    score_columns = [
        column
        for column in raw.columns
        if column in MAX_SCORES or re.fullmatch(r"(MMSE|CASI|WAIS)_Q\d+", str(column))
    ]
    raw = raw[score_columns]

    numeric = raw.apply(lambda col: col.map(clean_numeric_value))
    flagged = raw.apply(lambda col: col.map(was_flagged))

    numeric = numeric.dropna(how="all", axis=0)
    numeric = numeric.dropna(how="all", axis=1)
    flagged = flagged.reindex(index=numeric.index, columns=numeric.columns).fillna(False)

    return numeric.T, flagged.T


def radius_for_score(feature_name, value):
    if pd.isna(value) or value <= 0:
        return 0
    maximum = max_score_for(feature_name)
    scaled = min(float(value), maximum) / maximum
    return 0.12 + 0.28 * np.sqrt(scaled)


def plot_cognitive_heatmap(matrix, flagged, output_path, title=None, dpi=DPI):
    if matrix.empty:
        raise ValueError("No numeric cognitive scores found.")

    n_rows, n_cols = matrix.shape
    fig_width = max(10.0, min(48.0, 2.5 + n_cols * 0.8))
    fig_height = max(5.0, min(36.0, 1.8 + n_rows * 0.38))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.vlines(np.arange(n_cols + 1), 0, n_rows, color="#d9d9d9", linewidth=0.7)
    ax.hlines(np.arange(n_rows + 1), 0, n_cols, color="#bdbdbd", linewidth=0.8)

    for row_idx, feature in enumerate(matrix.index):
        for col_idx, value in enumerate(matrix.iloc[row_idx]):
            if pd.isna(value) or value <= 0:
                continue

            cx = col_idx + 0.5
            cy = row_idx + 0.5
            color = FLAGGED_COLOR if bool(flagged.iloc[row_idx, col_idx]) else COGNITIVE_COLOR
            radius = radius_for_score(feature, value)
            circle = patches.Circle((cx, cy), radius=radius, color=color, alpha=0.85)
            ax.add_patch(circle)
            ax.text(cx, cy, format_value(value), ha="center", va="center", fontsize=7, color="black")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title or "Cognitive Test Timeline", fontsize=18, pad=12)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([display_label(name) for name in matrix.index], fontsize=8)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def output_name_for(csv_path):
    stem = Path(csv_path).stem
    return f"{stem}_cognitive_heatmap.png"


def plot_selected_files(csv_paths, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    errors = []
    for index, csv_path in enumerate(csv_paths, start=1):
        csv_path = Path(csv_path)
        print(f"Processing {index}/{len(csv_paths)}: {csv_path.name}")
        try:
            matrix, flagged = load_cognitive_csv(csv_path)
            output_path = output_dir / output_name_for(csv_path)
            plot_cognitive_heatmap(matrix, flagged, output_path, title=csv_path.stem, dpi=DPI)
            saved.append(output_path)
            print(f"Saved: {output_path}")
        except Exception as exc:
            errors.append((csv_path.name, str(exc)))
            print(f"ERROR {csv_path.name}: {exc}")

    print(f"\nDone: {len(saved)} saved, {len(errors)} error(s).")
    if errors:
        print("Errors:")
        for name, error in errors:
            print(f"  {name}: {error}")
    return saved, errors


def main():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    csv_paths = askopenfilenames(
        title="Select cognitive timeline CSV files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not csv_paths:
        print("No CSV files selected.")
        root.destroy()
        return

    output_dir = askdirectory(title="Select output folder for cognitive heatmap PNG files")
    root.destroy()
    if not output_dir:
        print("No output folder selected.")
        return

    plot_selected_files(list(csv_paths), output_dir)


if __name__ == "__main__":
    main()
