import re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilenames

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 300
LAB_COLOR = "royalblue"
ABNORMAL_COLOR = "crimson"


NORMAL_RANGES = {
    "WBC": (3.25, 9.16),
    "RBC": (3.78, 5.90),
    "HB": (11.0, 17.2),
    "NA": (136, 145),
    "K": (3.5, 5.1),
    "BUN": (7, 20),
    "CREA": (0.7, 1.4),
    "EGFR": (60, None),
    "AST": (None, 31),
    "ALT": (None, 40),
    "CHOL": (None, 200),
    "LD": (None, 130),
    "HDL": (40, None),
    "HBA1C": (4, 6),
    "CBZ": (4, 12),
    "PTN": (10, 20),
    "VALP": (50, 100),
}


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("*", "")
    if text in {"", "-"}:
        return np.nan
    text = re.sub(r"^[<>]=?", "", text)
    text = re.sub(r"(\d+)\.\.+(\d+)", r"\1.\2", text)
    return pd.to_numeric(text, errors="coerce")


def normalize_feature_name(name):
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def matching_range(feature_name):
    normalized = normalize_feature_name(feature_name)
    for key, value_range in NORMAL_RANGES.items():
        if key in normalized:
            return value_range
    return None


def is_abnormal(feature_name, value):
    value_range = matching_range(feature_name)
    if value_range is None or pd.isna(value):
        return False

    low, high = value_range
    if low is not None and value < low:
        return True
    if high is not None and value > high:
        return True
    return False


def format_value(value):
    if pd.isna(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def load_lab_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame()

    date_column = "Date" if "Date" in df.columns else df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    if df.empty:
        return pd.DataFrame()

    df = df.set_index(date_column)
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.apply(lambda col: col.map(clean_numeric_value))
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    return df.T


def radius_for_value(value, positive_values):
    if pd.isna(value) or value <= 0:
        return 0
    cap = max(1.0, float(np.nanpercentile(positive_values, 95))) if positive_values.size else 1.0
    scaled = min(float(value), cap) / cap
    return 0.12 + 0.28 * np.sqrt(scaled)


def plot_lab_heatmap(matrix, output_path, title=None, dpi=DPI):
    if matrix.empty:
        raise ValueError("No numeric lab values found.")

    n_rows, n_cols = matrix.shape
    values = matrix.to_numpy(dtype=float)
    positive_values = values[np.isfinite(values) & (values > 0)]

    fig_width = max(10.0, min(48.0, 2.5 + n_cols * 0.65))
    fig_height = max(5.0, min(36.0, 1.8 + n_rows * 0.42))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.vlines(np.arange(n_cols + 1), 0, n_rows, color="#d9d9d9", linewidth=0.7)
    ax.hlines(np.arange(n_rows + 1), 0, n_cols, color="#bdbdbd", linewidth=0.8)

    for row_idx, feature in enumerate(matrix.index):
        for col_idx, value in enumerate(matrix.iloc[row_idx]):
            if pd.isna(value) or value <= 0:
                continue
            cx = col_idx + 0.5
            cy = row_idx + 0.5
            color = ABNORMAL_COLOR if is_abnormal(feature, value) else LAB_COLOR
            radius = radius_for_value(value, positive_values)
            circle = patches.Circle((cx, cy), radius=radius, color=color, alpha=0.85)
            ax.add_patch(circle)
            ax.text(cx, cy, format_value(value), ha="center", va="center", fontsize=7, color="black")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title or "Lab Test Timeline", fontsize=18, pad=12)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(matrix.index, fontsize=8)
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
    return f"{stem}_lab_heatmap.png"


def plot_selected_files(csv_paths, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    errors = []
    for index, csv_path in enumerate(csv_paths, start=1):
        csv_path = Path(csv_path)
        print(f"Processing {index}/{len(csv_paths)}: {csv_path.name}")
        try:
            matrix = load_lab_csv(csv_path)
            output_path = output_dir / output_name_for(csv_path)
            plot_lab_heatmap(matrix, output_path, title=csv_path.stem, dpi=DPI)
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
        title="Select lab timeline CSV files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not csv_paths:
        print("No CSV files selected.")
        root.destroy()
        return

    output_dir = askdirectory(title="Select output folder for lab heatmap PNG files")
    root.destroy()
    if not output_dir:
        print("No output folder selected.")
        return

    plot_selected_files(list(csv_paths), output_dir)


if __name__ == "__main__":
    main()
