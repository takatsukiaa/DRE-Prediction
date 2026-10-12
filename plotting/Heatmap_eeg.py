import argparse
from pathlib import Path
from tkinter import Tk
from tkinter import messagebox, simpledialog
from tkinter.filedialog import askdirectory, askopenfilenames

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import docx_parser_eeg_timeline


DPI = 300
SLOW_COLOR = "gold"
ED_COLOR = "limegreen"
UNSPECIFIED_COLOR = "lightslategray"
UNSPECIFIED_BURDEN_VALUE = 0.5
UNSPECIFIED_RADIUS = 0.22

GLOBAL_SLOW_ROWS = ["Delta waves", "Theta waves"]
GLOBAL_ED_ROWS = ["Spike ED", "Sharp ED", "Sharpish configuration", "Spike-and-wave ED", "Periodic Discharge"]

SLOW_REGION_ROWS = [
    ("Temporal Slow-wave", "TL", "TR"),
    ("Frontal Slow-wave", "FL", "FR"),
    ("Occipital Slow-wave", "OL", "OR"),
    ("Parietal Slow-wave", "PL", "PR"),
    ("Central Slow-wave", "CL", "CR"),
]

ED_REGION_ROWS = [
    ("Temporal ED", "TLED", "TRED"),
    ("Frontal ED", "FLED", "FRED"),
    ("Occipital ED", "OLED", "ORED"),
    ("Parietal ED", "PLED", "PRED"),
    ("Central ED", "CLED", "CRED"),
]

BASE_EEG_COLUMNS = (
    GLOBAL_SLOW_ROWS
    + GLOBAL_ED_ROWS
    + [column for _label, left, right in SLOW_REGION_ROWS + ED_REGION_ROWS for column in (left, right)]
)

PLOT_ROWS = (
    GLOBAL_SLOW_ROWS
    + [label for label, _left, _right in SLOW_REGION_ROWS]
    + GLOBAL_ED_ROWS
    + [label for label, _left, _right in ED_REGION_ROWS]
)


def safe_print(text=""):
    print(str(text))


def clean_numeric(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if text in {"", "-", "NA", "N/A", "nan", "None"}:
        return 0.0
    parsed = pd.to_numeric(text, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    return float(parsed)


def parse_month(value, label="Month"):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return pd.Period(value, freq="M")
    except ValueError:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{label} must be a parseable month, for example 2017-01.")
        return parsed.to_period("M")


def apply_month_range(months, start_month=None, end_month=None):
    start_period = parse_month(start_month, "Start month")
    end_period = parse_month(end_month, "End month")
    if start_period is not None and end_period is not None and start_period > end_period:
        raise ValueError("Start month cannot be after end month.")

    first_month = min(months)
    last_month = max(months)
    if start_period is not None:
        first_month = max(first_month, start_period)
    if end_period is not None:
        last_month = min(last_month, end_period)
    if first_month > last_month:
        raise ValueError(
            "Selected date range does not overlap the EEG timeline "
            f"({min(months).strftime('%Y-%m')} to {max(months).strftime('%Y-%m')})."
        )
    return pd.period_range(first_month, last_month, freq="M")


def month_columns(month_axis):
    return [period.strftime("%Y-%m") for period in month_axis]


def collect_input_files(paths):
    files = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path_obj.iterdir()
                    if child.suffix.lower() in {".csv", ".docx"} and not child.name.startswith("~$")
                )
            )
        elif path_obj.suffix.lower() in {".csv", ".docx"} and not path_obj.name.startswith("~$"):
            files.append(path_obj)
    return files


def label_for_path(path):
    stem = Path(path).stem
    return stem.replace("_eeg_timeline", "")


def load_csv_timeline(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return label_for_path(csv_path), df


def load_docx_timeline(docx_path):
    df = docx_parser_eeg_timeline.table_selection(str(docx_path))
    return Path(docx_path).stem, df


def load_eeg_timeline(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return load_csv_timeline(path)
    if path.suffix.lower() == ".docx":
        return load_docx_timeline(path)
    raise ValueError(f"Unsupported EEG input file type: {path.suffix}")


def build_eeg_matrix(timeline, start_month=None, end_month=None):
    if timeline is None or timeline.empty:
        raise ValueError("No EEG timeline rows found.")

    working = timeline.copy()
    if "month" in working.columns:
        parsed_months = pd.to_datetime(working["month"], format="%Y-%m", errors="coerce")
        parsed_months = parsed_months.fillna(pd.to_datetime(working["month"], errors="coerce"))
    elif "report_date" in working.columns:
        parsed_months = pd.to_datetime(working["report_date"], errors="coerce")
    else:
        parsed_months = pd.to_datetime(working.index, errors="coerce")

    working = working.loc[parsed_months.notna()].copy()
    if working.empty:
        raise ValueError("No parseable EEG months found.")

    working["_month"] = parsed_months[parsed_months.notna()].dt.to_period("M")
    month_axis = apply_month_range(list(working["_month"]), start_month=start_month, end_month=end_month)

    matrix = pd.DataFrame(0.0, index=list(BASE_EEG_COLUMNS), columns=month_columns(month_axis))
    for feature in BASE_EEG_COLUMNS:
        if feature not in working.columns:
            continue
        values = working[feature].map(clean_numeric)
        by_month = values.groupby(working["_month"]).max()
        for month, value in by_month.items():
            column = month.strftime("%Y-%m")
            if column in matrix.columns and pd.notna(value):
                matrix.loc[feature, column] = float(value)

    for label, left_column, right_column in SLOW_REGION_ROWS + ED_REGION_ROWS:
        matrix.loc[label] = np.maximum(matrix.loc[left_column], matrix.loc[right_column])

    return matrix


def tick_step(n_cols):
    if n_cols <= 36:
        return 1
    if n_cols <= 84:
        return 3
    if n_cols <= 180:
        return 6
    return 12


def burden_label(value):
    if value <= UNSPECIFIED_BURDEN_VALUE:
        return "U"
    if value <= 1:
        return "R"
    if value <= 2:
        return "In"
    if value <= 3:
        return "F"
    return "C"


def draw_lr_proportion(ax, row_idx, col_idx, left_value, right_value, color):
    total = float(left_value) + float(right_value)
    if total <= 0:
        return

    left_ratio = float(left_value) / total
    right_ratio = float(right_value) / total

    if left_ratio > 0:
        ax.add_patch(
            patches.Rectangle((col_idx, row_idx), left_ratio, 1.0, color=color, alpha=0.88, zorder=2)
        )
        ax.text(
            col_idx + left_ratio / 2,
            row_idx + 0.5,
            "L",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=3,
        )

    if left_ratio > 0 and right_ratio > 0:
        ax.plot(
            [col_idx + left_ratio, col_idx + left_ratio],
            [row_idx, row_idx + 1],
            color="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=4,
        )

    if right_ratio > 0:
        ax.add_patch(
            patches.Rectangle(
                (col_idx + left_ratio, row_idx), right_ratio, 1.0, color=color, alpha=0.88, zorder=2
            )
        )
        ax.text(
            col_idx + left_ratio + right_ratio / 2,
            row_idx + 0.5,
            "R",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=3,
        )


def draw_burden_circle(ax, row_idx, col_idx, value, color):
    if pd.isna(value) or value <= 0:
        return
    cx = col_idx + 0.5
    cy = row_idx + 0.5
    numeric_value = float(value)
    if numeric_value <= UNSPECIFIED_BURDEN_VALUE:
        radius = UNSPECIFIED_RADIUS
        color = UNSPECIFIED_COLOR
    else:
        radius = 0.12 + 0.25 * (numeric_value / 3)
    circle = patches.Circle((cx, cy), radius=radius, color=color, alpha=0.85, zorder=2)
    ax.add_patch(circle)
    ax.text(cx, cy, burden_label(numeric_value), ha="center", va="center", color="black", fontsize=8, zorder=3)


def plot_eeg_heatmap(matrix, output_path, title=None, dpi=DPI):
    if matrix.empty:
        raise ValueError("No EEG values available for plotting.")

    display_rows = [row for row in PLOT_ROWS if row in matrix.index]
    n_rows = len(display_rows)
    n_cols = matrix.shape[1]
    fig_width = max(12.0, min(72.0, 2.8 + n_cols * 0.18))
    fig_height = max(5.0, min(48.0, 1.8 + n_rows * 0.42))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.vlines(np.arange(n_cols + 1), 0, n_rows, color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.hlines(np.arange(n_rows + 1), 0, n_cols, color="#bdbdbd", linewidth=0.75, zorder=0)

    slow_region_map = {label: (left, right) for label, left, right in SLOW_REGION_ROWS}
    ed_region_map = {label: (left, right) for label, left, right in ED_REGION_ROWS}

    for row_idx, feature in enumerate(display_rows):
        for col_idx, value in enumerate(matrix.loc[feature]):
            if feature in slow_region_map:
                left_column, right_column = slow_region_map[feature]
                draw_lr_proportion(
                    ax,
                    row_idx,
                    col_idx,
                    matrix.loc[left_column, matrix.columns[col_idx]],
                    matrix.loc[right_column, matrix.columns[col_idx]],
                    SLOW_COLOR,
                )
            elif feature in ed_region_map:
                left_column, right_column = ed_region_map[feature]
                draw_lr_proportion(
                    ax,
                    row_idx,
                    col_idx,
                    matrix.loc[left_column, matrix.columns[col_idx]],
                    matrix.loc[right_column, matrix.columns[col_idx]],
                    ED_COLOR,
                )
            elif feature in GLOBAL_SLOW_ROWS:
                draw_burden_circle(ax, row_idx, col_idx, value, SLOW_COLOR)
            elif feature in GLOBAL_ED_ROWS:
                draw_burden_circle(ax, row_idx, col_idx, value, ED_COLOR)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title or "EEG Timeline", fontsize=18, pad=12)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(display_rows, fontsize=8)

    step = tick_step(n_cols)
    tick_positions = np.arange(0, n_cols, step) + 0.5
    tick_labels = [matrix.columns[idx] for idx in range(0, n_cols, step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def output_name_for(path):
    return f"{label_for_path(path)}_eeg_heatmap.png"


def plot_selected_files(input_paths, output_dir, start_month=None, end_month=None, dpi=DPI):
    files = collect_input_files(input_paths)
    if not files:
        safe_print("No EEG CSV or DOCX files found.")
        return [], []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    errors = []
    for index, input_path in enumerate(files, start=1):
        safe_print(f"Processing {index}/{len(files)}: {input_path.name}")
        try:
            label, timeline = load_eeg_timeline(input_path)
            matrix = build_eeg_matrix(timeline, start_month=start_month, end_month=end_month)
            output_path = output_dir / output_name_for(input_path)
            plot_eeg_heatmap(matrix, output_path, title=f"{label} EEG Timeline", dpi=dpi)
            saved.append(output_path)
            safe_print(f"Saved: {output_path}")
        except Exception as exc:
            errors.append((input_path.name, str(exc)))
            safe_print(f"ERROR {input_path.name}: {exc}")

    safe_print(f"\nDone: {len(saved)} saved, {len(errors)} error(s).")
    if errors:
        safe_print("Errors:")
        for name, error in errors:
            safe_print(f"  {name}: {error}")
    return saved, errors


def ask_month_range(root):
    while True:
        start_month = simpledialog.askstring(
            "Start month",
            "Enter first month to plot, such as 2017-01.\nLeave blank or Cancel to use each EEG timeline's oldest month.",
            parent=root,
        )
        end_month = simpledialog.askstring(
            "End month",
            "Enter last month to plot, such as 2024-12.\nLeave blank or Cancel to use each EEG timeline's latest month.",
            parent=root,
        )
        try:
            parse_month(start_month, "Start month")
            parse_month(end_month, "End month")
            if parse_month(start_month, "Start month") and parse_month(end_month, "End month"):
                if parse_month(start_month, "Start month") > parse_month(end_month, "End month"):
                    raise ValueError("Start month cannot be after end month.")
            return start_month, end_month
        except ValueError as exc:
            messagebox.showerror("Invalid date range", str(exc), parent=root)


def choose_gui_inputs():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    files = askopenfilenames(
        title="Select EEG timeline CSV or patient DOCX files",
        filetypes=[
            ("EEG CSV or DOCX files", "*.csv *.docx"),
            ("CSV files", "*.csv"),
            ("DOCX files", "*.docx"),
            ("All files", "*.*"),
        ],
    )
    if not files:
        root.destroy()
        return [], "", None, None

    output_dir = askdirectory(title="Select output folder for EEG heatmap PNG files")
    if not output_dir:
        root.destroy()
        return list(files), "", None, None

    start_month, end_month = ask_month_range(root)
    root.destroy()
    return list(files), output_dir, start_month, end_month


def main():
    parser = argparse.ArgumentParser(
        description="Create EEG timeline heatmap PNGs from parser CSV files or patient DOCX files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="EEG timeline CSV files, DOCX files, or folders containing CSV/DOCX files. If omitted, a file selector opens.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output folder. Defaults to ./new_plots/eeg_heatmap when paths are provided.",
    )
    parser.add_argument(
        "--start-month",
        default=None,
        help="Optional first month to plot, for example 2017-01.",
    )
    parser.add_argument(
        "--end-month",
        default=None,
        help="Optional last month to plot, for example 2024-12.",
    )
    parser.add_argument("--dpi", type=int, default=DPI, help=f"Output DPI. Default: {DPI}.")
    args = parser.parse_args()

    paths = args.paths
    output_dir = args.output_dir
    start_month = args.start_month
    end_month = args.end_month

    if not paths:
        paths, output_dir, start_month, end_month = choose_gui_inputs()
        if not paths:
            safe_print("No EEG files selected.")
            return
        if not output_dir:
            safe_print("No output folder selected.")
            return

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "new_plots" / "eeg_heatmap"

    try:
        parse_month(start_month, "Start month")
        parse_month(end_month, "End month")
        if parse_month(start_month, "Start month") and parse_month(end_month, "End month"):
            if parse_month(start_month, "Start month") > parse_month(end_month, "End month"):
                raise ValueError("Start month cannot be after end month.")
    except ValueError as exc:
        safe_print(f"Invalid date range: {exc}")
        return

    plot_selected_files(paths, output_dir, start_month=start_month, end_month=end_month, dpi=args.dpi)


if __name__ == "__main__":
    main()
