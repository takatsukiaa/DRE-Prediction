import os
from pathlib import Path
from tkinter import Button, Label, Radiobutton, StringVar, Tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter import messagebox, simpledialog

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import docx_parser_seizure


DPI = 600

COUNT_COLUMN = "seizure_count_estimate"
INPUT_MODE_DOCX = "docx"
INPUT_MODE_CSV = "csv"


def format_count(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def load_docx_timeline(docx_path):
    label = Path(docx_path).stem
    df = docx_parser_seizure.extract_seizure_timeline(docx_path)
    return label, dataframe_to_series(df, Path(docx_path).name)


def load_csv_timeline(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    label = Path(csv_path).stem.replace("_seizure_timeline", "")
    if "source_file" in df.columns:
        source_files = df["source_file"].dropna().astype(str)
        source_files = source_files[source_files.str.strip() != ""]
        if not source_files.empty:
            label = Path(source_files.iloc[0]).stem
    return label, dataframe_to_series(df, Path(csv_path).name)


def dataframe_to_series(df, source_name):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    timeline = df.copy()
    if COUNT_COLUMN not in timeline.columns:
        raise ValueError(f"{source_name} has no {COUNT_COLUMN} column.")

    if "month" in timeline.columns:
        month_values = pd.to_datetime(timeline["month"], format="%Y-%m", errors="coerce")
        if month_values.isna().any():
            fallback_months = pd.to_datetime(timeline["month"], errors="coerce")
            month_values = month_values.fillna(fallback_months)
    elif "Date" in timeline.columns:
        month_values = pd.to_datetime(timeline["Date"], errors="coerce")
    else:
        month_values = pd.to_datetime(timeline.index, errors="coerce")

    valid_months = month_values.notna()
    if not valid_months.any():
        return pd.Series(dtype=float)

    months = month_values[valid_months].dt.to_period("M")
    values = pd.to_numeric(timeline.loc[valid_months, COUNT_COLUMN], errors="coerce").fillna(0)
    series = pd.Series(values.to_numpy(dtype=float), index=months)
    return series.groupby(level=0).max()


def parse_start_month(value):
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
            raise ValueError("Start month must be a parseable month, for example 2017-01.")
        return parsed.to_period("M")


def timeline_loader(input_mode):
    if input_mode == INPUT_MODE_CSV:
        return load_csv_timeline
    if input_mode == INPUT_MODE_DOCX:
        return load_docx_timeline
    raise ValueError(f"Unsupported input mode: {input_mode}")


def build_plot_matrix(input_paths, start_month=None, input_mode=INPUT_MODE_DOCX, show_progress=False):
    start_period = parse_start_month(start_month)
    loader = timeline_loader(input_mode)
    rows = []
    all_months = []
    errors = []
    input_paths = list(input_paths)

    for index, input_path in enumerate(input_paths, start=1):
        if show_progress:
            print(f"Loading {index}/{len(input_paths)}: {Path(input_path).name}", flush=True)
        try:
            label, series = loader(input_path)
        except Exception as exc:
            label = Path(input_path).stem.replace("_seizure_timeline", "")
            series = pd.Series(dtype=float)
            errors.append((str(input_path), str(exc)))
        rows.append((label, series))
        all_months.extend(series.index.tolist())

    if not rows:
        raise ValueError("No input files were selected.")
    if not all_months:
        raise ValueError("No valid seizure months were found in the selected files.")

    first_month = min(all_months)
    last_month = max(all_months)
    if start_period is not None:
        if start_period > last_month:
            raise ValueError(
                f"Start month {start_period.strftime('%Y-%m')} is after the latest available month "
                f"{last_month.strftime('%Y-%m')}."
            )
        first_month = max(first_month, start_period)

    month_index = pd.period_range(first_month, last_month, freq="M")
    matrix = pd.DataFrame(
        0.0,
        index=[label for label, _series in rows],
        columns=[period.strftime("%Y-%m") for period in month_index],
    )

    for row_idx, (_label, series) in enumerate(rows):
        for month, value in series.items():
            column = month.strftime("%Y-%m")
            if column in matrix.columns:
                col_idx = matrix.columns.get_loc(column)
                matrix.iat[row_idx, col_idx] = value

    return matrix, errors


def tick_step(n_cols):
    if n_cols <= 36:
        return 1
    if n_cols <= 84:
        return 3
    if n_cols <= 180:
        return 6
    return 12


def draw_seizure_heatmap(matrix, output_path, dpi=DPI):
    n_rows, n_cols = matrix.shape
    values = matrix.to_numpy(dtype=float)
    positive = values[values > 0]
    value_cap = max(1.0, float(np.percentile(positive, 95))) if positive.size else 1.0

    fig_width = max(12.0, min(72.0, 2.8 + n_cols * 0.14))
    fig_height = max(6.0, min(48.0, 1.8 + n_rows * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.set_facecolor("white")
    ax.vlines(np.arange(n_cols + 1), 0, n_rows, color="#d9d9d9", linewidth=0.55, zorder=0)
    ax.hlines(np.arange(n_rows + 1), 0, n_cols, color="#bdbdbd", linewidth=0.75, zorder=0)

    label_size = 6 if n_rows > 80 or n_cols > 120 else 7
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = values[row_idx, col_idx]
            if value <= 0:
                continue

            scaled = min(value, value_cap) / value_cap
            radius = 0.11 + 0.29 * np.sqrt(scaled)
            cx = col_idx + 0.5
            cy = row_idx + 0.5
            circle = patches.Circle((cx, cy), radius=radius, color="orange", alpha=0.88, zorder=2)
            ax.add_patch(circle)
            ax.text(
                cx,
                cy,
                format_count(value),
                ha="center",
                va="center",
                fontsize=label_size,
                color="black",
                zorder=3,
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title("Seizure", fontsize=18, pad=12)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(matrix.index, fontsize=8)

    step = tick_step(n_cols)
    xtick_positions = np.arange(0, n_cols, step) + 0.5
    xtick_labels = [matrix.columns[idx] for idx in range(0, n_cols, step)]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def create_seizure_heatmap(
    input_paths,
    output_path,
    dpi=DPI,
    start_month=None,
    input_mode=INPUT_MODE_DOCX,
    show_progress=False,
):
    matrix, errors = build_plot_matrix(
        input_paths,
        start_month=start_month,
        input_mode=input_mode,
        show_progress=show_progress,
    )
    if show_progress:
        print("Rendering heatmap image...", flush=True)
    draw_seizure_heatmap(matrix, output_path, dpi=dpi)
    return matrix, errors


def bring_to_front(window):
    window.update_idletasks()
    window.lift()
    window.focus_force()
    window.attributes("-topmost", True)
    window.after(500, lambda: window.attributes("-topmost", False))


def ask_input_mode(root):
    root.deiconify()
    root.title("Seizure Heatmap Input Mode")
    root.resizable(False, False)
    root.geometry("360x180")
    bring_to_front(root)

    mode = StringVar(value=INPUT_MODE_DOCX)
    Label(root, text="Choose input file type", font=("Arial", 12, "bold"), padx=18, pady=12).pack(anchor="w")
    Radiobutton(root, text="DOCX files", variable=mode, value=INPUT_MODE_DOCX, padx=18, pady=6).pack(anchor="w")
    Radiobutton(
        root,
        text="Seizure timeline CSV files",
        variable=mode,
        value=INPUT_MODE_CSV,
        padx=18,
        pady=6,
    ).pack(anchor="w")

    selected = {"value": INPUT_MODE_DOCX}

    def confirm():
        selected["value"] = mode.get()
        root.quit()

    def cancel():
        selected["value"] = None
        root.quit()

    Button(root, text="OK", command=confirm, width=12).pack(pady=(8, 12))
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    for widget in root.winfo_children():
        widget.destroy()
    root.withdraw()
    return selected["value"]


def ask_start_month(root):
    while True:
        value = simpledialog.askstring(
            "Start month",
            "Enter first month to plot, such as 2017-01.\nLeave blank or Cancel to use the oldest extracted month.",
            parent=root,
        )
        try:
            return parse_start_month(value)
        except ValueError as exc:
            messagebox.showerror("Invalid start month", str(exc), parent=root)


def main():
    root = Tk()
    root.withdraw()

    print("Choose input mode in the dialog window.", flush=True)
    input_mode = ask_input_mode(root)
    if input_mode is None:
        print("No input mode selected. Exiting.")
        root.destroy()
        return

    if input_mode == INPUT_MODE_CSV:
        print("Select seizure timeline CSV files.", flush=True)
        input_paths = askopenfilenames(
            title="Select seizure timeline CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    else:
        print("Select DOCX files.", flush=True)
        input_paths = askopenfilenames(
            title="Select DOCX files for seizure timeline plot",
            filetypes=[("DOCX files", "*.docx"), ("All files", "*.*")],
        )

    if not input_paths:
        print("No input files selected. Exiting.")
        root.destroy()
        return

    start_month = ask_start_month(root)

    print("Choose where to save the PNG output.", flush=True)
    output_path = asksaveasfilename(
        title="Save seizure timeline plot",
        defaultextension=".png",
        initialfile="seizure_event_heatmap.png",
        filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
    )
    root.destroy()

    if not output_path:
        print("No output path selected. Exiting.")
        return

    print(f"Selected {len(input_paths)} {input_mode.upper()} files.")
    try:
        matrix, errors = create_seizure_heatmap(
            input_paths,
            output_path,
            dpi=DPI,
            start_month=start_month,
            input_mode=input_mode,
            show_progress=True,
        )
    except ValueError as exc:
        print(f"Could not create heatmap: {exc}")
        return
    print(f"Saved: {output_path}")
    print(f"Rows: {matrix.shape[0]}, months: {matrix.shape[1]}")

    if errors:
        print("Files with parser errors were retained as zero rows:")
        for input_path, error in errors:
            print(f"  {os.path.basename(input_path)}: {error}")


if __name__ == "__main__":
    main()
