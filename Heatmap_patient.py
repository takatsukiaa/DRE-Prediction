import argparse
import re
import sys
import textwrap
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilenames
from tkinter import messagebox, simpledialog

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import docx_parser_asm
import docx_parser_blood_test
import docx_parser_cognitive
import docx_parser_seizure
import docx_parser_static


DPI = 300
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

ASM_COLOR = "royalblue"
SEIZURE_COLOR = "orange"
LAB_COLOR = "royalblue"
LAB_ABNORMAL_COLOR = "crimson"
COGNITIVE_COLOR = "blueviolet"
COGNITIVE_FLAGGED_COLOR = "crimson"

LAB_FEATURES = [
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

LAB_ALIASES = {
    "WBC(10^3/uL)": ["WBC(10^3/uL)", "WBC (10^3/uL)"],
    "RBC(10^3/uL)": ["RBC(10^3/uL)", "RBC (10^3/uL)"],
    "Hb(g/uL)": ["Hb(g/uL)", "Hb (g/uL)"],
    "Na(mmol/L)": ["Na(mmol/L)", "Na (mmol/L)"],
    "K(mmol/L)": ["K(mmol/L)", "K (mmol/L)"],
    "BUN(mg/dL)": ["BUN(mg/dL)", "BUN (mg/dL)"],
    "CREA(mg/dL)": ["CREA(mg/dL)", "CREA (mg/dL)"],
    "eGFR": ["eGFR"],
    "AST(U/L)": ["AST(U/L)", "AST (U/L)"],
    "ALT(U/L)": ["ALT(U/L)", "ALT (U/L)"],
    "CHOL(mg/dL)": ["CHOL(mg/dL)", "CHOL (mg/dL)"],
    "LDL-C(mg/dL)": ["LDL-C(mg/dL)", "LDL-C (mg/dL)", "LD-C(mg/dL)", "LD-C (mg/dL)"],
    "HDL-C(mg/dL)": ["HDL-C(mg/dL)", "HDL-C (mg/dL)"],
    "HbA1c(%)": ["HbA1c(%)", "HbA1c (%)"],
    "CBZ(ug/mL)": ["CBZ(ug/mL)", "CBZ (ug/mL)"],
    "PTN(ug/mL)": ["PTN(ug/mL)", "PTN (ug/mL)"],
    "VALP(ug/mL)": ["VALP(ug/mL)", "VALP (ug/mL)"],
}

LAB_NORMAL_RANGES = {
    "WBC(10^3/uL)": (3.25, 9.16),
    "RBC(10^3/uL)": (3.78, 5.90),
    "Hb(g/uL)": (11.0, 17.2),
    "Na(mmol/L)": (136, 145),
    "K(mmol/L)": (3.5, 5.1),
    "BUN(mg/dL)": (7, 20),
    "CREA(mg/dL)": (0.7, 1.4),
    "eGFR": (60, None),
    "AST(U/L)": (None, 31),
    "ALT(U/L)": (None, 40),
    "CHOL(mg/dL)": (None, 200),
    "LDL-C(mg/dL)": (None, 130),
    "HDL-C(mg/dL)": (40, None),
    "HbA1c(%)": (4, 6),
    "CBZ(ug/mL)": (4, 12),
    "PTN(ug/mL)": (10, 20),
    "VALP(ug/mL)": (50, 100),
}

COGNITIVE_FEATURES = [
    "MMSE",
    "CASI",
    "CASI_Q1",
    "CASI_Q2",
    "CASI_Q3",
    "CASI_Q4",
    "CASI_Q5",
    "CASI_Q6",
    "CASI_Q7",
    "CASI_Q8",
    "CASI_Q9",
    "WAIS",
]

COGNITIVE_MAX_SCORES = {
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


def safe_print(text=""):
    encoding = sys.stdout.encoding or "utf-8"
    print(str(text).encode(encoding, errors="backslashreplace").decode(encoding))


def collect_docx_files(paths):
    files = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path_obj.glob("*.docx")
                    if not child.name.startswith("~$")
                )
            )
        elif path_obj.suffix.lower() == ".docx" and not path_obj.name.startswith("~$"):
            files.append(path_obj)
    return files


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("*", "")
    if text in {"", "-", "NA", "N/A", "nan", "None"}:
        return np.nan
    text = re.sub(r"^[<>]=?", "", text)
    text = re.sub(r"(\d+)\.\.+(\d+)", r"\1.\2", text)
    return pd.to_numeric(text, errors="coerce")


def format_value(value):
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def month_periods_from_index(index):
    dates = pd.to_datetime(index, errors="coerce")
    valid = dates.notna()
    if not valid.any():
        return pd.PeriodIndex([], freq="M")
    return dates[valid].to_period("M")


def make_month_axis(asm_df):
    if asm_df is None or asm_df.empty:
        raise ValueError("no ASM timeline")

    periods = month_periods_from_index(asm_df.index)
    if len(periods) == 0:
        raise ValueError("ASM timeline has no valid dates")
    return pd.period_range(periods.min(), periods.max(), freq="M")


def parse_month(value, label):
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


def apply_month_range(month_axis, start_month=None, end_month=None):
    start_period = parse_month(start_month, "Start month")
    end_period = parse_month(end_month, "End month")
    if start_period is not None and end_period is not None and start_period > end_period:
        raise ValueError("Start month cannot be after end month.")

    first_month = month_axis.min()
    last_month = month_axis.max()
    if start_period is not None:
        first_month = max(first_month, start_period)
    if end_period is not None:
        last_month = min(last_month, end_period)
    if first_month > last_month:
        raise ValueError(
            "Selected date range does not overlap the ASM timeline "
            f"({month_axis.min().strftime('%Y-%m')} to {month_axis.max().strftime('%Y-%m')})."
        )
    return pd.period_range(first_month, last_month, freq="M")


def month_columns(month_axis):
    return [month.strftime("%Y-%m") for month in month_axis]


def dataframe_to_monthly_matrix(df, month_axis, rows=None, aggregation="max"):
    if rows is None:
        rows = list(df.columns) if df is not None and not df.empty else []

    matrix = pd.DataFrame(0.0, index=rows, columns=month_columns(month_axis))
    if df is None or df.empty or not rows:
        return matrix

    working = df.copy()
    dates = pd.to_datetime(working.index, errors="coerce")
    working = working.loc[dates.notna()].copy()
    if working.empty:
        return matrix
    working["_month"] = dates[dates.notna()].to_period("M")

    for row_name in rows:
        if row_name not in working.columns:
            continue
        values = pd.to_numeric(working[row_name], errors="coerce")
        by_month = values.groupby(working["_month"]).max() if aggregation == "max" else values.groupby(working["_month"]).mean()
        for month, value in by_month.items():
            column = month.strftime("%Y-%m")
            if column in matrix.columns and pd.notna(value):
                matrix.loc[row_name, column] = float(value)
    return matrix


def build_asm_matrix(asm_df, month_axis):
    rows = list(asm_df.columns)
    return dataframe_to_monthly_matrix(asm_df, month_axis, rows=rows, aggregation="mean")


def build_seizure_matrix(seizure_df, month_axis):
    matrix = pd.DataFrame(
        0.0,
        index=["Seizure", "Seizure in the past 1 year"],
        columns=month_columns(month_axis),
    )
    if seizure_df is None or seizure_df.empty or "seizure_count_estimate" not in seizure_df.columns:
        return matrix

    working = seizure_df.copy()
    if "month" in working.columns:
        dates = pd.to_datetime(working["month"], format="%Y-%m", errors="coerce")
        dates = dates.fillna(pd.to_datetime(working["month"], errors="coerce"))
    else:
        dates = pd.to_datetime(working.index, errors="coerce")

    working = working.loc[dates.notna()].copy()
    if working.empty:
        return matrix
    working["_month"] = dates[dates.notna()].dt.to_period("M")
    values = pd.to_numeric(working["seizure_count_estimate"], errors="coerce").fillna(0)
    by_month = values.groupby(working["_month"]).max()
    annual_totals = by_month.groupby(by_month.index.year).sum()

    for month, value in by_month.items():
        column = month.strftime("%Y-%m")
        if column in matrix.columns:
            matrix.loc["Seizure", column] = float(value)

    for column in matrix.columns:
        year, month = column.split("-", 1)
        if month != "12":
            continue
        matrix.loc["Seizure in the past 1 year", column] = float(annual_totals.get(int(year), 0))
    return matrix


def find_column(df, aliases):
    normalized = {
        re.sub(r"\s+", "", str(column).lower()): column
        for column in df.columns
    }
    for alias in aliases:
        key = re.sub(r"\s+", "", alias.lower())
        if key in normalized:
            return normalized[key]
    return None


def build_lab_matrix(lab_result, month_axis):
    matrix = pd.DataFrame(0.0, index=LAB_FEATURES, columns=month_columns(month_axis))
    if lab_result is None:
        return matrix

    lab_frames = [df for df in lab_result if df is not None and not df.empty]
    for feature in LAB_FEATURES:
        aliases = LAB_ALIASES[feature]
        for df in lab_frames:
            source_column = find_column(df, aliases)
            if source_column is None:
                continue

            working = df[[source_column]].copy()
            dates = pd.to_datetime(working.index, errors="coerce")
            working = working.loc[dates.notna()].copy()
            if working.empty:
                continue
            working["_month"] = dates[dates.notna()].to_period("M")
            values = working[source_column].map(clean_numeric_value)
            by_month = values.groupby(working["_month"]).mean()
            for month, value in by_month.items():
                column = month.strftime("%Y-%m")
                if column in matrix.columns and pd.notna(value):
                    matrix.loc[feature, column] = float(value)
            break
    return matrix


def build_cognitive_matrices(cognitive_df, month_axis):
    matrix = pd.DataFrame(0.0, index=COGNITIVE_FEATURES, columns=month_columns(month_axis))
    flagged = pd.DataFrame(False, index=COGNITIVE_FEATURES, columns=month_columns(month_axis))
    if cognitive_df is None or cognitive_df.empty:
        return matrix, flagged

    working = cognitive_df.copy()
    dates = pd.to_datetime(working.index, errors="coerce")
    working = working.loc[dates.notna()].copy()
    if working.empty:
        return matrix, flagged
    working["_month"] = dates[dates.notna()].to_period("M")

    for feature in COGNITIVE_FEATURES:
        if feature not in working.columns:
            continue
        values = working[feature].map(clean_numeric_value)
        flags = working[feature].map(lambda value: "*" in str(value))
        by_month = values.groupby(working["_month"]).max()
        flags_by_month = flags.groupby(working["_month"]).max()
        for month, value in by_month.items():
            column = month.strftime("%Y-%m")
            if column in matrix.columns and pd.notna(value):
                matrix.loc[feature, column] = float(value)
                flagged.loc[feature, column] = bool(flags_by_month.get(month, False))
    return matrix, flagged


def lab_is_abnormal(feature, value):
    low, high = LAB_NORMAL_RANGES.get(feature, (None, None))
    if pd.isna(value):
        return False
    if low is not None and value < low:
        return True
    if high is not None and value > high:
        return True
    return False


def value_cap(matrix):
    values = matrix.to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0
    return max(1.0, float(np.nanpercentile(positive, 95)))


def generic_radius(value, cap):
    if pd.isna(value) or value <= 0:
        return 0
    scaled = min(float(value), cap) / cap
    return 0.12 + 0.28 * np.sqrt(scaled)


def cognitive_radius(feature, value):
    if pd.isna(value) or value <= 0:
        return 0
    maximum = COGNITIVE_MAX_SCORES.get(feature, 100)
    scaled = min(float(value), maximum) / maximum
    return 0.12 + 0.28 * np.sqrt(scaled)


def asm_radius(feature, value):
    if feature == "Total DDD":
        return min(0.44, 0.12 + 0.25 * (float(value) / 4))
    if feature == "Number of Drugs":
        return min(0.44, 0.12 + 0.25 * (float(value) / 6))
    return min(0.44, 0.12 + 0.25 * (float(value) / 2))


def tick_step(n_cols):
    if n_cols <= 36:
        return 1
    if n_cols <= 84:
        return 3
    if n_cols <= 180:
        return 6
    return 12


def draw_heatmap(
    matrix,
    output_path,
    title,
    color_for_cell,
    radius_for_cell=None,
    row_label_formatter=None,
    dpi=DPI,
):
    n_rows, n_cols = matrix.shape
    fig_width = max(12.0, min(72.0, 2.8 + n_cols * 0.18))
    fig_height = max(2.8, min(48.0, 1.8 + max(n_rows, 1) * 0.42))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.vlines(np.arange(n_cols + 1), 0, n_rows, color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.hlines(np.arange(n_rows + 1), 0, n_cols, color="#bdbdbd", linewidth=0.75, zorder=0)

    cap = value_cap(matrix)
    for row_idx, feature in enumerate(matrix.index):
        for col_idx, value in enumerate(matrix.iloc[row_idx]):
            if pd.isna(value) or value <= 0:
                continue
            cx = col_idx + 0.5
            cy = row_idx + 0.5
            radius = radius_for_cell(feature, value, cap) if radius_for_cell else generic_radius(value, cap)
            circle = patches.Circle(
                (cx, cy),
                radius=radius,
                color=color_for_cell(feature, value, row_idx, col_idx),
                alpha=0.86,
                zorder=2,
            )
            ax.add_patch(circle)
            ax.text(cx, cy, format_value(value), ha="center", va="center", fontsize=5, color="black", zorder=3)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=18, pad=12)

    labels = [row_label_formatter(row) if row_label_formatter else row for row in matrix.index]
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(labels, fontsize=8)

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


def wrap_static_value(value, width=70):
    text = str(value or "").strip()
    if not text:
        return ""
    return textwrap.fill(text, width=width, break_long_words=False)


def plot_static_data(static_data, output_path, title="Static Data", dpi=DPI):
    items = [(key, wrap_static_value(value)) for key, value in static_data.items()]
    if not items:
        items = [("Static data", "No static data extracted")]

    n_rows = len(items)
    fig_height = max(5.0, 1.4 + n_rows * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=dpi)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1)

    ax.text(0.02, n_rows + 0.55, title, fontsize=18, fontweight="bold", va="center")
    ax.text(0.02, n_rows + 0.05, "Field", fontsize=11, fontweight="bold", va="center")
    ax.text(0.34, n_rows + 0.05, "Value", fontsize=11, fontweight="bold", va="center")
    ax.hlines(n_rows - 0.2, 0.02, 0.98, color="#666666", linewidth=1.0)

    for idx, (key, value) in enumerate(items):
        y = n_rows - idx - 0.65
        if idx % 2 == 0:
            ax.add_patch(
                patches.Rectangle((0.02, y - 0.22), 0.96, 0.38, color="#f2f2f2", zorder=0)
            )
        ax.text(0.02, y, key, fontsize=10, fontweight="bold", va="center")
        ax.text(0.34, y, value, fontsize=10, va="center")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def save_category_plots(stem, output_dir, matrices):
    asm_matrix = matrices["asm"]
    seizure_matrix = matrices["seizure"]
    lab_matrix = matrices["lab"]
    cognitive_matrix, cognitive_flagged = matrices["cognitive"]
    static_data = matrices["static"]

    outputs = {}
    outputs["asm"] = output_dir / f"{stem}_asm_heatmap.png"
    draw_heatmap(
        asm_matrix,
        outputs["asm"],
        "ASM Timeline (DDD Units)",
        lambda _feature, _value, _row, _col: ASM_COLOR,
        radius_for_cell=lambda feature, value, _cap: asm_radius(feature, value),
    )

    outputs["seizure"] = output_dir / f"{stem}_seizure_heatmap.png"
    draw_heatmap(
        seizure_matrix,
        outputs["seizure"],
        "Seizure",
        lambda _feature, _value, _row, _col: SEIZURE_COLOR,
    )

    outputs["lab"] = output_dir / f"{stem}_lab_heatmap.png"
    draw_heatmap(
        lab_matrix,
        outputs["lab"],
        "Lab Tests",
        lambda feature, value, _row, _col: LAB_ABNORMAL_COLOR if lab_is_abnormal(feature, value) else LAB_COLOR,
    )

    outputs["cognitive"] = output_dir / f"{stem}_cognitive_heatmap.png"
    draw_heatmap(
        cognitive_matrix,
        outputs["cognitive"],
        "Cognitive Tests",
        lambda _feature, _value, row, col: (
            COGNITIVE_FLAGGED_COLOR if bool(cognitive_flagged.iloc[row, col]) else COGNITIVE_COLOR
        ),
        radius_for_cell=lambda feature, value, _cap: cognitive_radius(feature, value),
        row_label_formatter=lambda row: str(row).replace("_", " "),
    )

    outputs["static"] = output_dir / f"{stem}_static_data.png"
    plot_static_data(static_data, outputs["static"], title=f"{stem} Static Data")

    return outputs


def build_patient_matrices(docx_path, start_month=None, end_month=None):
    errors = []
    asm_df = docx_parser_asm.extract_asm_timeline_from_docx(str(docx_path))
    month_axis = apply_month_range(make_month_axis(asm_df), start_month=start_month, end_month=end_month)
    matrices = {"asm": build_asm_matrix(asm_df, month_axis)}

    try:
        seizure_df = docx_parser_seizure.extract_seizure_timeline(str(docx_path))
    except Exception as exc:
        seizure_df = None
        errors.append(("seizure", str(exc)))
    matrices["seizure"] = build_seizure_matrix(seizure_df, month_axis)

    try:
        lab_result = docx_parser_blood_test.extract_lab_timeline_from_docx(str(docx_path))
    except Exception as exc:
        lab_result = None
        errors.append(("lab", str(exc)))
    matrices["lab"] = build_lab_matrix(lab_result, month_axis)

    try:
        cognitive_df = docx_parser_cognitive.extract_cognitive_timeline_from_docx(str(docx_path))
    except Exception as exc:
        cognitive_df = None
        errors.append(("cognitive", str(exc)))
    matrices["cognitive"] = build_cognitive_matrices(cognitive_df, month_axis)

    try:
        static_data = docx_parser_static.extract_static_data_from_docx(str(docx_path))
    except Exception as exc:
        static_data = {}
        errors.append(("static", str(exc)))
    matrices["static"] = static_data

    return month_axis, matrices, errors


def process_patient(docx_path, output_root, start_month=None, end_month=None):
    docx_path = Path(docx_path)
    patient_dir = Path(output_root) / docx_path.stem
    patient_dir.mkdir(parents=True, exist_ok=True)

    month_axis, matrices, errors = build_patient_matrices(
        docx_path,
        start_month=start_month,
        end_month=end_month,
    )
    outputs = save_category_plots(docx_path.stem, patient_dir, matrices)
    return {
        "docx": str(docx_path),
        "output_dir": str(patient_dir),
        "months": len(month_axis),
        "outputs": outputs,
        "errors": errors,
    }


def process_inputs(paths, output_dir, start_month=None, end_month=None):
    files = collect_docx_files(paths)
    if not files:
        safe_print("No DOCX files found.")
        return []

    results = []
    skipped = []
    for index, docx_path in enumerate(files, start=1):
        safe_print(f"Processing {index}/{len(files)}: {docx_path.name}")
        try:
            result = process_patient(
                docx_path,
                output_dir,
                start_month=start_month,
                end_month=end_month,
            )
        except Exception as exc:
            skipped.append((docx_path.name, str(exc)))
            safe_print(f"SKIP {docx_path.name}: {exc}")
            continue

        results.append(result)
        safe_print(f"Saved {len(result['outputs'])} PNG files to {result['output_dir']}")
        if result["errors"]:
            for category, error in result["errors"]:
                safe_print(f"  {category} parser issue: {error}")

    safe_print(f"\nDone: {len(results)} patient(s) plotted, {len(skipped)} skipped.")
    if skipped:
        safe_print("Skipped:")
        for name, error in skipped:
            safe_print(f"  {name}: {error}")
    return results


def ask_month_range(root):
    while True:
        start_month = simpledialog.askstring(
            "Start month",
            "Enter first month to plot, such as 2017-01.\nLeave blank or Cancel to use each patient's oldest ASM month.",
            parent=root,
        )
        end_month = simpledialog.askstring(
            "End month",
            "Enter last month to plot, such as 2024-12.\nLeave blank or Cancel to use each patient's latest ASM month.",
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
        title="Select Patient DOCX Files",
        filetypes=[("DOCX files", "*.docx"), ("All files", "*.*")],
    )
    if not files:
        root.destroy()
        return [], "", None, None

    output_dir = askdirectory(title="Select Output Folder for Patient Heatmaps")
    if not output_dir:
        root.destroy()
        return list(files), "", None, None

    start_month, end_month = ask_month_range(root)
    root.destroy()
    return list(files), output_dir, start_month, end_month


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create per-patient ASM, seizure, lab, cognitive, and static-data PNGs "
            "directly from DOCX files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="DOCX files or folders containing DOCX files. If omitted, a file selector opens.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output folder. Defaults to ./patient_heatmaps when paths are provided.",
    )
    parser.add_argument(
        "--start-month",
        default=None,
        help="Optional first month to plot, for example 2017-01. Defaults to each patient's oldest ASM month.",
    )
    parser.add_argument(
        "--end-month",
        default=None,
        help="Optional last month to plot, for example 2024-12. Defaults to each patient's latest ASM month.",
    )
    args = parser.parse_args()

    paths = args.paths
    output_dir = args.output_dir
    start_month = args.start_month
    end_month = args.end_month

    if not paths:
        paths, output_dir, start_month, end_month = choose_gui_inputs()
        if not paths:
            safe_print("No DOCX files selected.")
            return
        if not output_dir:
            safe_print("No output folder selected.")
            return

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "patient_heatmaps"

    try:
        parse_month(start_month, "Start month")
        parse_month(end_month, "End month")
        if parse_month(start_month, "Start month") and parse_month(end_month, "End month"):
            if parse_month(start_month, "Start month") > parse_month(end_month, "End month"):
                raise ValueError("Start month cannot be after end month.")
    except ValueError as exc:
        safe_print(f"Invalid date range: {exc}")
        return

    process_inputs(paths, output_dir, start_month=start_month, end_month=end_month)


if __name__ == "__main__":
    main()
