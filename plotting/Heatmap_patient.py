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
import docx_parser_eeg_timeline
import docx_parser_seizure
import docx_parser_static
from structured_features import extract_structured_features


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
EEG_SLOW_COLOR = "gold"
EEG_ED_COLOR = "limegreen"
EEG_UNSPECIFIED_COLOR = "lightslategray"
EEG_UNSPECIFIED_BURDEN_VALUE = 0.5
EEG_UNSPECIFIED_RADIUS = 0.14

# Use most of each heatmap cell so clinically meaningful differences are easy
# to see.  A power greater than one avoids making small values look too similar
# to large values (the previous square-root scaling did the opposite).
CONTRAST_MIN_RADIUS = 0.09
CONTRAST_MAX_RADIUS = 0.46
CONTRAST_EXPONENT = 1.35

ETIOLOGY_LIST = [
    "Hypoxia at birth",
    "Developmental delay",
    "Cerebral palsy",
    "Family history of epilepsy",
    "Cerebral structure lesion",
    "Cerebral trauma",
    "Cerebral tumor",
    "Cerebral infection",
    "Cerebrovascular disorders",
    "Cerebral immunologic disorders",
    "Degenerative and other neurological conditions",
    "Hydrocephalus",
    "Brain Surgery",
    "Encephalitis",
]

EEG_FOCALITY_GROUPS = {
    "Frontal": ["Fp1", "F3", "F7", "Fp2", "F4", "F8", "Fz"],
    "Temporal": ["T3", "T4", "T5", "T6", "S1", "S2"],
    "Central": ["C3", "C4", "Cz"],
    "Parietal": ["P3", "P4", "Pz"],
    "Occipital": ["O1", "O2"],
}

FOCAL_TYPE_CODES = ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "4"]

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

EEG_GLOBAL_SLOW_ROWS = ["Delta waves", "Theta waves"]
EEG_GLOBAL_ED_ROWS = ["Spike ED", "Sharp ED", "Sharpish configuration", "Spike-and-wave ED", "Periodic Discharge"]
EEG_SLOW_REGION_ROWS = [
    ("Temporal Slow-wave", "TL", "TR"),
    ("Frontal Slow-wave", "FL", "FR"),
    ("Occipital Slow-wave", "OL", "OR"),
    ("Parietal Slow-wave", "PL", "PR"),
    ("Central Slow-wave", "CL", "CR"),
]
EEG_ED_REGION_ROWS = [
    ("Temporal ED", "TLED", "TRED"),
    ("Frontal ED", "FLED", "FRED"),
    ("Occipital ED", "OLED", "ORED"),
    ("Parietal ED", "PLED", "PRED"),
    ("Central ED", "CLED", "CRED"),
]
EEG_BASE_ROWS = (
    EEG_GLOBAL_SLOW_ROWS
    + EEG_GLOBAL_ED_ROWS
    + [row for _label, left, right in EEG_SLOW_REGION_ROWS + EEG_ED_REGION_ROWS for row in (left, right)]
)
EEG_PLOT_ROWS = (
    EEG_GLOBAL_SLOW_ROWS
    + [label for label, _left, _right in EEG_SLOW_REGION_ROWS]
    + EEG_GLOBAL_ED_ROWS
    + [label for label, _left, _right in EEG_ED_REGION_ROWS]
)


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


def parse_month_count(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        month_count = int(text)
    except ValueError as exc:
        raise ValueError("Months to plot must be a positive whole number.") from exc
    if month_count < 1:
        raise ValueError("Months to plot must be at least 1.")
    return month_count


def parse_month_direction(value):
    if value is None:
        return "first"
    direction = str(value).strip().lower()
    if direction not in {"first", "last"}:
        raise ValueError("Month direction must be either 'first' or 'last'.")
    return direction


def apply_month_range(
    month_axis,
    start_month=None,
    end_month=None,
    months_to_plot=None,
    month_direction="first",
):
    start_period = parse_month(start_month, "Start month")
    end_period = parse_month(end_month, "End month")
    month_count = parse_month_count(months_to_plot)
    direction = parse_month_direction(month_direction)
    if start_period is not None and end_period is not None and start_period > end_period:
        raise ValueError("Start month cannot be after end month.")

    # Explicit bounds define the requested observation window even when the
    # ASM table has no entry in an early/late month. This is required for
    # patient-specific clinical anchors such as the first Dr. Wu visit; the
    # matrix builders will represent months without observations as zero.
    first_month = start_period if start_period is not None else month_axis.min()
    last_month = end_period if end_period is not None else month_axis.max()
    if first_month > last_month:
        raise ValueError("Selected start month cannot be after the selected end month.")
    selected_months = pd.period_range(first_month, last_month, freq="M")
    if month_count is not None and len(selected_months) > month_count:
        if direction == "first":
            selected_months = selected_months[:month_count]
        else:
            selected_months = selected_months[-month_count:]
    return selected_months


def month_columns(month_axis):
    return [month.strftime("%Y-%m") for month in month_axis]


def has_dated_observation(df, month_axis, date_columns=()):
    if df is None or df.empty:
        return False
    date_source = next(
        (df[column] for column in date_columns if column in df.columns),
        df.index,
    )
    dates = pd.to_datetime(pd.Series(date_source), errors="coerce").dropna()
    if dates.empty:
        return False
    return bool(dates.dt.to_period("M").isin(month_axis).any())


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


def build_eeg_matrix(eeg_df, month_axis):
    matrix = pd.DataFrame(0.0, index=EEG_BASE_ROWS, columns=month_columns(month_axis))
    if eeg_df is None or eeg_df.empty:
        return matrix

    working = eeg_df.copy()
    if "month" in working.columns:
        dates = pd.to_datetime(working["month"], format="%Y-%m", errors="coerce")
        dates = dates.fillna(pd.to_datetime(working["month"], errors="coerce"))
    elif "report_date" in working.columns:
        dates = pd.to_datetime(working["report_date"], errors="coerce")
    else:
        dates = pd.to_datetime(working.index, errors="coerce")

    working = working.loc[dates.notna()].copy()
    if working.empty:
        return matrix
    working["_month"] = dates[dates.notna()].dt.to_period("M")

    for feature in EEG_BASE_ROWS:
        if feature not in working.columns:
            continue
        values = working[feature].map(clean_numeric_value).fillna(0)
        by_month = values.groupby(working["_month"]).max()
        for month, value in by_month.items():
            column = month.strftime("%Y-%m")
            if column in matrix.columns and pd.notna(value):
                matrix.loc[feature, column] = float(value)

    for label, left_column, right_column in EEG_SLOW_REGION_ROWS + EEG_ED_REGION_ROWS:
        matrix.loc[label] = np.maximum(matrix.loc[left_column], matrix.loc[right_column])
    return matrix


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


def contrast_radius(value, cap):
    if pd.isna(value) or value <= 0:
        return 0
    cap = max(float(cap), 1e-12)
    scaled = min(float(value), cap) / cap
    return CONTRAST_MIN_RADIUS + (CONTRAST_MAX_RADIUS - CONTRAST_MIN_RADIUS) * (
        scaled ** CONTRAST_EXPONENT
    )


def cognitive_radius(feature, value):
    if pd.isna(value) or value <= 0:
        return 0
    maximum = COGNITIVE_MAX_SCORES.get(feature, 100)
    return contrast_radius(value, maximum)


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


def responsive_cell_fontsize(fig_width, fig_height, n_rows, n_cols):
    """Scale circle labels with the physical size of a rendered heatmap cell."""
    plot_width = max(fig_width - 2.8, 1.0)
    plot_height = max(fig_height - 1.8, 1.0)
    cell_inches = min(
        plot_width / max(n_cols, 1),
        plot_height / max(n_rows, 1),
    )
    return float(np.clip(cell_inches * 34.0, 8.0, 18.0))


def draw_heatmap(
    matrix,
    output_path,
    title,
    color_for_cell,
    radius_for_cell=None,
    row_label_formatter=None,
    label_scale=1.0,
    dpi=DPI,
):
    n_rows, n_cols = matrix.shape
    fig_width = max(12.0, min(72.0, 2.8 + n_cols * 0.18))
    fig_height = max(2.8, min(48.0, 1.8 + max(n_rows, 1) * 0.42))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    # Keep numeric values responsive to cell size so dense monthly rows do not
    # overlap, while the axis labels below can be enlarged for publication.
    cell_fontsize = responsive_cell_fontsize(
        fig_width, fig_height, n_rows, n_cols
    )

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
            ax.text(
                cx,
                cy,
                format_value(value),
                ha="center",
                va="center",
                fontsize=cell_fontsize,
                color="black",
                zorder=3,
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=22, pad=14)

    labels = [row_label_formatter(row) if row_label_formatter else row for row in matrix.index]
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(labels, fontsize=round(11 * label_scale))

    step = tick_step(n_cols)
    tick_positions = np.arange(0, n_cols, step) + 0.5
    tick_labels = [matrix.columns[idx] for idx in range(0, n_cols, step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
        fontsize=round(10 * label_scale),
    )
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def eeg_burden_label(value):
    if value <= EEG_UNSPECIFIED_BURDEN_VALUE:
        return "U"
    if value <= 1:
        return "R"
    if value <= 2:
        return "In"
    if value <= 3:
        return "F"
    return "C"


def eeg_burden_radius(value):
    """Map the ordinal EEG burden levels to clearly separated bubble sizes."""
    numeric_value = float(value)
    if numeric_value <= EEG_UNSPECIFIED_BURDEN_VALUE:
        return EEG_UNSPECIFIED_RADIUS
    if numeric_value <= 1:
        return 0.20
    if numeric_value <= 2:
        return 0.32
    if numeric_value <= 3:
        return 0.43
    return 0.48


def draw_eeg_burden_circle(ax, row_idx, col_idx, value, color, fontsize=6):
    if pd.isna(value) or value <= 0:
        return
    cx = col_idx + 0.5
    cy = row_idx + 0.5
    numeric_value = float(value)
    if numeric_value <= EEG_UNSPECIFIED_BURDEN_VALUE:
        color = EEG_UNSPECIFIED_COLOR
    radius = eeg_burden_radius(numeric_value)
    circle = patches.Circle((cx, cy), radius=radius, color=color, alpha=0.85, zorder=2)
    ax.add_patch(circle)
    ax.text(
        cx,
        cy,
        eeg_burden_label(numeric_value),
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
        zorder=3,
    )


def draw_eeg_lr_proportion(
    ax, row_idx, col_idx, left_value, right_value, color, fontsize=8
):
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
            fontsize=fontsize,
            color="black",
            zorder=3,
        )

    if left_ratio > 0 and right_ratio > 0:
        ax.plot(
            [col_idx + left_ratio, col_idx + left_ratio],
            [row_idx, row_idx + 1],
            color="black",
            linewidth=0.7,
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
            fontsize=fontsize,
            color="black",
            zorder=3,
        )


def draw_eeg_heatmap(matrix, output_path, title, dpi=DPI):
    display_rows = [row for row in EEG_PLOT_ROWS if row in matrix.index]
    if not display_rows:
        raise ValueError("No EEG rows available for plotting.")

    n_rows = len(display_rows)
    n_cols = matrix.shape[1]
    fig_width = max(12.0, min(72.0, 2.8 + n_cols * 0.18))
    fig_height = max(5.0, min(48.0, 1.8 + n_rows * 0.42))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    cell_fontsize = responsive_cell_fontsize(fig_width, fig_height, n_rows, n_cols)

    ax.vlines(np.arange(n_cols + 1), 0, n_rows, color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.hlines(np.arange(n_rows + 1), 0, n_cols, color="#bdbdbd", linewidth=0.75, zorder=0)

    slow_region_map = {label: (left, right) for label, left, right in EEG_SLOW_REGION_ROWS}
    ed_region_map = {label: (left, right) for label, left, right in EEG_ED_REGION_ROWS}

    for row_idx, feature in enumerate(display_rows):
        for col_idx, value in enumerate(matrix.loc[feature]):
            column = matrix.columns[col_idx]
            if feature in slow_region_map:
                left_column, right_column = slow_region_map[feature]
                draw_eeg_lr_proportion(
                    ax,
                    row_idx,
                    col_idx,
                    matrix.loc[left_column, column],
                    matrix.loc[right_column, column],
                    EEG_SLOW_COLOR,
                    fontsize=cell_fontsize,
                )
            elif feature in ed_region_map:
                left_column, right_column = ed_region_map[feature]
                draw_eeg_lr_proportion(
                    ax,
                    row_idx,
                    col_idx,
                    matrix.loc[left_column, column],
                    matrix.loc[right_column, column],
                    EEG_ED_COLOR,
                    fontsize=cell_fontsize,
                )
            elif feature in EEG_GLOBAL_SLOW_ROWS:
                draw_eeg_burden_circle(
                    ax, row_idx, col_idx, value, EEG_SLOW_COLOR, fontsize=cell_fontsize
                )
            elif feature in EEG_GLOBAL_ED_ROWS:
                draw_eeg_burden_circle(
                    ax, row_idx, col_idx, value, EEG_ED_COLOR, fontsize=cell_fontsize
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=22, pad=14)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(display_rows, fontsize=11)

    step = tick_step(n_cols)
    tick_positions = np.arange(0, n_cols, step) + 0.5
    tick_labels = [matrix.columns[idx] for idx in range(0, n_cols, step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def wrap_static_text(value, width):
    text = str(value or "").strip()
    if not text:
        return ""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def static_text_line_count(text):
    if not text:
        return 1
    return max(1, text.count("\n") + 1)


def draw_check_indicator(ax, x, y, is_active, color):
    ax.scatter(
        [x],
        [y],
        s=95,
        facecolors=color if is_active else "none",
        edgecolors=color if is_active else "#b6b6b6",
        linewidths=1.1,
        zorder=2,
    )
    if is_active:
        ax.text(x, y, "V", color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")


def draw_etiology_checklist(ax, y_top, raw_value):
    active_codes = {
        int(code)
        for code in re.findall(r"\d+", str(raw_value or ""))
        if 1 <= int(code) <= len(ETIOLOGY_LIST)
    }
    ax.text(0.02, y_top - 0.30, "Etiology", fontsize=14, fontweight="bold", va="center")

    first_row_y = y_top - 0.83
    rows_per_column = (len(ETIOLOGY_LIST) + 1) // 2
    for index, name in enumerate(ETIOLOGY_LIST):
        column, row = divmod(index, rows_per_column)
        x = 0.04 + column * 0.49
        y = first_row_y - row * 0.43
        code = index + 1
        is_active = code in active_codes
        draw_check_indicator(ax, x, y, is_active, "mediumpurple")
        ax.text(
            x + 0.025,
            y,
            f"{code}. {name}",
            fontsize=9,
            va="center",
            fontweight="bold" if is_active else "normal",
        )


def draw_eeg_focality_checklist(ax, y_top, raw_value):
    active_electrodes = {
        electrode.casefold()
        for electrode in docx_parser_static.extract_eeg_electrodes(raw_value)
    }
    ax.text(0.02, y_top - 0.30, "EEG Focality", fontsize=14, fontweight="bold", va="center")

    first_row_y = y_top - 0.86
    for row, (region, electrodes) in enumerate(EEG_FOCALITY_GROUPS.items()):
        y = first_row_y - row * 0.52
        ax.text(0.02, y, region, fontsize=10, fontweight="bold", color="darkorange", va="center")
        for column, electrode in enumerate(electrodes):
            x = 0.18 + column * 0.115
            is_active = electrode.casefold() in active_electrodes
            draw_check_indicator(ax, x, y, is_active, "orange")
            ax.text(
                x + 0.018,
                y,
                electrode,
                fontsize=9,
                va="center",
                fontweight="bold" if is_active else "normal",
            )


def draw_focal_type_checklist(ax, y_top, raw_value):
    active_codes = set(re.findall(r"\d+\.\d+|\d+", str(raw_value or "")))
    ax.text(
        0.02,
        y_top - 0.30,
        "Seizure Type (Focal/Gen)",
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    for index, code in enumerate(FOCAL_TYPE_CODES):
        column, row = index % 5, index // 5
        x = 0.04 + column * 0.19
        y = y_top - 0.82 - row * 0.52
        is_active = code in active_codes
        draw_check_indicator(ax, x, y, is_active, "crimson")
        ax.text(
            x + 0.025,
            y,
            code,
            fontsize=10,
            va="center",
            fontweight="bold" if is_active else "normal",
        )


def plot_static_data(static_data, output_path, title="Static Data", dpi=DPI):
    special_fields = {"etiology", "focal", "eeg focality"}
    items = [
        (wrap_static_text(key, width=26), wrap_static_text(value, width=78))
        for key, value in static_data.items()
        if str(key).strip().casefold() not in special_fields
    ]
    if not items:
        items = [("Static data", "No static data extracted")]

    etiology_value = next(
        (value for key, value in static_data.items() if str(key).strip().casefold() == "etiology"),
        "",
    )
    eeg_focality_value = next(
        (value for key, value in static_data.items() if str(key).strip().casefold() == "eeg focality"),
        "",
    )
    focal_type_value = next(
        (value for key, value in static_data.items() if str(key).strip().casefold() == "focal"),
        "",
    )

    row_heights = []
    for key, value in items:
        line_count = max(static_text_line_count(key), static_text_line_count(value))
        row_heights.append(max(0.72, 0.34 * line_count + 0.34))

    table_header_height = 1.35
    etiology_height = 3.75
    focal_type_height = 1.75
    eeg_focality_height = 3.35
    section_gap = 0.20
    bottom_padding = 0.25
    total_height = (
        table_header_height
        + sum(row_heights)
        + section_gap
        + etiology_height
        + section_gap
        + focal_type_height
        + section_gap
        + eeg_focality_height
        + bottom_padding
    )
    fig_height = max(5.0, min(80.0, total_height * 0.55))
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=dpi)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)

    title_y = total_height - 0.4
    header_y = total_height - 1.0
    table_top = total_height - table_header_height
    ax.text(0.02, title_y, title, fontsize=18, fontweight="bold", va="center")
    ax.text(0.02, header_y, "Field", fontsize=11, fontweight="bold", va="center")
    ax.text(0.34, header_y, "Value", fontsize=11, fontweight="bold", va="center")
    ax.hlines(table_top, 0.02, 0.98, color="#666666", linewidth=1.0)

    y_top = table_top
    for idx, ((key, value), row_height) in enumerate(zip(items, row_heights)):
        y_bottom = y_top - row_height
        if idx % 2 == 0:
            ax.add_patch(
                patches.Rectangle((0.02, y_bottom), 0.96, row_height, color="#f2f2f2", zorder=0)
            )
        text_y = y_top - 0.17
        ax.text(0.02, text_y, key, fontsize=10, fontweight="bold", va="top", linespacing=1.15)
        ax.text(0.34, text_y, value, fontsize=10, va="top", linespacing=1.15)
        y_top = y_bottom

    y_top -= section_gap
    ax.hlines(y_top, 0.02, 0.98, color="#cccccc", linewidth=0.8)
    draw_etiology_checklist(ax, y_top, etiology_value)

    y_top -= etiology_height + section_gap
    ax.hlines(y_top, 0.02, 0.98, color="#cccccc", linewidth=0.8)
    draw_focal_type_checklist(ax, y_top, focal_type_value)

    y_top -= focal_type_height + section_gap
    ax.hlines(y_top, 0.02, 0.98, color="#cccccc", linewidth=0.8)
    draw_eeg_focality_checklist(ax, y_top, eeg_focality_value)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def save_category_plots(stem, output_dir, matrices, include_static=True):
    asm_matrix = matrices["asm"]
    seizure_matrix = matrices["seizure"]
    eeg_matrix = matrices["eeg"]
    lab_matrix = matrices["lab"]
    cognitive_matrix, cognitive_flagged = matrices["cognitive"]
    static_data = matrices.get("static", {})

    outputs = {}
    outputs["asm"] = output_dir / f"{stem}_asm_heatmap.png"
    draw_heatmap(
        asm_matrix,
        outputs["asm"],
        "ASM Timeline (DDD Units)",
        lambda _feature, _value, _row, _col: ASM_COLOR,
        radius_for_cell=lambda feature, value, _cap: asm_radius(feature, value),
        label_scale=1.4,
    )

    outputs["seizure"] = output_dir / f"{stem}_seizure_heatmap.png"
    seizure_caps = {
        feature: value_cap(seizure_matrix.loc[[feature]])
        for feature in seizure_matrix.index
    }
    draw_heatmap(
        seizure_matrix,
        outputs["seizure"],
        "Seizure",
        lambda _feature, _value, _row, _col: SEIZURE_COLOR,
        radius_for_cell=lambda feature, value, _cap: contrast_radius(
            value, seizure_caps[feature]
        ),
        label_scale=1.8,
    )

    outputs["eeg"] = output_dir / f"{stem}_eeg_heatmap.png"
    draw_eeg_heatmap(eeg_matrix, outputs["eeg"], "EEG Timeline")

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

    if include_static:
        outputs["static"] = output_dir / f"{stem}_static_data.png"
        plot_static_data(static_data, outputs["static"], title=f"{stem} Static Data")

    return outputs


def build_patient_matrices(
    docx_path,
    start_month=None,
    end_month=None,
    months_to_plot=None,
    month_direction="first",
    include_static=True,
):
    errors = []
    asm_df = docx_parser_asm.extract_asm_timeline_from_docx(str(docx_path))
    month_axis = apply_month_range(
        make_month_axis(asm_df),
        start_month=start_month,
        end_month=end_month,
        months_to_plot=months_to_plot,
        month_direction=month_direction,
    )
    matrices = {"asm": build_asm_matrix(asm_df, month_axis)}
    presence = {
        "asm": has_dated_observation(asm_df, month_axis),
    }

    try:
        seizure_df = docx_parser_seizure.extract_seizure_timeline(str(docx_path))
    except Exception as exc:
        seizure_df = None
        errors.append(("seizure", str(exc)))
    matrices["seizure"] = build_seizure_matrix(seizure_df, month_axis)
    presence["seizure"] = has_dated_observation(seizure_df, month_axis, ("month",))

    try:
        eeg_df = docx_parser_eeg_timeline.table_selection(str(docx_path))
    except Exception as exc:
        eeg_df = None
        errors.append(("eeg", str(exc)))
    matrices["eeg"] = build_eeg_matrix(eeg_df, month_axis)
    presence["eeg"] = has_dated_observation(eeg_df, month_axis, ("month", "report_date"))

    try:
        lab_result = docx_parser_blood_test.extract_lab_timeline_from_docx(str(docx_path))
    except Exception as exc:
        lab_result = None
        errors.append(("lab", str(exc)))
    matrices["lab"] = build_lab_matrix(lab_result, month_axis)
    presence["lab"] = bool(
        lab_result
        and any(has_dated_observation(frame, month_axis) for frame in lab_result)
    )

    try:
        cognitive_df = docx_parser_cognitive.extract_cognitive_timeline_from_docx(str(docx_path))
    except Exception as exc:
        cognitive_df = None
        errors.append(("cognitive", str(exc)))
    matrices["cognitive"] = build_cognitive_matrices(cognitive_df, month_axis)
    presence["cognitive"] = has_dated_observation(cognitive_df, month_axis)
    matrices["_presence"] = presence

    if include_static:
        try:
            static_data = docx_parser_static.extract_static_data_from_docx(str(docx_path))
        except Exception as exc:
            static_data = {}
            errors.append(("static", str(exc)))
        matrices["static"] = static_data

    return month_axis, matrices, errors


def process_patient(
    docx_path,
    output_root,
    start_month=None,
    end_month=None,
    months_to_plot=None,
    month_direction="first",
    output_stem=None,
    include_static=True,
):
    docx_path = Path(docx_path)
    output_stem = str(output_stem or docx_path.stem)
    patient_dir = Path(output_root) / output_stem
    patient_dir.mkdir(parents=True, exist_ok=True)

    month_axis, matrices, errors = build_patient_matrices(
        docx_path,
        start_month=start_month,
        end_month=end_month,
        months_to_plot=months_to_plot,
        month_direction=month_direction,
        include_static=include_static,
    )
    outputs = save_category_plots(
        output_stem,
        patient_dir,
        matrices,
        include_static=include_static,
    )
    structured = extract_structured_features(matrices, lab_is_abnormal)
    return {
        "docx": str(docx_path),
        "output_dir": str(patient_dir),
        "months": len(month_axis),
        "start_month": month_axis.min().strftime("%Y-%m"),
        "end_month": month_axis.max().strftime("%Y-%m"),
        "outputs": outputs,
        "errors": errors,
        "structured_features": structured,
    }


def process_inputs(
    paths,
    output_dir,
    start_month=None,
    end_month=None,
    months_to_plot=None,
    month_direction="first",
):
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
                months_to_plot=months_to_plot,
                month_direction=month_direction,
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


def ask_month_count(root):
    while True:
        value = simpledialog.askstring(
            "Months to plot",
            "Enter the number of months to plot.\n"
            "Leave blank or Cancel to plot every available month.",
            parent=root,
        )
        try:
            return parse_month_count(value)
        except ValueError as exc:
            messagebox.showerror("Invalid month count", str(exc), parent=root)


def ask_month_direction(root, month_count):
    if month_count is None:
        return "first"
    plot_first = messagebox.askyesno(
        "Choose month direction",
        f"Which {month_count} months should be plotted?\n\n"
        f"Yes: first {month_count} months\n"
        f"No: last {month_count} months",
        parent=root,
    )
    return "first" if plot_first else "last"


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
        return [], "", None, None, None, "first"

    output_dir = askdirectory(title="Select Output Folder for Patient Heatmaps")
    if not output_dir:
        root.destroy()
        return list(files), "", None, None, None, "first"

    start_month, end_month = ask_month_range(root)
    months_to_plot = ask_month_count(root)
    month_direction = ask_month_direction(root, months_to_plot)
    root.destroy()
    return list(files), output_dir, start_month, end_month, months_to_plot, month_direction


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create per-patient ASM, seizure, EEG, lab, cognitive, and static-data PNGs "
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
    parser.add_argument(
        "--months-to-plot",
        default=None,
        help=(
            "Optional number of months to plot. If a patient has fewer months, "
            "all available months are plotted."
        ),
    )
    parser.add_argument(
        "--month-direction",
        choices=("first", "last"),
        default="first",
        help=(
            "Choose whether --months-to-plot keeps the first or last months "
            "in the selected range. Defaults to first."
        ),
    )
    args = parser.parse_args()

    paths = args.paths
    output_dir = args.output_dir
    start_month = args.start_month
    end_month = args.end_month
    months_to_plot = args.months_to_plot
    month_direction = args.month_direction

    if not paths:
        (
            paths,
            output_dir,
            start_month,
            end_month,
            months_to_plot,
            month_direction,
        ) = choose_gui_inputs()
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
        months_to_plot = parse_month_count(months_to_plot)
        month_direction = parse_month_direction(month_direction)
        if parse_month(start_month, "Start month") and parse_month(end_month, "End month"):
            if parse_month(start_month, "Start month") > parse_month(end_month, "End month"):
                raise ValueError("Start month cannot be after end month.")
    except ValueError as exc:
        safe_print(f"Invalid date range: {exc}")
        return

    process_inputs(
        paths,
        output_dir,
        start_month=start_month,
        end_month=end_month,
        months_to_plot=months_to_plot,
        month_direction=month_direction,
    )


if __name__ == "__main__":
    main()
