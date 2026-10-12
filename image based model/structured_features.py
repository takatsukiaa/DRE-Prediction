"""Structured clinical summaries derived from the first-12-month plot matrices."""

from collections.abc import Callable

import numpy as np
import pandas as pd


STRUCTURED_FEATURE_NAMES = (
    "asm_max_drug_count",
    "asm_mean_drug_count",
    "asm_last_drug_count",
    "asm_drug_count_change",
    "asm_max_total_ddd",
    "asm_mean_total_ddd",
    "asm_last_total_ddd",
    "asm_total_ddd_change",
    "asm_distinct_drugs",
    "asm_new_drug_starts",
    "asm_regimen_changes",
    "seizure_data_present",
    "seizure_total",
    "seizure_active_months",
    "seizure_max_monthly",
    "seizure_past_year_max",
    "eeg_data_present",
    "eeg_active_months",
    "eeg_slowing_feature_months",
    "eeg_epileptiform_feature_months",
    "eeg_slowing_burden",
    "eeg_epileptiform_burden",
    "lab_data_present",
    "lab_observation_count",
    "lab_abnormal_count",
    "lab_abnormal_fraction",
    "cognitive_data_present",
    "cognitive_observation_count",
    "cognitive_flagged_count",
    "cognitive_flagged_fraction",
    "modality_count_present",
)

EEG_SLOWING_ROWS = (
    "Delta waves",
    "Theta waves",
    "Temporal Slow-wave",
    "Frontal Slow-wave",
    "Occipital Slow-wave",
    "Parietal Slow-wave",
    "Central Slow-wave",
)
EEG_EPILEPTIFORM_ROWS = (
    "Spike ED",
    "Sharp ED",
    "Sharpish configuration",
    "Spike-and-wave ED",
    "Periodic Discharge",
    "Temporal ED",
    "Frontal ED",
    "Occipital ED",
    "Parietal ED",
    "Central ED",
)
ASM_SUMMARY_ROWS = {"total ddd", "number of drugs"}


def _numeric_row(matrix: pd.DataFrame, row_name: str) -> np.ndarray:
    if row_name not in matrix.index:
        return np.zeros(matrix.shape[1], dtype=float)
    return (
        pd.to_numeric(matrix.loc[row_name], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _selected_rows(matrix: pd.DataFrame, row_names: tuple[str, ...]) -> np.ndarray:
    rows = [_numeric_row(matrix, row_name) for row_name in row_names]
    if not rows:
        return np.zeros((0, matrix.shape[1]), dtype=float)
    return np.vstack(rows)


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def extract_structured_features(
    matrices: dict,
    lab_is_abnormal: Callable[[str, float], bool],
) -> dict[str, float]:
    """Summarize the same matrices used to render the five model inputs.

    Zeros are treated as absent plot observations because the rendering code
    also leaves zero-valued cells blank. This makes the presence indicators
    consistent with what the image branch can see.
    """

    asm = matrices["asm"]
    seizure = matrices["seizure"]
    eeg = matrices["eeg"]
    lab = matrices["lab"]
    cognitive, cognitive_flagged = matrices["cognitive"]
    source_presence = matrices.get("_presence", {})

    drug_rows = [
        row
        for row in asm.index
        if str(row).strip().casefold() not in ASM_SUMMARY_ROWS
    ]
    drug_values = (
        asm.loc[drug_rows]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        if drug_rows
        else np.zeros((0, asm.shape[1]), dtype=float)
    )
    active_drugs = drug_values > 0
    observed_drug_count = active_drugs.sum(axis=0).astype(float)
    drug_count = _numeric_row(asm, "Number of Drugs")
    if not np.any(drug_count > 0):
        drug_count = observed_drug_count
    total_ddd = _numeric_row(asm, "Total DDD")
    if not np.any(total_ddd > 0) and drug_values.size:
        total_ddd = drug_values.sum(axis=0)

    if active_drugs.shape[1] > 1:
        new_drug_starts = int((~active_drugs[:, :-1] & active_drugs[:, 1:]).sum())
        regimen_changes = int(np.any(active_drugs[:, 1:] != active_drugs[:, :-1], axis=0).sum())
    else:
        new_drug_starts = 0
        regimen_changes = 0

    monthly_seizures = _numeric_row(seizure, "Seizure")
    past_year_seizures = _numeric_row(seizure, "Seizure in the past 1 year")
    seizure_present = bool(
        source_presence.get(
            "seizure",
            np.any(monthly_seizures > 0) or np.any(past_year_seizures > 0),
        )
    )

    eeg_slowing = _selected_rows(eeg, EEG_SLOWING_ROWS)
    eeg_epileptiform = _selected_rows(eeg, EEG_EPILEPTIFORM_ROWS)
    eeg_visible = np.vstack([eeg_slowing, eeg_epileptiform])
    eeg_present = bool(source_presence.get("eeg", np.any(eeg_visible > 0)))

    lab_values = lab.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    lab_observation_count = int((lab_values.to_numpy(dtype=float) != 0).sum())
    lab_abnormal_count = sum(
        int(lab_is_abnormal(str(feature), float(value)))
        for feature in lab_values.index
        for value in lab_values.loc[feature]
        if float(value) != 0
    )

    cognitive_values = cognitive.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cognitive_observation_count = int((cognitive_values.to_numpy(dtype=float) != 0).sum())
    cognitive_flagged_count = int(cognitive_flagged.to_numpy(dtype=bool).sum())
    asm_present = bool(
        source_presence.get("asm", np.any(drug_count > 0) or np.any(total_ddd > 0))
    )
    lab_present = bool(source_presence.get("lab", lab_observation_count > 0))
    cognitive_present = bool(
        source_presence.get("cognitive", cognitive_observation_count > 0)
    )

    modality_count_present = sum(
        (
            asm_present,
            seizure_present,
            eeg_present,
            lab_present,
            cognitive_present,
        )
    )

    features = {
        "asm_max_drug_count": float(drug_count.max(initial=0.0)),
        "asm_mean_drug_count": float(drug_count.mean()) if drug_count.size else 0.0,
        "asm_last_drug_count": float(drug_count[-1]) if drug_count.size else 0.0,
        "asm_drug_count_change": float(drug_count[-1] - drug_count[0]) if drug_count.size else 0.0,
        "asm_max_total_ddd": float(total_ddd.max(initial=0.0)),
        "asm_mean_total_ddd": float(total_ddd.mean()) if total_ddd.size else 0.0,
        "asm_last_total_ddd": float(total_ddd[-1]) if total_ddd.size else 0.0,
        "asm_total_ddd_change": float(total_ddd[-1] - total_ddd[0]) if total_ddd.size else 0.0,
        "asm_distinct_drugs": float(active_drugs.any(axis=1).sum()),
        "asm_new_drug_starts": float(new_drug_starts),
        "asm_regimen_changes": float(regimen_changes),
        "seizure_data_present": float(seizure_present),
        "seizure_total": float(monthly_seizures.sum()),
        "seizure_active_months": float((monthly_seizures > 0).sum()),
        "seizure_max_monthly": float(monthly_seizures.max(initial=0.0)),
        "seizure_past_year_max": float(past_year_seizures.max(initial=0.0)),
        "eeg_data_present": float(eeg_present),
        "eeg_active_months": float((eeg_visible > 0).any(axis=0).sum()),
        "eeg_slowing_feature_months": float((eeg_slowing > 0).sum()),
        "eeg_epileptiform_feature_months": float((eeg_epileptiform > 0).sum()),
        "eeg_slowing_burden": float(eeg_slowing.clip(min=0).sum()),
        "eeg_epileptiform_burden": float(eeg_epileptiform.clip(min=0).sum()),
        "lab_data_present": float(lab_present),
        "lab_observation_count": float(lab_observation_count),
        "lab_abnormal_count": float(lab_abnormal_count),
        "lab_abnormal_fraction": _safe_fraction(lab_abnormal_count, lab_observation_count),
        "cognitive_data_present": float(cognitive_present),
        "cognitive_observation_count": float(cognitive_observation_count),
        "cognitive_flagged_count": float(cognitive_flagged_count),
        "cognitive_flagged_fraction": _safe_fraction(
            cognitive_flagged_count,
            cognitive_observation_count,
        ),
        "modality_count_present": float(modality_count_present),
    }
    if tuple(features) != STRUCTURED_FEATURE_NAMES:
        raise AssertionError("Structured feature names and extraction order diverged")
    if not np.isfinite(np.fromiter(features.values(), dtype=float)).all():
        raise ValueError("Structured feature extraction produced a non-finite value")
    return features
