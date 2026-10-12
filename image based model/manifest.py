import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path("36-month_plots")

# Confirm this mapping before training.
LABELS = {
    "DRE": 1,
    "NS": 0,
    "S": 0,
    "ASM": 0,
}

SUFFIXES = {
    "asm": "_asm_heatmap.png",
    "cognitive": "_cognitive_heatmap.png",
    "eeg": "_eeg_heatmap.png",
    "lab": "_lab_heatmap.png",
}

# Resolve folders that contain more than one seizure image.
# Change this if the other seizure image is the correct one.
SEIZURE_OVERRIDES = {
    "0117": "0117 NS1_seizure_heatmap.png",
}


def find_by_suffix(folder: Path, suffix: str) -> Path:
    matches = [
        path
        for path in folder.glob("*.png")
        if path.name.lower().endswith(suffix.lower())
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{folder.name}: expected one {suffix} file, found {len(matches)}"
        )

    return matches[0]


def find_seizure(folder: Path, patient_id: str) -> Path:
    if patient_id in SEIZURE_OVERRIDES:
        selected = folder / SEIZURE_OVERRIDES[patient_id]
        if not selected.exists():
            raise FileNotFoundError(selected)
        return selected

    matches = [
        path
        for path in folder.glob("*.png")
        if "seizure" in path.name.lower()
    ]

    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError(
            f"{folder.name}: expected one seizure image, found {names}"
        )

    return matches[0]


rows = []

for folder in sorted(path for path in ROOT.iterdir() if path.is_dir()):
    match = re.fullmatch(
        r"(\d+)\s+(DRE|NS|S|ASM)\d*",
        folder.name,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(f"Unrecognized folder name: {folder.name}")

    patient_id, status = match.groups()
    status = status.upper()

    row = {
        "patient_id": patient_id,
        "label": LABELS[status],
    }

    for modality, suffix in SUFFIXES.items():
        path = find_by_suffix(folder, suffix)
        row[modality] = path.relative_to(ROOT).as_posix()

    row["seizure"] = find_seizure(
        folder, patient_id
    ).relative_to(ROOT).as_posix()

    # No static path is added.
    rows.append(row)


manifest = pd.DataFrame(rows)

if manifest["patient_id"].duplicated().any():
    raise ValueError("Duplicate patient IDs found")

# Each patient is one row, so stratified folds are patient-level.
splitter = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

manifest["fold"] = -1

for fold, (_, validation_indices) in enumerate(
    splitter.split(manifest, manifest["label"])
):
    manifest.loc[validation_indices, "fold"] = fold

manifest.to_csv("patient_manifest.csv", index=False)

print(manifest["label"].value_counts())
print(manifest.groupby(["fold", "label"]).size())
print(f"Saved {len(manifest)} patients")