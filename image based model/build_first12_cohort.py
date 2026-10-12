import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import Heatmap_patient
import docx_parser_asm
import docx_parser_seizure
from structured_features import STRUCTURED_FEATURE_NAMES, extract_structured_features


MODALITIES = ("asm", "seizure", "eeg", "lab", "cognitive")
DEFAULT_TARGET_PER_CLASS = 72
DEFAULT_FOLDS = 4
DEFAULT_SEED = 42


def patient_id_from_path(path):
    match = re.search(r"\d+", Path(path).stem)
    if not match:
        raise ValueError(f"No numeric patient ID in {path}")
    return match.group(0).zfill(4)


def baseline_label_from_text(value):
    text = re.sub(r"\s+", "", str(value).upper())
    # A few legacy records use the consistent transposition "DER2" for
    # "DRE2". Keep the raw source text in the manifest, but normalize that
    # known spelling here for cohort-label validation.
    if re.search(r"(?<!N)(?:DRE|DER)\d*", text):
        return 1
    if "NDRE" in text or re.search(r"\((?:NS|S|ASM)\d*", text):
        return 0
    return None


def inclusive_month_span(dates):
    periods = pd.to_datetime(dates, errors="coerce").dropna().to_period("M")
    if len(periods) == 0:
        raise ValueError("ASM timeline has no parseable dates")
    return (periods.max() - periods.min()).n + 1, periods.min(), periods.max()


def inspect_source(path, label):
    patient_id = patient_id_from_path(path)
    asm_df = docx_parser_asm.extract_asm_timeline_from_docx(str(path))
    span, first_month, last_month = inclusive_month_span(asm_df.index)
    try:
        sections = docx_parser_seizure.extract_sections(path)
        baseline_text = sections.get("dre_label", "")
        baseline_label = baseline_label_from_text(baseline_text)
        baseline_audit_error = None
    except Exception as exc:
        baseline_text = ""
        baseline_label = None
        baseline_audit_error = f"{type(exc).__name__}:{exc}"

    reasons = []
    if span < 24:
        reasons.append("timeline_span_under_24_months")

    if baseline_audit_error is not None:
        label_audit = f"baseline_audit_error:{baseline_audit_error}"
    elif baseline_label is None:
        label_audit = "baseline_label_unparseable"
    elif baseline_label != label:
        label_audit = "folder_baseline_label_discordant"
    else:
        label_audit = "folder_baseline_label_agrees"

    return {
        "patient_id": patient_id,
        "label": int(label),
        "label_source": "curated_source_folder",
        "baseline_label_text": baseline_text,
        "label_audit": label_audit,
        "source_docx": str(Path(path).resolve()),
        "input_start_month": str(first_month),
        "input_end_month": str(first_month + 11),
        "timeline_end_month": str(last_month),
        "followup_span_months": int(span),
        "eligible": not reasons,
        "exclusion_reason": ";".join(reasons),
    }


def collect_candidates(dre_root, ndre_root):
    rows = []
    for label, root in ((1, Path(dre_root)), (0, Path(ndre_root))):
        # The active patient records live directly under DRE/ and NDRE/.
        # Do not recurse into archive folders such as NDRE/excel_oldfiles,
        # which contain older copies of the same patient documents.
        for path in sorted(root.glob("*.docx")):
            if path.name.startswith("~$"):
                continue
            try:
                rows.append(inspect_source(path, label))
            except Exception as exc:
                rows.append(
                    {
                        "patient_id": patient_id_from_path(path),
                        "label": label,
                        "label_source": "curated_source_folder",
                        "baseline_label_text": "",
                        "label_audit": "source_parse_error",
                        "source_docx": str(path.resolve()),
                        "input_start_month": "",
                        "input_end_month": "",
                        "timeline_end_month": "",
                        "followup_span_months": 0,
                        "eligible": False,
                        "exclusion_reason": f"source_parse_error:{type(exc).__name__}:{exc}",
                    }
                )

    candidates = pd.DataFrame(rows)
    duplicates = candidates[candidates["patient_id"].duplicated(keep=False)]
    if not duplicates.empty:
        ids = sorted(duplicates["patient_id"].unique())
        raise ValueError(f"Duplicate patient IDs across source documents: {ids}")
    return candidates


def select_balanced_cohort(candidates, target_per_class, folds, seed):
    if target_per_class % folds:
        raise ValueError("target_per_class must be divisible by folds")

    eligible = candidates[candidates["eligible"]].copy()
    counts = eligible["label"].value_counts().to_dict()
    for label, name in ((1, "DRE"), (0, "NDRE")):
        available = int(counts.get(label, 0))
        if available < target_per_class:
            raise ValueError(
                f"Need {target_per_class} eligible {name} patients, found {available}. "
                "Do not pad or oversample patients across outer folds."
            )

    rng = np.random.default_rng(seed)
    selected_indices = []
    for label in (1, 0):
        indices = eligible.index[eligible["label"] == label].to_numpy()
        chosen = rng.choice(indices, size=target_per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    selected = eligible.loc[selected_indices].copy()
    selected = selected.sort_values("patient_id").reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    selected["outer_fold"] = -1
    for fold, (_, test_indices) in enumerate(splitter.split(selected, selected["label"])):
        selected.loc[test_indices, "outer_fold"] = fold

    unselected = candidates.loc[~candidates.index.isin(selected_indices)].copy()
    balanced_out = unselected["eligible"] & unselected["exclusion_reason"].eq("")
    unselected.loc[balanced_out, "exclusion_reason"] = "not_selected_for_class_balance"
    return selected, unselected


def add_expected_plot_paths(selected, output_root):
    output_root = Path(output_root)
    for modality in MODALITIES:
        selected[modality] = selected["patient_id"].map(
            lambda patient_id: (Path(patient_id) / f"{patient_id}_{modality}_heatmap.png").as_posix()
        )
    return selected


def validate_selected_cohort(selected, target_per_class, folds):
    if selected["patient_id"].duplicated().any():
        raise AssertionError("Selected cohort contains duplicate patient IDs")
    if len(selected) != target_per_class * 2:
        raise AssertionError("Selected cohort has the wrong total size")
    if selected["label"].value_counts().to_dict() != {0: target_per_class, 1: target_per_class}:
        raise AssertionError("Selected cohort is not class-balanced")
    if (selected["followup_span_months"] < 24).any():
        raise AssertionError("Selected cohort contains a timeline shorter than 24 months")

    expected_per_class = target_per_class // folds
    table = pd.crosstab(selected["outer_fold"], selected["label"])
    if table.shape != (folds, 2) or not (table.to_numpy() == expected_per_class).all():
        raise AssertionError(
            f"Each outer fold must contain {expected_per_class} DRE and "
            f"{expected_per_class} NDRE patients; got\n{table}"
        )
    return table


def generate_plots(selected, output_root, overwrite=False):
    output_root = Path(output_root)
    warning_counts = []
    structured_rows = []
    for position, row in selected.iterrows():
        expected = [output_root / row[modality] for modality in MODALITIES]
        if not overwrite and all(path.exists() for path in expected):
            print(f"[{position + 1}/{len(selected)}] Reusing {row['patient_id']}")
            month_axis, matrices, errors = Heatmap_patient.build_patient_matrices(
                row["source_docx"],
                months_to_plot=12,
                month_direction="first",
                include_static=False,
            )
            structured = extract_structured_features(matrices, Heatmap_patient.lab_is_abnormal)
            months = len(month_axis)
        else:
            print(f"[{position + 1}/{len(selected)}] Generating {row['patient_id']}")
            result = Heatmap_patient.process_patient(
                row["source_docx"],
                output_root,
                months_to_plot=12,
                month_direction="first",
                output_stem=row["patient_id"],
                include_static=False,
            )
            structured = result["structured_features"]
            errors = result["errors"]
            months = result["months"]
        if months != 12:
            raise AssertionError(f"{row['patient_id']} generated {months} months, expected 12")
        warning_counts.append(len(errors))
        structured_rows.append(structured)

    selected["parser_warning_count"] = warning_counts
    for feature in STRUCTURED_FEATURE_NAMES:
        selected[feature] = [row[feature] for row in structured_rows]
    structured_values = selected.loc[:, list(STRUCTURED_FEATURE_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(structured_values).all():
        raise ValueError("Structured feature table contains non-finite values")
    for path in [output_root / row[modality] for _, row in selected.iterrows() for modality in MODALITIES]:
        if not path.exists():
            raise FileNotFoundError(path)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a balanced earliest-12-month image cohort for final DRE status prediction."
    )
    parser.add_argument("--dre-root", default="DRE")
    parser.add_argument("--ndre-root", default="NDRE")
    parser.add_argument("--output-root", default="12-month_plots")
    parser.add_argument("--manifest", default="first12_final_status_manifest.csv")
    parser.add_argument("--exclusions", default="first12_final_status_exclusions.csv")
    parser.add_argument("--target-per-class", type=int, default=DEFAULT_TARGET_PER_CLASS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and select the cohort without generating plots or writing CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = collect_candidates(args.dre_root, args.ndre_root)
    selected, exclusions = select_balanced_cohort(
        candidates,
        target_per_class=args.target_per_class,
        folds=args.folds,
        seed=args.seed,
    )
    selected = add_expected_plot_paths(selected, args.output_root)
    fold_table = validate_selected_cohort(selected, args.target_per_class, args.folds)

    print("Eligible source patients by label:")
    print(candidates[candidates["eligible"]]["label"].value_counts().sort_index())
    excluded = candidates[~candidates["eligible"]]
    if not excluded.empty:
        print("Excluded source records by label and reason:")
        print(excluded.groupby(["label", "exclusion_reason"]).size())
    print("Selected outer-fold composition:")
    print(fold_table)

    if args.dry_run:
        print("Dry run complete; no plots or CSV files were written.")
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    selected = generate_plots(selected, args.output_root, overwrite=args.overwrite)
    selected.to_csv(args.manifest, index=False, encoding="utf-8-sig")
    exclusions.to_csv(args.exclusions, index=False, encoding="utf-8-sig")
    print(f"Saved {len(selected)} selected patients to {args.manifest}")
    print(f"Saved {len(exclusions)} exclusions to {args.exclusions}")


if __name__ == "__main__":
    main()
