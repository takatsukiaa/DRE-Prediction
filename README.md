# DRE/NDRE visit-status prediction

The value-based workflow estimates **P(DRE at each medication visit)** with
four independent random forests using 3, 6, 9, or 12 calendar months of clinical
history. P(NDRE) is 1 minus P(DRE). The future state-switch model is retired.

## History windows

Every history window starts **12 calendar months before the prediction visit**.
The start is inclusive and the end is exclusive:

| Model | History start | History end (exclusive) |
|---|---|---|
| 3-month history | Prediction minus 12 months | Prediction minus 9 months |
| 6-month history | Prediction minus 12 months | Prediction minus 6 months |
| 9-month history | Prediction minus 12 months | Prediction minus 3 months |
| 12-month history | Prediction minus 12 months | Prediction visit |

For a prediction on January 15, 2020, the windows are January 15 to April 15, 2019;
January 15 to July 15, 2019; January 15 to October 15, 2019; and January 15, 2019
to January 15, 2020, respectively.
Calendar offsets are calculated from the prediction date, clipping to the last
day when necessary. Each model's probabilities may differ without a required
ordering.

Medications, labs, cognition, EEG, imaging timing, seizure summaries, and changes
are restricted to the respective history window. Linked seizure reports and
embedded EEG findings must also be available strictly before its end. Evidence
from the gap between a shorter window and prediction cannot enter that model.
Measurement ages are measured at the prediction visit. Age and immutable sex
remain available. Prior monthly status labels and unrestricted history-length
features are excluded. Missing values remain missing until training-only
imputation. Visits with sparse or no dynamic history are included; a complete
recorded history window is not required.

## Visit labels

All models share the same visit labels and a reproducible 50/50 patient split
generated from the snapshot. The operational target uses visit-day medications
and seizure evidence
available by that date under the project's existing three-valued rules. Its
seizure interval is always `(visit minus 12 calendar months, visit]`, regardless
of input length.

If visit-date status is unknown, the latest monthly assessment strictly before
the visit supplies the label only when that assessment is known. An unknown
latest month is not skipped to retrieve an older known label. Known visit-date
labels take precedence. Remaining unknown labels are retained for audit and
scoring but excluded from fitting and evaluation.

## Workflow

| Entry point under `value based model/` | Reads | Produces |
|---|---|---|
| `extract_sql.py` | Live EpilepsyDB through read-only queries | SQL snapshot, schema, checksums |
| `build_visit_status.py` | Saved snapshot | Separate histories, shared visit targets, generated patient split, audits, manifest |
| `train.py` | Prepared visit-status bundle | Forests, feature rankings, probabilities, held-out evaluation, confusion matrices |

Extract when the source data need updating:

```powershell
.\.venv\Scripts\python.exe "value based model/extract_sql.py" --output-dir "value based model/data/sql_snapshots/epilepsydb_NEW_RUN"
```

Prepare histories and labels with a new reproducible patient split:

```powershell
.\.venv\Scripts\python.exe "value based model/build_visit_status.py" --snapshot-dir "value based model/data/sql_snapshots/20261017" --output-dir "value based model/data/features/visit_status_NEW_RUN" --seed 42
```

Preparation shuffles sorted snapshot patient IDs using `--seed` (default 42),
assigns half to training (rounded down) and the remainder to testing, and saves
`patient_split.json` inside the new bundle. For 265 patients this gives 132
training and 133 test patients. All visits for a patient stay in one group.
Assignment does not use outcomes or visit counts. The same patient IDs and seed
reproduce the same split regardless of snapshot row order. The seed and method
are recorded in provenance. No earlier split file is required.

Train and evaluate from the prepared bundle:

```powershell
.\.venv\Scripts\python.exe "value based model/train.py" --feature-dir "value based model/data/features/visit_status_NEW_RUN" --output-dir "value based model/results/rf_visit_status_NEW_RUN"
```

Use a new output directory for every run. Training verifies checksums and
alignment and does not reopen SQL or the snapshot, rebuild features, recalculate
labels, or change patient assignments. Format-2 visit-status bundles record the
12-month anchor and all four history windows; bundles containing only 3-, 6-,
and 12-month histories must be rebuilt before training.

All-feature and training-selected top-20 forests use training-only preprocessing
and feature ranking. Each patient has equal total fitting and evaluation weight.
All models are evaluated on the same labeled visits from held-out patients,
alongside a training-prevalence baseline. No cross-validation, threshold tuning,
test-based feature selection, or calibration fitting is performed.

Outputs include `report.md`, `metrics.json`, saved `VisitStatusModel` pipelines,
feature rankings, and `heldout_status_predictions.jsonl`. Each prediction records
DRE/NDRE percentages, a threshold-0.5 class, and the inclusive history start and
exclusive history end for each model. `status_predictions.jsonl` also includes
explicitly marked in-sample training patients.

Training automatically saves twelve PNG confusion matrices under
`confusion_matrices/`: all-feature, selected-feature, and prevalence-baseline
results for each 3-, 6-, 9-, and 12-month history. The report links to each image.
Rows show actual DRE/NDRE status and columns show predicted DRE/NDRE status,
with the predicted labels at the top, at threshold 0.5. DRE is the positive
class. The default blue heatmap reflects visit counts. Each cell shows only an
unweighted held-out visit count and its
percentage within the observed class; N/A indicates an absent observed class.
The report's performance metrics continue to use patient weights.

Training also saves `roc_comparison.png`, overlaying the 3-, 6-, 9-, and 12-month
selected-feature models on one ROC plot, with AUROCs in the legend and a chance
diagonal. All four curves use the same labeled held-out visits and equal-patient
weights as the performance metrics. Unknown labels are excluded. Curve points
and AUROCs are saved in `roc_curves.json`, and the report embeds the plot.

Generate plots for an existing visit-status results directory without retraining:

```powershell
.\.venv\Scripts\python.exe "value based model/plot_confusion_matrices.py" --results-dir "value based model/results/rf_visit_status_NEW_RUN"
.\.venv\Scripts\python.exe "value based model/plot_roc_curves.py" --results-dir "value based model/results/rf_visit_status_NEW_RUN"
```

Percentages are uncalibrated estimates of the project-derived status. Labels
are not independently adjudicated outcomes. Observation and linked visit dates
approximate availability because recording timestamps are unavailable. Scores
on visits with unknown outcomes are not validated by the evaluation. Historical
data and model runs remain saved.

## Supporting modules and verification

`sql_snapshot.py` handles snapshots; `longitudinal_features.py` builds clinical
histories; `eeg_features.py` adapts the existing EEG encoder;
`visit_status_targets.py` builds visit labels; `forest_workflow.py` supplies
shared fitting and evaluation; `visit_status_workflow.py` validates bundles,
trains status models, and writes reports. Clinical rules remain under
`Database/dre_onset_analysis/`, with the existing
`docx_parsers/docx_parser_eeg_timeline.py` encoder.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "value based model" -v
.\.venv\Scripts\python.exe "value based model/tools/verify_all_tables.py" "value based model/data/sql_snapshots/epilepsydb_NEW_RUN"
```

Synthetic fixtures live in `tests/helpers.py`. Tests cover history boundaries,
evidence availability, target timing, model serialization, reports, snapshot
integrity, and separation of extraction, preparation, and training.
