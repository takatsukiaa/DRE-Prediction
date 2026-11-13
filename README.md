# DRE/NDRE visit-status prediction

The value-based workflow estimates **P(DRE at each medication visit)** with
three independent random forests using 3, 6, or 9 calendar months of clinical
history. P(NDRE) is 1 minus P(DRE). The future state-switch model is retired.

## History windows

Every history window starts **12 calendar months before the prediction visit**.
The start is inclusive and the end is exclusive:

| Model | History start | History end (exclusive) |
|---|---|---|
| 3-month history | Prediction minus 12 months | Prediction minus 9 months |
| 6-month history | Prediction minus 12 months | Prediction minus 6 months |
| 9-month history | Prediction minus 12 months | Prediction minus 3 months |

For a prediction on January 15, 2020, the windows are January 15 to April 15, 2019;
January 15 to July 15, 2019; and January 15 to October 15, 2019, respectively.
Calendar offsets are calculated from the prediction date, clipping to the last
day when necessary. Each model's probabilities may differ without a required
ordering.

Only these three prospective models are supported. The 12-month history model
is retired because its inputs end at prediction and leave no prediction gap.
All future feature audits, ablations, training, evaluation, and reports use only
the 3-, 6-, and 9-month models. Historical artifacts are retained for provenance.

Medications, labs, cognition, EEG, imaging timing, seizure summaries, and changes
are restricted to the respective history window. Linked seizure reports and
embedded EEG findings must also be available strictly before its end. Evidence
from the gap between a shorter window and prediction cannot enter that model.
Measurement ages are measured at the prediction visit. Age and immutable sex
remain available. Prior monthly status labels and unrestricted history-length
features are excluded. Missing values remain missing until training-only
imputation. Visits with sparse or no dynamic history are included; a complete
recorded history window is not required.

## Active feature representations

New bundles use the `recorded_counts_actual_visits_v2` feature schema. ASM burden has one
count, `asm_drug_count_latest`. Seizure summaries use `seizure_count_{window}m`:
the sum of recorded counts from fully contained, available intervals. Building
features checks that each patient's source seizure intervals do not overlap and
reports conflicting record IDs if that contract is violated. Boundary-crossing
positive counts are not allocated to the window. Partial-positive interval
predictors and seizure-rate predictors are removed.

Seizure coverage fractions are restored as predictors for every active history
window, including the recent 6-month summary in the 9-month model. The earlier
covered-day predictors remain removed. Documented zero-seizure days remain a
distinct predictor. Coverage is also summarized in audit diagnostics.
Per-finding EEG scores remain without their binary presence copies. Numeric
measurements retain their latest values, deltas, measurement ages, observation
counts and change gaps; derived change rates are removed. Medication doses,
DDD, cognitive totals and components remain available. Entirely missing and
truly constant features are excluded using training data; their source catalog
entries remain available for future populated records.

Missing assessments remain missing. A recorded seizure count of zero does not
assert seizure freedom on unobserved days. New target audit exports use
`target_seizure_count`. Outcome rules and their boundary evidence remain
unchanged. SQL schema names and historical artifacts retain their original
names for compatibility and provenance.

Medication comparisons use the original source legend in
`docx_parsers/docx_parser_asm.py`: `IsWuVisit=1` identifies a visit with Dr. Wu,
`0` identifies a visit with another physician, and `-` means no visit record.
Both `0` and `1` count as clinic visits. Missing or unrecognized markers do not.
Monthly medication records without a confirmed visit cannot create a visit pair.

The 3-month model excludes `asm_dose_comparable_drugs` and all medication
comparison predictors: drug-count/dose/DDD deltas, added/stopped medicines,
regimen changes, pair availability, previous-visit gaps, assessable pairs,
composition-change counts and time since a composition change. Single-record
ASM count, presence, dose and total DDD remain.

For 6 and 9 months, comparisons use the last two actual visits inside the model's
input window. Identical regimens on distinct visit dates are valid pairs.
Regimen-change histories use consecutive actual visits, including in the recent
6-month summaries. Missing data at an actual visit are not bypassed by choosing
an older visit. Fewer than two eligible visits leave change features missing.
`asm_visits_*` counts actual visits. The latest medication level still comes
from the latest available medication record, independently of comparison
endpoints. Prediction rows and calculated target labels remain unchanged.

## Feature terminology

Use the same display name for a feature in explanations, documentation, plots,
and reports. User-facing feature lists use readable names without internal
identifiers. The identifiers below are technical mappings and remain unchanged
in machine-readable artifacts. If a feature has no documented display name,
record an accurate readable name here and reuse it consistently.
For window-specific features, replace `{window}` with the summary length in
months; the 9-month model also contains recent 6-month summaries.

| Feature identifier | Canonical display name | Meaning |
|---|---|---|
| `asm_dose_comparable_drugs` | ASMs with doses at both visits | Number of continuing ASMs with unambiguous daily doses recorded at the last two actual visits inside the input window. Doses may change. This does not measure dose stability. |
| `asm_composition_changes_{window}m` | ASM composition changes, {window}m | Number of changes in the set of ASMs between assessable consecutive actual visits within that summary window; dose-only changes do not count. |
| `seizure_coverage_fraction_{window}m` | Seizure coverage fraction, {window}m | Fraction of days in the summary window covered by known-count seizure intervals available before the input cutoff. It reflects timing and interval coverage, not only missing counts. |
| `seizure_count_{window}m` | Recorded seizure count, {window}m | Sum of counts from fully contained, available seizure intervals. |
| `seizure_zero_days_{window}m` | Documented zero-seizure days, {window}m | Days covered by available zero-count seizure intervals. |
| `asm_drug_count_latest` | ASM count at latest assessment | Number of active scheduled ASMs at the latest eligible medication assessment in the input window. |
| `asm_total_ddd_latest` | Total ASM dose (DDD) | Sum of each active ASM's daily dose divided by its recorded reference daily dose, at the latest eligible medication assessment. |
| `asm_total_ddd_delta` | Change in total ASM dose (DDD) | Latest minus previous total DDD at the last two eligible actual visits. |
| `asm__{drug}_daily_mg` | {Drug} dose (mg/day) | Drug's daily dose at the latest eligible medication assessment; use the drug's conventional English name. |
| `asm__{drug}_daily_mg_delta` | {Drug} dose change (mg/day) | Latest minus previous daily dose at the last two eligible actual visits. |
| `asm__{drug}_present` | {Drug} use | Whether the drug is an active scheduled ASM at the latest eligible medication assessment. |
| `asm_visits_{window}m` | Actual visit count, {window}m | Number of confirmed actual visits in the summary window. |
| `asm_days_since_last_observed_composition_change` | Days since last recorded ASM composition change | Days from the latest assessable composition change in the input window to the prediction date. |
| `asm_previous_visit_gap_days` | Days between the last two actual visits | Date gap between the last two eligible actual visits in the input window. |
| `age_years` | Age (years) | Age at the prediction date. |
| `sex_male` | Male sex | Recorded sex category, encoded as male versus female. |
| `biochem__alt_u_l_91de6add__latest` | Latest ALT (U/L) | Latest eligible alanine aminotransferase measurement in the input window. |

Always show mg/day for individual medication doses and dose changes, and DDD
for total ASM dose and its change. Latest medication assessments are inside
the input window and need not be actual visits. Actual-visit dose comparisons
use the last two eligible actual visits inside the model's input window.

Do not use "stable consecutive ASM doses" or "ASM with comparable dose" as
alternate names for `asm_dose_comparable_drugs`. For its first mention in a
feature list, include the model history window alongside the canonical name.
This terminology policy applies to new material; historical artifacts retain
their original wording for provenance.

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
| `extract_sql.py` | Live EpilepsyDB through read-only queries | SQL snapshot, schema, checksums, or verification of a saved snapshot |
| `build_visit_status.py` | Saved snapshot | Separate histories, shared visit targets, patient split, 12-month evaluation window, audits, manifest |
| `train.py` | Prepared visit-status bundle | Forests, feature rankings, probabilities, recent held-out evaluation, plots |
| `evaluate_latest_visits.py` | Saved visit-status run and optional reference labels | Latest-visit performance, predictions, plots, and label audit |
| `plot_visit_status.py` | Saved training or latest-visit results | Regenerated ROC curves and confusion matrices |

Extract when the source data need updating:

```powershell
.\.venv\Scripts\python.exe "value based model/extract_sql.py" --output-dir "value based model/data/sql_snapshots/epilepsydb_NEW_RUN"
```

Prepare histories and labels with a new reproducible patient split:

```powershell
.\.venv\Scripts\python.exe "value based model/build_visit_status.py" --snapshot-dir "value based model/data/sql_snapshots/20261019" --output-dir "value based model/data/features/visit_status_NEW_RUN" --seed 42
```

Preparation shuffles sorted snapshot patient IDs using `--seed` (default 42),
assigns half to training (rounded down) and the remainder to testing, and saves
`patient_split.json` inside the new bundle. For 265 patients this gives 132
training and 133 test patients. All visits for a patient stay in one group.
Assignment does not use outcomes or visit counts. The same patient IDs and seed
reproduce the same split regardless of snapshot row order. The seed and method
are recorded in provenance. No earlier split file is required.

Preparation also tags visits in the 12 calendar months ending on the snapshot's
Asia/Taipei extraction date as recent. Both boundary dates are included.
Historical visits precede the cutoff. Features are built for both periods using
each visit's own bounded history so recent visits can be scored by the saved models.

Train and evaluate from the prepared bundle:

```powershell
.\.venv\Scripts\python.exe "value based model/train.py" --feature-dir "value based model/data/features/visit_status_NEW_RUN"
```

Training automatically creates a new result folder named
`rf_visit_status_YYYYMMDD_HHMMSS_<trees>trees` using Asia/Taipei time. Omit
`--output-dir` for future runs unless a specific path is requested.
Training verifies checksums, visit alignment, and the saved evaluation window.
It uses the prepared feature and label files. Format-3 visit-status bundles record
the temporal holdout, 12-month history anchor, and three active history windows.
Historical format-2 bundles remain readable for provenance. Rebuild them before
training under the temporal holdout.

All-feature and training-selected top-20 forests use labeled visits before the
cutoff from training patients. The same rows determine preprocessing, feature
ranking, and the training-prevalence baseline. All models are evaluated on the
same labeled recent visits from held-out patients. Each patient has equal total
weight within fitting and evaluation.

Outputs include `report.md`, `metrics.json`, saved `VisitStatusModel` pipelines,
feature rankings, and `heldout_status_predictions.jsonl`. Each prediction records
DRE/NDRE percentages, a threshold-0.5 class, and the inclusive history start and
exclusive history end for each model. `heldout_status_predictions.jsonl` contains
recent visits from held-out patients. `status_predictions.jsonl` records all
scored visits, tagged by period and prediction scope.

Performance reports include patient-weighted AUROC, average precision, Brier
score, log loss, accuracy, sensitivity, specificity, precision, F1 and NPV. All
classification metrics use threshold 0.5; NPV is undefined when no negative
predictions exist.

Training automatically saves nine PNG confusion matrices under
`confusion_matrices/`: all-feature, selected-feature, and prevalence-baseline
results for each 3-, 6-, and 9-month history. The report links to each image.
Rows show actual DRE/NDRE status and columns show predicted DRE/NDRE status,
with the predicted labels at the top, at threshold 0.5. DRE is the positive
class. The default blue heatmap reflects visit counts. Each cell shows only an
unweighted held-out visit count and its
percentage within the observed class; N/A indicates an absent observed class.
The report's performance metrics continue to use patient weights.

Training also saves `roc_comparison.png`, overlaying the 3-, 6-, and 9-month
selected-feature models on one ROC plot, with AUROCs in the legend and a chance
diagonal. All three curves use the same labeled held-out visits and equal-patient
weights as the performance metrics. Unknown labels are excluded. Curve points
and AUROCs are saved in `roc_curves.json`, and the report embeds the plot.

Generate plots for an existing visit-status results directory without retraining:

```powershell
.\.venv\Scripts\python.exe "value based model/plot_visit_status.py" --results-dir "value based model/results/rf_visit_status_NEW_RUN"
```

Use `--plot roc` or `--plot confusion` to regenerate a single plot type.

Percentages are uncalibrated estimates of the project-derived status. Labels
are not independently adjudicated outcomes. Observation and linked visit dates
approximate availability because recording timestamps are unavailable. Scores
on visits with unknown outcomes are not validated by the evaluation. Historical
data and model runs remain saved.

## Latest-visit evaluation with provided labels

Use a model run fitted with the 12-month holdout. The evaluator selects each
patient's latest recorded medication visit and retains visits within the saved
window. The supplied `DRE/` and `NDRE/` classifications override computed labels
at those visits, including known computed labels. Patients absent from both
folders retain their computed latest status. Performance uses held-out patients
with a known status at their selected visit.

```powershell
.\.venv\Scripts\python.exe "value based model/evaluate_latest_visits.py" --results-dir "value based model/results/rf_visit_status_YYYYMMDD_HHMMSS_500trees" --output-dir "value based model/results/latest_visit_folder_labels_NEW_RUN" --reference-label-root . --label-overrides "value based model/latest_status_overrides.json"
```

The new output directory contains the report, metrics, ROC comparison, nine
confusion matrices, and `label_audit.jsonl` with both labels, folder source paths,
file checksums, resolved unknowns, and known-label disagreements. Only held-out
patients contribute to performance. Without `--reference-label-root`, the script
evaluates the original computed labels.

Explicit user-confirmed latest-status labels are saved in
`value based model/latest_status_overrides.json` and take precedence over folder
labels when passed with `--label-overrides`. Patient 0258 was confirmed as DRE
on October 24, 2026. These labels apply only to each patient's latest visit and
are recorded with their source in the evaluation audit.

## Feature redundancy and assumption audit

The historical feature redundancy audit used the October 21 bundle and 500-tree
baseline, five patient-group folds, seed 42, and 1,000 paired patient bootstrap
resamples. Its optional runner is absent from this checkout.

## Simplified models and comparison

The restored coverage-containing bundle at
`value based model/data/features/visit_status_20261025_144154` is a historical artifact.
New training uses a fresh standard build with the temporal holdout and automatic
timestamped result folders.
The previous coverage-containing fitted model folder is no longer present in
the workspace. The no-coverage forests in
`value based model/results/rf_visit_status_20261025_no_coverage_500trees` and their
supplied latest-label evaluation in
`value based model/results/latest_visit_20261025_no_coverage` remain historical
results of the removed configuration.
The before/after feature inventory, actual comparison visit IDs/dates, metrics,
paired bootstrap intervals and verification report are in
`value based model/results/actual_visit_comparison_20261025`.

The medication-comparison audit below reproduces historical model comparisons.
It uses the original patient-only split and requires the matching archived
results and feature bundles.

```powershell
.\.venv\Scripts\python.exe "value based model/historical_audit.py" actual-visits --baseline-results "value based model/results/ARCHIVED_BASELINE" --results "value based model/results/rf_visit_status_20261025_181835_500trees" --output "value based model/results/actual_visit_comparison_NEW_RUN"
```

The comparison independently verifies every actual visit pair against the
snapshot's visit markers and date cutoffs. It checks that labels, patient splits,
prediction rows and unrelated feature values match the prior simplified bundle.
All requested metrics have 1,000 paired patient-cluster bootstrap resamples.
Computed historical labels remain primary; supplied latest labels are secondary.
These are the previously evaluated held-out patients. This run uses seed 42 and
training-only top-20 selection, without a new ablation search. Nonsignificant
differences are not evidence of equivalence. Historical models and bundles are
preserved. Earlier audit utility sources are optional; their historical test
module is explicitly skipped when those sources are absent.
All classification summaries include NPV at threshold 0.5 with equal patient
weighting. The command writes `report.md`, the feature and visit-pair evidence,
predictions, metrics, verification checks, and provenance.

## Supporting modules and verification

### Restored seizure coverage fraction predictors

The standard feature builder includes `seizure_coverage_fraction_3m` for the
3-month model, `seizure_coverage_fraction_6m` for the 6-month model, and both
`seizure_coverage_fraction_6m` and `seizure_coverage_fraction_9m` for the 9-month
model. The removal was reverted at the user's request after reviewing the
6-month sensitivity reduction. Seizure counts, zero-seizure days, outcome
labels, and prospective cutoffs retain their existing definitions.

The previous coverage ablation runner and its dedicated tests remain removed.
`historical_audit.py without-coverage` reproduces the historical
`recorded_counts_actual_visits_no_coverage_v3` bundle. Its output is not the
active schema and is rejected by current training. Historical models and
bundles remain readable for reproduction; existing artifacts are not modified.

`sql_snapshot.py` handles snapshots; `longitudinal_features.py` builds clinical
histories; `eeg_features.py` adapts the existing EEG encoder;
`visit_status_targets.py` builds visit labels; `forest_workflow.py` supplies
shared fitting and evaluation; `visit_status_workflow.py` validates bundles,
trains status models, and writes reports. Clinical rules remain under
`Database/dre_onset_analysis/`, with the existing
`docx_parsers/docx_parser_eeg_timeline.py` encoder.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "value based model" -v
.\.venv\Scripts\python.exe "value based model/extract_sql.py" --verify "value based model/data/sql_snapshots/epilepsydb_NEW_RUN"
```

To print a source DOCX table, run `docx_parsers/inspect_docx_tables.py eeg PATH`
or `docx_parsers/inspect_docx_tables.py brain-image PATH`.
Omit the path to use the file or folder picker.

Synthetic fixtures live in `tests/helpers.py`. Tests cover history boundaries,
evidence availability, target timing, model serialization, reports, snapshot
integrity, and separation of extraction, preparation, and training.
