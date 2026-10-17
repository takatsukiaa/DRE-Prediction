# Building pre-visit features

`train.py` is the command-line entry point; `longitudinal_features.py` contains the testable temporal calculations. The user confirmed predictions before each medication visit, the preceding 12 months as the history window, and stored DailyDose in mg/day. The feature bundle retains the originally requested endpoint labels for audit. The current training target is any switch during 6/12-month follow-up, built separately by `risk_targets.py`; it does not use those old endpoint labels or their weights.

Run from the project root, choosing a new output directory for each run:

```powershell
.\.venv\Scripts\python.exe "value based model/train.py" --output-dir "value based model/data/features/previsit_NEW_RUN"
```

The default snapshot is `data/sql_snapshots/epilepsydb_20261016/`, relative to the script. Snapshot loading checks saved checksums. No database connection is made. An existing output directory is never overwritten. The output directory is ignored by Git because it contains patient-level derived data.

The user confirmed `Gender=1` is male and `Gender=0` is female. This is now the default mapping; `--male-code` remains available for explicitly documented alternative extractions. The user also confirmed that unusually large stored doses should be preserved and flagged for review. DailyDose is used as stored without clipping; it is not multiplied by Quantity or frequency. The ten largest source doses are listed by source IDs in the audit for review. Multiple active details for one substance make that substance's dose missing until reconciled. Use `--no-dose-features` to omit dose/DDD predictors while retaining medication presence, counts, and composition changes in a separately declared analysis.

## Features

- Latest scheduled ASM count and normalized substance indicators, exact versus uncertain regimen flags, latest per-drug daily mg, total DDD where complete, and observation ages.
- Last-two-visit drug-count differences, added/stopped substances, substitutions, per-drug dose differences, continuing-drug increase/decrease counts, and total DDD difference. Both visits must lie in the 12-month history. A substitution is detected even if drug count is unchanged.
- Six- and twelve-month counts of observed composition changes with the number of assessable pairs. Time since last observed composition change is missing when none is observed, not zero.
- Interval-aware seizure-count/rate lower bounds, known and zero coverage days, partial positive and unknown intervals, and recent-versus-earlier six-month lower-bound rate differences. Rates are lower bounds over calendar windows, not exact estimates of seizure frequency.
- Recognized numeric laboratory measurements and cognitive totals/domain scores: latest value, latest-minus-previous value, elapsed time, change per 30 days, and observation count. Asterisks after numeric scores are tolerated. Censored or uninterpretable values are not guessed. Different laboratory units remain distinct.
- EEG text features reuse `docx_parser_eeg_timeline.parse_report()` directly on the saved database report text: original ordinal scores, presence flags, latest-minus-previous score changes, and maximum observed scores in 6/12 months. The 27 encoded morphology/location features retain their original scales; these values are not probabilities or a validated continuous biological severity scale.
- EEG normal/probably-normal/review flags, known ED/slow-wave indicators, newly/no-longer detected ED between assessable reports, assessable-report counts, and positive fractions. Sharpish configuration is retained separately and does not alone establish the aggregate ED indicator.
- EEG, focality-record, and brain-imaging examination counts and ages. Focality-record text and imaging clinical interpretations are not yet encoded.
- Age at prediction, confirmed sex encoding where supplied, observed history length, and prior completed monthly state/history. Current month-end labels cannot be used before that assessment date. Unknown months are not replaced with older known values.

Only completed observations strictly before prediction enter a feature row. Seizure intervals linked to a medication visit are available no earlier than both the interval end and linked visit date. Without recording timestamps, dates provide an imperfect availability proxy; this limitation is retained in the audit.

Aggregate each modality separately. Same-day medication visits cause a clear error rather than an invented ordering. Repeated identical numeric results on a date count as one date; conflicting same-date results make that date's value missing. Missing history produces missing changes, not zero.

The column list is a candidate inventory, not a selected model panel. All-missing and constant columns are retained at this stage; later screening and imputation must be fitted using training patients only.

### EEG text adapter

`eeg_features.py` calls the existing report-level encoder on `EegRecords.Note` without reading a DOCX file. It does not call `extract_reports()` or `aggregate_evidence()`: the DOCX workflow excludes normal reports and its blank feature cells do not establish negative findings.

Explicit normal reports and explicit negative final verdicts can provide zero-valued scores. Unmentioned findings remain missing. Empty, unrecognized, probably-normal, or contradictory text is flagged for review; it does not automatically become a normal EEG. Parser evidence snippets, methods, confidence labels, and source IDs are retained in a separate local audit. Parser confidence labels are rule annotations, not calibrated probabilities.

Only report segments dated within the prior 12 months and available strictly before prediction are eligible. Embedded dated reports are segmented using the existing encoder; segment availability is the later of the enclosing database date and embedded report date. This prevents a later embedded report from being backdated. The database date is still a proxy when actual recording timestamps are unavailable. Latest findings are not carried forward over a more recent uninterpretable report.

For multiple reports on the same clinical date, positive/negative conflicts produce missing scores and review flags. Otherwise the largest available ordinal score is used for that date. Same-date aggregation happens after availability filtering at each prediction date. Two distinct dates are required for change features. Review representative normal, positive, ambiguous, and embedded-report cases before relying on the encoded clinical findings.

## Targets and patient split

Each valid medication visit has a prediction row. The archived `target_dre_*` columns describe endpoint status at the month-end containing the visit date plus 6/12 months. They are retained unchanged for reproducibility and are not the targets of the current forest workflow.

Training reconstructs any-switch outcomes from the verified monthly table, for each known starting state. An opposite state at least once strictly after prediction and on/before the exact 6/12-calendar-month anniversary is positive, even if the state subsequently reverses. For January 15, the six-month cutoff is July 15, so assessments January 31 through June 30 are included. A no-switch outcome requires every expected assessment to be known and the complete horizon to have elapsed. Observed switches establish positives even with later missing follow-up; other incomplete intervals remain unknown. Unknown starting states are ineligible for directional models. This estimates observed monthly switches, not all continuous-time clinical transitions.

The unique patient list is split once with `test_size=0.5` and seed 42: 132 training and 133 test patients before task-specific eligibility. Stored monthly-history strata balance allocation where feasible; no seed search or cross-validation runs. Patient assignments apply to all visits and both horizons. The forest workflow recalculates equal-patient weights separately within each direction, horizon, and partition; the old endpoint weights are not reused.

## Output bundle

| File | Contents |
|---|---|
| `features.pkl.gz` | Float32 candidate feature DataFrame, with NaN preserved |
| `visits.jsonl` | Patient/date identifiers, past-state audit fields, future labels/dates, unknown reasons, patient split, and horizon-specific sample weights |
| `feature_dictionary.json` | Ordered feature names, source fields/tables, and definitions |
| `patient_split.json` | Frozen patient-level training/test IDs and seed |
| `audit.json` | Counts, source issues, outcome availability, transition summaries, largest-dose review list, and limitations |
| `eeg_report_audit.jsonl` | Per-report/segment source dates, availability dates, ordinal scores, evidence snippets, and review issues; never model inputs |
| `provenance.json` | Snapshot manifest checksum, code checksums, dependency versions, and row alignment |

Feature row `i` corresponds to JSONL visit line `i+1`; preserve that alignment when applying a horizon-specific eligibility mask. Outcome labels and identifiers are deliberately saved separately from the model input columns.

```python
from pathlib import Path
import pandas as pd

bundle = Path("value based model/data/features/previsit_NEW_RUN")
X = pd.read_pickle(bundle / "features.pkl.gz")
visits = pd.read_json(bundle / "visits.jsonl", lines=True,
                      dtype={"patient_id": "string", "source_asm_record_id": "string"})
# Use train.py --train-forest to rebuild any-switch targets from the snapshot,
# apply task-specific eligibility/weights, and fit preprocessing on training only.
# Do not use the archived target_dre_* columns for the any-switch analysis.
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "value based model" -p test_longitudinal_features.py -v
```

Checks cover substitutions, dose changes, prediction-visit exclusion, future-label independence, unknown monthly states, insufficient/expired history, ambiguous medication snapshots, numeric change timing, different lab units, conflicting same-date values, seizure interval availability/overlap, explicit duplicate/orphan failures, optional dose omission, and reproducible disjoint patient assignments. EEG tests additionally cover score/location reuse, normal versus missing, negation, normal sleep, sharpish wording, report changes, same-day conflicts, and availability of embedded reports.
