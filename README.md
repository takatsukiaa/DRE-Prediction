# Value-based DRE/NDRE prediction

`train.py` builds pre-visit features or trains random forests estimating any
observed monthly state switch within 6 and 12 months. It reuses one saved
50-50 patient split (132 training, 133 test); no cross-validation is used.
Training ranks all usable candidates on training patients and refits a model
with the top 30 source features separately for each direction and horizon.

## Active scripts

| Path | Purpose | Used by `train.py`? |
|---|---|---|
| `train.py` | Feature-building/training entry point and patient split | Entry point |
| `forest_workflow.py` | Training-only preprocessing, feature ranking, forests, risk outputs, evaluation | With `--train-forest` |
| `risk_targets.py` | Any-switch labels, follow-up coverage, directional risk percentages | Through forest workflow |
| `longitudinal_features.py` | Past-only levels, changes, coverage, outcomes | Directly |
| `eeg_features.py` | Adapt database EEG text to existing ordinal/presence/change features | Through feature builder |
| `sql_snapshot.py` | Load and verify saved snapshot files | Directly |
| `tools/sql_data_loader.py` | Extract or inspect the live database | Separate maintenance command |
| `tools/verify_all_tables.py` | Compare a saved extraction with the live database | Separate maintenance command |
| `tests/test_longitudinal_features.py` | Feature timing, changes, labels, and split regression checks | Test suite |
| `tests/test_eeg_features.py` | EEG encoding, negatives, report timing, and change checks | Test suite |
| `tests/test_sql_snapshot.py` | Snapshot serialization/checksum/schema regression checks | Test suite |
| `tests/test_risk_targets.py` | Intermediate switches, reversals, censoring, monthly boundaries | Test suite |
| `tests/test_forest_workflow.py` | Training-only filtering/ranking, patient weights, model reload, metrics | Test suite |

The feature module also imports audited helpers from
`Database/dre_onset_analysis/analyze_onset.py` and `monthly_dre.py`.
Those files remain required dependencies. Other repository folders support
separate parsing, plotting, image-model, and database workflows.

The EEG adapter imports `docx_parsers/docx_parser_eeg_timeline.py` and reuses
`EegReport`/`parse_report()` directly on `EegRecords.Note`. It does not read DOCX
files or call the abnormal-only DOCX extraction/aggregation path. The original
encoder remains unchanged for existing document/plotting workflows.

The extraction and validation scripts are retained because they are needed to
rebuild or verify data. They are separated from the active training files rather
than treated as dead code merely because training does not call them.

## Commands from the project root

Train the all-feature comparison and fixed top-30 models:

```powershell
.\.venv\Scripts\python.exe "value based model/train.py" --train-forest --panel all --top-features 30 --output-dir "value based model/results/rf_NEW_RUN"
```

The default feature input is `data/features/previsit_20261016_eeg_v1/`.
Each task fits an all-feature forest, ranks source features by training impurity
importance (including their missingness indicators), then fits the selected
forest. Test data never select features, feature count, settings, or thresholds.
A full-panel comparison excluding explicit change features is also reported.
The selected model supplies the saved percentages even if the full model scores
better. These are raw, uncalibrated research estimates.

For a starting NDRE state, the applicable risk is NDRE-to-DRE; for a starting DRE
state it is DRE-to-NDRE. A subsequent reversal does not erase the event. The
other direction is inapplicable; no-switch risk is the complementary percentage.
This concerns observed monthly labels and cannot detect intervening switches
that reverse between monthly assessments.

Results include `report.md`, `metrics.json`, `feature_importance.json` (every
candidate and exclusion reason), `selected_features.json`, saved pipelines,
`switch_targets.jsonl`, and `heldout_risk_predictions.jsonl`. The combined
`risk_predictions.jsonl` includes explicitly marked in-sample training rows.
Unknown starting states receive no directional risk; missing follow-up is not
automatically labeled no-switch. Output folders contain patient-level data and
are ignored by Git. Use a new directory for each run.

Build a new feature bundle:

```powershell
.\.venv\Scripts\python.exe "value based model/train.py" --output-dir "value based model/data/features/previsit_NEW_RUN"
```

Run all local model/snapshot regression tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "value based model" -v
```

Create a new database extraction:

```powershell
.\.venv\Scripts\python.exe "value based model/tools/sql_data_loader.py" --all-tables --output-dir "value based model/data/sql_snapshots/epilepsydb_NEW_RUN"
```

Compare that extraction with a fresh database read:

```powershell
.\.venv\Scripts\python.exe "value based model/tools/verify_all_tables.py" "value based model/data/sql_snapshots/epilepsydb_NEW_RUN"
```

See `FEATURE_BUILDING.md` for output files and feature definitions,
`SQL_DATA_LOADING.md` for extraction details, and `TRAINING_PLAN.md` for the model plan.
