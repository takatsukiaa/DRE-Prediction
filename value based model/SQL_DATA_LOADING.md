# Pull EpilepsyDB data into Python

The project's `.venv` already includes pandas, pyodbc, and ODBC Driver 18 for SQL Server. The loader uses Windows authentication to connect to `.\SQLEXPRESS`, database `EpilepsyDB`. Its built-in queries only read data.

## Complete extraction for feature selection

All **32 base tables, 220,509 rows, and 340 columns** were extracted on 2026-10-16 and verified against a second live database read. This includes every clinical table, lookup/settings table, the derived DRE labels, migration metadata, and the two empty tables. The preserved snapshot is:

`value based model/data/sql_snapshots/epilepsydb_20261016/`

The snapshot contains one JSONL file per table, `schema.json` with original SQL types and primary/foreign keys, `inventory.json` with missingness profiles and feature-review flags, and `manifest.json` with row counts and SHA-256 checksums. `verification.json` records the successful comparison with the database. Complete local snapshots are excluded from Git.

Load **every table directly from SQL**:

```python
from tools.sql_data_loader import connect_epilepsydb, load_all_tables

with connect_epilepsydb() as connection:
    tables = load_all_tables(connection)

patients = tables["dbo.Patients"]
asm_records = tables["dbo.AsmRecords"]
asm_drugs = tables["dbo.AsmDrugDetail"]
seizures = tables["dbo.SeizureEventViewModel"]
eeg = tables["dbo.EegRecords"]
biochemistry = tables["dbo.BioChemRecords"]
mmse = tables["dbo.MmseRecords"]
```

Every dictionary value is a pandas DataFrame with **original database column names**, rather than the aliases used by the smaller `load_raw_tables` helper. For example, full-table ASM drug details link via `AsmRecordViewModelId`, and ASM visit records retain their `Id` primary key and `PatientBaselineViewModelPatientId` foreign key. Retrieve all declared relationships from `schema.json` instead of guessing joins.

Load the **saved extraction without a SQL connection**:

```python
from pathlib import Path
from tools.sql_data_loader import load_saved_snapshot

# When this script is beside sql_data_loader.py:
snapshot_dir = Path(__file__).resolve().parent / "data/sql_snapshots/epilepsydb_20261016"
snapshot = load_saved_snapshot(snapshot_dir)
tables = snapshot.tables
patients = tables["dbo.Patients"]

for table in snapshot.inventory:
    print(table["table"], table["rows"], table["numeric_columns_to_review"])
```

For richer live metadata, use `load_database_snapshot(connection)`, which returns the same `.tables`, `.schema`, and `.inventory` interface. Snapshot reload verifies all saved checksums. Missing values remain missing, identifiers remain strings, and nullable integers retain their exact values. Empty tables retain their columns and dtypes.

Full extraction represents SQL temporal columns as **ISO text**, converted inside SQL Server. This preserves time-zone offsets, seven-digit timestamp fractions, and out-of-range years needing review. Original SQL types and precision remain in the schema. Convert the appropriate time fields to pandas datetimes when defining feature windows; review invalid/future source dates explicitly. The specialized `load_monthly_dre` function still returns parsed datetime columns and nullable integer 0/1/NA labels.

To create a new full snapshot, choose a new directory:

```powershell
.\.venv\Scripts\python.exe "value based model\tools\sql_data_loader.py" --all-tables --output-dir "value based model\data\sql_snapshots\epilepsydb_NEW_RUN"
```

Extraction uses a consistent serializable read transaction, verifies each table's row count, and closes the SQL connection before writing the local snapshot. It refuses to overwrite an existing directory. Without `--output-dir`, `--all-tables` only loads the DataFrames and prints an inventory. All-table mode is unfiltered and cannot be combined with patient/month filters.

`train.py` now builds historical features from the verified saved snapshot, attaches stored future monthly outcomes, and saves one 50-50 patient assignment. It does not connect to SQL or fit a random forest. See `FEATURE_BUILDING.md` for commands, output files, and current limitations.

## Review candidate feature sources

| Domain | Source tables |
|---|---|
| Patient baseline | Patients |
| ASM exposure and regimen | AsmRecords, AsmDrugDetail, AdverseReactionDetail |
| Seizure history | SeizureEventViewModel |
| EEG | EegRecords, EegFocalityRecords |
| Brain imaging | BrainImages |
| Laboratory measurements | BioChemRecords, HematologyRecords, ImmunoRecords, DrugLevelRecords, AntigenRecords |
| Cognitive measurements | MmseRecords, CasiRecords, WaisRecords |
| Hospitalization and pregnancy | HospitalizationRecords, PregnancyRecords |
| Outcome labels and provenance | PatientMonthlyDreStatus |
| Category definitions and units | The settings tables |

`LabRecords` and `MonthlyRecords` are empty. Laboratory observations are present in the more specific laboratory tables listed above. Some measurement and cognitive-score fields are stored as strings; review their values and convert valid numeric scores explicitly.

The inventory provides per-column SQL types, pandas dtypes, missing counts/fractions, distinct-value counts, and the shortest declared foreign-key path to `Patients`, where one exists. `numeric_columns_to_review` is an initial list rather than an automatic predictor selection. The complete raw extraction includes identifiers and target fields; inventory flags identify these for exclusion or timing review. In particular:

- Patient IDs, medical record numbers, and record keys serve joins/grouping rather than predictors.
- `Patients.IsDRE`, `DreStatusDetail`, `DrugRefractoryDetail`, and derived-label provenance can reveal the outcome.
- `Age2026` should be replaced by age at the prediction date.
- Latest-seizure, reminder, and other current-state fields may have been updated after the intended prediction date.
- Free text may mention DRE status or future outcomes and requires explicit review before encoding.

Aggregate each clinical table to the chosen patient/feature window before joining the modalities. Keep tables separate until then so multiple medication, seizure, EEG, and laboratory observations do not multiply each other.

## Run from the project root

```powershell
.\.venv\Scripts\python.exe "value based model\tools\sql_data_loader.py" --include-raw
.\.venv\Scripts\python.exe "value based model\tools\sql_data_loader.py" --patient-id 0003 --start-month 2015-01 --end-month 2016-12
```

The smaller monthly mode prints counts and a preview. No patient-data files are written without `--output-dir`. The connection uses certificate trust for the existing local SQL Server instance.

## Use in a training script

The training entry point loads saved data directly through `sql_snapshot.py`. For separate live database scripts with the model directory on the Python path, import the maintenance API:

```python
from tools.sql_data_loader import connect_epilepsydb, load_monthly_dre, load_raw_tables

with connect_epilepsydb() as connection:
    monthly = load_monthly_dre(connection)
    raw = load_raw_tables(connection)

asm_visits = raw["asm_records"]
asm_drugs = raw["asm_drugs"]
seizures = raw["seizures"]

# Known monthly labels only. Unknown is not a negative label.
known = monthly.loc[monthly["IsDRE"].notna()].copy()
known["IsDRE"] = known["IsDRE"].astype("int8")

# To inspect a patient's entire monthly timeline:
timeline = monthly.loc[monthly["PatientId"] == "0003"].copy()
```

`monthly` has one row per patient-month. `PatientId` is a string, dates are pandas datetimes, and `IsDRE` is nullable integer `0`, `1`, or `pd.NA`. Optional patient and month filters are parameterized; month bounds are inclusive.

The source tables stay separate. ASM drug rows link to visits by `AsmRecordId`. Seizure records are intervals, not exact event dates. Source date strings and missing counts are retained, so invalid/future dates and unknown observations must be handled explicitly during feature engineering.

## Connect extraction to model training

1. Choose the prediction date, feature window, and future outcome window before assembling a training table.
2. For future DRE development, keep patients already positive before the end of the feature window out of the incident-development cohort. Early converters during the feature window require a separate task or an earlier prediction date.
3. Aggregate each modality separately using information available by the prediction date, then merge the resulting patient/window features. Do not directly join all medication rows to all seizure intervals: that multiplies records.
4. Construct the future outcome from subsequent monthly DRE labels, retaining unknown follow-up as indeterminate unless a positive event establishes the outcome. Do not convert unknown outcome months to negatives.
5. Split by patient so all windows for one patient stay in the same fold. Fit preprocessing on the training fold only.

The monthly table's same-month `AsmCount` and `SeizureCountLowerBound` are components of its `IsDRE` label. Using them to predict that same label primarily reproduces the definition. `Reason`, `EvidenceJson`, and source audit fields are provenance, not automatically model inputs. For future prediction, features must precede the outcome window.

The monthly label table is a stored calculation, not recomputed by this loader. `CalculationAsOf`, `CalculatedAtUtc`, and `MethodVersion` identify its provenance. Refresh it with the existing monthly refresh script when you intentionally need updated labels.

This loader fetches through a pyodbc cursor and constructs a DataFrame directly. It does not pass a pyodbc connection into `pandas.read_sql_query`, whose documented generic database interface uses SQLAlchemy or ADBC; only sqlite3 is supported for the bare DB-API fallback.

References: https://github.com/mkleehammer/pyodbc and https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html
