r"""Read EpilepsyDB into pandas using the project's existing pyodbc dependency.

Run from the project root:
    .\.venv\Scripts\python.exe "value based model\tools\sql_data_loader.py"
    .\.venv\Scripts\python.exe "value based model\tools\sql_data_loader.py" --patient-id 0003

Run --all-tables to load every base table, including empty and lookup tables.
Add --output-dir PATH to save a complete JSONL snapshot and schema inventory.
Queries read data only. This module does not refresh DRE labels or train a model.
"""
import argparse
import datetime as dt
import json
from contextlib import contextmanager
from pathlib import Path
import sys

import pandas as pd
import pyodbc

# Allow this maintenance command to run directly from any working directory.
MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))


@contextmanager
def connect_epilepsydb(server=r".\SQLEXPRESS"):
    """Use Windows authentication; close the connection even if a query fails."""
    connection = pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};DATABASE=EpilepsyDB;"
        "Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;",
        timeout=10,
    )
    try:
        yield connection
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


def read_dataframe(connection, sql, params=()):
    """Fetch with pyodbc and construct a DataFrame without a SQLAlchemy dependency."""
    cursor = connection.cursor()
    try:
        cursor.execute(sql, *params)
        columns = [column[0] for column in cursor.description]
        return pd.DataFrame.from_records([tuple(row) for row in cursor.fetchall()], columns=columns)
    finally:
        cursor.close()


def _month(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = dt.date.fromisoformat(value + "-01" if len(value) == 7 else value)
    if isinstance(value, dt.datetime):
        value = value.date()
    if not isinstance(value, dt.date):
        raise TypeError("Month must be YYYY-MM, YYYY-MM-DD, or a Python date")
    return value.replace(day=1)


def load_monthly_dre(connection, patient_id=None, start_month=None, end_month=None):
    """One row per patient-month, including Unknown labels as pandas NA.

    Both month bounds are inclusive. Patient IDs must be strings, e.g. '0003'.
    Medication counts and seizure counts describe the label month; they are not
    automatically appropriate predictors of that same label.
    """
    start, end = _month(start_month), _month(end_month)
    if start is not None and end is not None and start > end:
        raise ValueError("start_month must not be later than end_month")
    if patient_id is not None and not isinstance(patient_id, str):
        raise TypeError("Use a string patient ID to preserve leading zeros")
    filters, params = [], []
    for expression, value in (("PatientId = ?", patient_id), ("MonthStart >= ?", start), ("MonthStart <= ?", end)):
        if value is not None:
            filters.append(expression)
            params.append(value)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    sql = """
        SELECT PatientId,MonthStart,AssessmentDate,WindowStart,IsDRE,
               AsmCount,AsmRecordId,AsmRecordDate,AsmRecordAgeDays,
               SeizureCountLowerBound,SeizureCoverageComplete,Reason,EvidenceJson,
               CalculationAsOf,CalculatedAtUtc,MethodVersion,SourceFingerprint
        FROM dbo.PatientMonthlyDreStatus
    """ + where + " ORDER BY PatientId,MonthStart"
    frame = read_dataframe(connection, sql, params)
    frame["PatientId"] = frame["PatientId"].astype("string")
    for column in ("MonthStart", "AssessmentDate", "WindowStart", "AsmRecordDate", "CalculationAsOf", "CalculatedAtUtc"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in ("AsmCount", "AsmRecordId", "AsmRecordAgeDays", "SeizureCountLowerBound"):
        frame[column] = frame[column].astype("Int64")
    frame["IsDRE"] = frame["IsDRE"].astype("Int8")
    frame["SeizureCoverageComplete"] = frame["SeizureCoverageComplete"].astype("boolean")
    if frame.duplicated(["PatientId", "MonthStart"]).any():
        raise ValueError("Duplicate patient-month labels in the database")
    if not frame["IsDRE"].dropna().isin([0, 1]).all():
        raise ValueError("DRE labels must be 0, 1, or unknown")
    return frame


RAW_QUERIES = {
    "asm_records": """SELECT Id AS AsmRecordId, PatientBaselineViewModelPatientId AS PatientId,
        CONVERT(varchar(10),Date,23) AS Date FROM dbo.AsmRecords
        ORDER BY PatientBaselineViewModelPatientId,Date,Id""",
    "asm_drugs": """SELECT d.Id AS DrugDetailId,a.PatientBaselineViewModelPatientId AS PatientId,
        d.AsmRecordViewModelId AS AsmRecordId,d.DrugName,d.Quantity,d.Unit,d.MgPerUnit,d.DailyDose,d.Frequency
        FROM dbo.AsmDrugDetail d JOIN dbo.AsmRecords a ON a.Id=d.AsmRecordViewModelId
        ORDER BY a.PatientBaselineViewModelPatientId,d.AsmRecordViewModelId,d.Id""",
    "seizures": """SELECT Id AS SeizureIntervalId,PatientBaselineViewModelPatientId AS PatientId,
        DateString AS IntervalStart,EndDateString AS IntervalEnd,SeizureCount,RecordType
        FROM dbo.SeizureEventViewModel ORDER BY PatientBaselineViewModelPatientId,DateString,Id""",
}


def load_raw_tables(connection):
    """Separate source tables for feature engineering; dates/counts retain source meaning.

    Raw dates remain strings so invalid/future source values are visible for
    explicit cleaning. Avoid a many-to-many join between drugs and seizures.
    """
    tables = {}
    for name, sql in RAW_QUERIES.items():
        frame = read_dataframe(connection, sql)
        frame["PatientId"] = frame["PatientId"].astype("string")
        if name == "seizures":
            frame["SeizureCount"] = frame["SeizureCount"].astype("Int64")
        tables[name] = frame
    return tables


def load_database_snapshot(connection, progress=None):
    """All tables plus schema, foreign keys, and feature-review inventory."""
    from sql_snapshot import extract_snapshot
    return extract_snapshot(connection,progress=progress)


def load_all_tables(connection, progress=None):
    """Return every base table keyed by schema-qualified names, e.g. dbo.Patients.

    Original column names, missing values, and SQL timestamp text are preserved.
    Close the connection context after loading to release consistency locks.
    """
    return load_database_snapshot(connection,progress=progress).tables


def load_saved_snapshot(directory):
    """Reload a complete extraction into DataFrames without connecting to SQL."""
    from sql_snapshot import read_snapshot
    return read_snapshot(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=r".\SQLEXPRESS")
    parser.add_argument("--patient-id")
    parser.add_argument("--start-month", help="Inclusive YYYY-MM")
    parser.add_argument("--end-month", help="Inclusive YYYY-MM")
    parser.add_argument("--include-raw", action="store_true", help="Also load ASM and seizure source tables and print their row counts")
    parser.add_argument("--all-tables", action="store_true", help="Read all clinical, lookup, label, and metadata base tables")
    parser.add_argument("--output-dir",type=Path,help="New directory for a complete JSONL snapshot; requires --all-tables")
    args = parser.parse_args()
    if args.output_dir is not None and not args.all_tables:
        parser.error("--output-dir requires --all-tables")
    if args.all_tables and (args.patient_id is not None or args.start_month is not None or args.end_month is not None or args.include_raw):
        parser.error("--all-tables extracts the full database; do not combine it with monthly filters or --include-raw")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("Output directory already exists; select a new snapshot directory")
    if args.all_tables:
        with connect_epilepsydb(args.server) as connection:
            snapshot = load_database_snapshot(connection,progress=lambda name,count:print(f"{name}: {count:,} rows"))
        # All persistence and profiling output happen after the database context closes.
        summary = dict(tables=len(snapshot.tables),total_rows=sum(len(t) for t in snapshot.tables.values()),
                       empty_tables=[name for name,frame in snapshot.tables.items() if frame.empty])
        if args.output_dir is not None:
            from sql_snapshot import write_snapshot
            manifest = write_snapshot(snapshot,args.output_dir)
            summary.update(output_dir=str(args.output_dir.resolve()),saved_tables=manifest["table_count"])
        print(json.dumps(summary,indent=2))
        return
    with connect_epilepsydb(args.server) as connection:
        frame = load_monthly_dre(connection,args.patient_id,args.start_month,args.end_month)
        raw = load_raw_tables(connection) if args.include_raw else None
    summary = dict(patient_month_rows=len(frame),patients=int(frame["PatientId"].nunique()),
                   label_yes=int(frame["IsDRE"].eq(1).sum()),label_no=int(frame["IsDRE"].eq(0).sum()),
                   label_unknown=int(frame["IsDRE"].isna().sum()))
    if raw is not None:
        summary["raw_rows"] = {name:len(table) for name,table in raw.items()}
    print(json.dumps(summary,indent=2))
    print(frame[["PatientId","MonthStart","IsDRE","AsmCount","SeizureCountLowerBound"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
