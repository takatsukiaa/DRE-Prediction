r"""Extract or verify an EpilepsyDB snapshot using read-only SQL queries.

Run from the project root:
    .\.venv\Scripts\python.exe "value based model/extract_sql.py" --output-dir PATH

Uses Windows authentication.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import pandas as pd

from sql_snapshot import extract_snapshot, read_snapshot, write_snapshot


@contextmanager
def connect_epilepsydb(server=r'.\SQLEXPRESS'):
    import pyodbc
    connection = pyodbc.connect(
        'DRIVER={ODBC Driver 18 for SQL Server};'
        f'SERVER={server};DATABASE=EpilepsyDB;'
        'Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;', timeout=10)
    try:
        yield connection
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


def verify_snapshot(directory, server):
    saved = read_snapshot(directory)
    with connect_epilepsydb(server) as connection:
        live = extract_snapshot(connection)
    if set(saved.tables) != set(live.tables):
        raise AssertionError('Saved tables differ from current database tables')
    for name, expected in live.tables.items():
        pd.testing.assert_frame_equal(saved.tables[name], expected, check_exact=True, obj=name)
    for key in ('columns', 'primary_keys', 'foreign_keys'):
        if saved.schema[key] != live.schema[key]:
            raise AssertionError('Schema metadata mismatch: ' + key)
    report = dict(passed=True, tables_verified=len(saved.tables),
                  rows_verified=sum(len(table) for table in saved.tables.values()),
                  columns_verified=sum(len(table.columns) for table in saved.tables.values()),
                  exact_values_and_dtypes_match=True, checksums_verified=True,
                  empty_tables=[name for name, table in saved.tables.items() if table.empty],
                  patients=len(saved.tables['dbo.Patients']), primary_keys_and_foreign_keys_match=True)
    (directory / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default=r'.\SQLEXPRESS')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--output-dir', type=Path, help='Extract to a new directory')
    action.add_argument('--verify', type=Path, metavar='SNAPSHOT', help='Compare a saved snapshot to the current database')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_snapshot(args.verify, args.server), indent=2))
        return
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new snapshot directory')
    with connect_epilepsydb(args.server) as connection:
        snapshot = extract_snapshot(connection, progress=lambda name, count: print(f'{name}: {count:,} rows', flush=True))
    manifest = write_snapshot(snapshot, args.output_dir)
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), tables=manifest['table_count'],
                          total_rows=manifest['total_rows']), indent=2))


if __name__ == '__main__':
    main()
