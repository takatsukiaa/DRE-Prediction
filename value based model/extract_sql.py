r"""Extract every EpilepsyDB base table into a new read-only source snapshot.

Run from the project root:
    .\.venv\Scripts\python.exe "value based model/extract_sql.py" --output-dir PATH

Uses Windows authentication. Does not calculate labels, build features, or train.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from sql_snapshot import extract_snapshot, write_snapshot


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default=r'.\SQLEXPRESS')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwritten')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new snapshot directory')
    with connect_epilepsydb(args.server) as connection:
        snapshot = extract_snapshot(connection, progress=lambda name, count: print(f'{name}: {count:,} rows', flush=True))
    manifest = write_snapshot(snapshot, args.output_dir)
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), tables=manifest['table_count'],
                          total_rows=manifest['total_rows']), indent=2))


if __name__ == '__main__':
    main()
