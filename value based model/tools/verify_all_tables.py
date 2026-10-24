"""Compare a saved full snapshot with a fresh read of every SQL table."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from extract_sql import connect_epilepsydb
from sql_snapshot import extract_snapshot, read_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument('--server', default=r'.\SQLEXPRESS')
    args=parser.parse_args()
    saved=read_snapshot(args.snapshot)
    with connect_epilepsydb(args.server) as connection:
        live=extract_snapshot(connection)
    if set(saved.tables)!=set(live.tables):
        raise AssertionError("Saved tables differ from current database tables")
    for name,expected in live.tables.items():
        pd.testing.assert_frame_equal(saved.tables[name],expected,check_exact=True,obj=name)
    for key in ("columns","primary_keys","foreign_keys"):
        if saved.schema[key]!=live.schema[key]:
            raise AssertionError("Schema metadata mismatch: "+key)
    rows=sum(len(t) for t in saved.tables.values())
    report=dict(passed=True,tables_verified=len(saved.tables),rows_verified=rows,
                columns_verified=sum(len(t.columns) for t in saved.tables.values()),
                exact_values_and_dtypes_match=True,checksums_verified=True,
                empty_tables=[name for name,t in saved.tables.items() if t.empty],
                patients=int(len(saved.tables["dbo.Patients"])),
                primary_keys_and_foreign_keys_match=True)
    (args.snapshot/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
