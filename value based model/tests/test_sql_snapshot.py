"""Snapshot serialization and schema checks; no database connection required."""
import datetime as dt
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sql_snapshot import (DatabaseSnapshot, _projection, _typed_frame,
                          make_inventory, quote_identifier, read_snapshot, write_snapshot)


def column(table,name,kind,ordinal=1):
    return dict(SchemaName="dbo",TableName=table,ColumnName=name,SqlType=kind,Ordinal=ordinal)


def local_temp_directory():
    # Verify the actual cleanup target is inside the workspace before the
    # TemporaryDirectory context can recursively remove its own test files.
    root=Path(__file__).resolve().parent
    directory=tempfile.TemporaryDirectory(prefix=".snapshot-test-",dir=root)
    if not Path(directory.name).resolve().is_relative_to(root):
        raise ValueError("Test temporary directory escaped the workspace")
    return directory


def example_snapshot():
    columns=[column("Patients","PatientId","nvarchar"),column("Patients","IsDRE","bit",2),
             column("Patients","Timestamp","datetimeoffset",3),column("Patients","ExactValue","decimal",4),
             column("Patients","BinaryValue","varbinary",5),column("Empty","Id","bigint")]
    schema=dict(format_version=1,database="EpilepsyDB",server="test",extracted_at_utc="2026-10-16T00:00:00+00:00",
                tables=[dict(SchemaName="dbo",TableName="Patients"),dict(SchemaName="dbo",TableName="Empty")],
                columns=columns,primary_keys=[dict(SchemaName="dbo",TableName="Patients",ColumnName="PatientId")],foreign_keys=[])
    records=[dict(PatientId="0001",IsDRE=None,Timestamp="5017-04-13T10:20:30.1234567+08:00",ExactValue=Decimal("123.456789123456789"),BinaryValue=b"\x00\xff"),
             dict(PatientId="0002",IsDRE=False,Timestamp=None,ExactValue=None,BinaryValue=None)]
    tables={"dbo.Patients":_typed_frame(records,columns[:5]),"dbo.Empty":_typed_frame([],columns[5:])}
    return DatabaseSnapshot(tables,schema,make_inventory(tables,schema))


class SqlSnapshotTests(unittest.TestCase):
    def test_bigint_null_does_not_round_to_float(self):
        frame=_typed_frame([dict(Id=9223372036854775807),dict(Id=None)],[column("T","Id","bigint")])
        self.assertEqual(int(frame.Id.iloc[0]),9223372036854775807)
        self.assertTrue(pd.isna(frame.Id.iloc[1]))
        self.assertEqual(str(frame.Id.dtype),"Int64")

    def test_identifiers_empty_strings_and_nulls_remain_distinct(self):
        frame=_typed_frame([dict(PatientId="0001"),dict(PatientId=""),dict(PatientId=None)],[column("T","PatientId","nvarchar")])
        self.assertEqual(frame.PatientId.iloc[0],"0001")
        self.assertEqual(frame.PatientId.iloc[1],"")
        self.assertTrue(pd.isna(frame.PatientId.iloc[2]))

    def test_bit_preserves_unknown_instead_of_false(self):
        frame=_typed_frame([dict(IsDRE=True),dict(IsDRE=False),dict(IsDRE=None)],[column("T","IsDRE","bit")])
        self.assertEqual(str(frame.IsDRE.dtype),"boolean")
        self.assertTrue(pd.isna(frame.IsDRE.iloc[2]))

    def test_projection_preserves_timestamp_precision_and_quotes_names(self):
        sql=_projection([column("T","LastUpdatedAt","datetimeoffset"),column("T","odd]name","nvarchar")])
        self.assertIn("CONVERT(varchar(48),[LastUpdatedAt],127)",sql)
        self.assertIn("[odd]]name]",sql)
        self.assertEqual(quote_identifier("a]b"),"[a]]b]")

    def test_full_round_trip_including_empty_table_decimal_binary_and_dates(self):
        snapshot=example_snapshot()
        with local_temp_directory() as tmp:
            directory=Path(tmp)/"new"
            manifest=write_snapshot(snapshot,directory)
            actual=read_snapshot(directory)
            self.assertEqual(manifest["table_count"],2)
            self.assertEqual(manifest["total_rows"],2)
            self.assertEqual(actual.schema,snapshot.schema)
            for name in snapshot.tables:
                pd.testing.assert_frame_equal(actual.tables[name],snapshot.tables[name])
            self.assertEqual(list(actual.tables["dbo.Empty"].columns),["Id"])
            self.assertEqual(str(actual.tables["dbo.Empty"].Id.dtype),"Int64")

    def test_existing_output_directory_is_not_overwritten(self):
        with local_temp_directory() as tmp:
            directory=Path(tmp)/"new"
            write_snapshot(example_snapshot(),directory)
            before=(directory/"manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_snapshot(example_snapshot(),directory)
            self.assertEqual((directory/"manifest.json").read_bytes(),before)

    def test_changed_table_file_is_detected(self):
        with local_temp_directory() as tmp:
            directory=Path(tmp)/"new"
            manifest=write_snapshot(example_snapshot(),directory)
            with (directory/manifest["tables"][0]["file"]).open("a",encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaises(ValueError):
                read_snapshot(directory)

    def test_patient_join_path_and_feature_flags(self):
        snapshot=example_snapshot()
        snapshot.schema["foreign_keys"]=[dict(ConstraintName="FK_Empty_Patients",ChildSchema="dbo",ChildTable="Empty",ChildColumn="Id",ParentSchema="dbo",ParentTable="Patients",ParentColumn="PatientId")]
        inventory=make_inventory(snapshot.tables,snapshot.schema)
        self.assertIn("PatientId",inventory[0]["feature_review_flags"])
        self.assertIn("IsDRE",inventory[0]["feature_review_flags"])
        self.assertEqual(inventory[1]["patient_join_path"][0]["columns"],[dict(child="Id",parent="PatientId")])
        self.assertIsNone(inventory[1]["column_profile"][0]["missing_fraction"])


if __name__=="__main__":
    unittest.main()
