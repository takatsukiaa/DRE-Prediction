"""Shared synthetic fixtures and workspace-scoped temporary directories."""
import datetime as dt
from decimal import Decimal
from pathlib import Path
import tempfile

import pandas as pd
from sql_snapshot import DatabaseSnapshot, _typed_frame, make_inventory


def drug(name, dose=100, record=1, **extra):
    return dict(Id=record * 10, AsmRecordViewModelId=record, DrugName=name,
                DailyDose=dose, Quantity=1, Frequency='BID', WhoDddValue=100, **extra)


def fixture():
    return {
        'dbo.Patients': pd.DataFrame([dict(PatientId='0001', Birthday='1980-01-01', Gender='Male'),
                                      dict(PatientId='0002', Birthday='1990-01-01', Gender='Female')]),
        'dbo.AsmRecords': pd.DataFrame([dict(Id=i, Date=date, PatientBaselineViewModelPatientId='0001')
                                       for i, date in enumerate(['2019-11-01', '2019-12-01', '2020-01-15'], 1)]),
        'dbo.AsmDrugDetail': pd.DataFrame([drug('Lamotrigine', record=1), drug('Valproate', record=1),
            drug('Lamotrigine', dose=200, record=2), drug('Levetiracetam', record=2),
            drug('Carbamazepine', dose=900, record=3)]),
        'dbo.PatientMonthlyDreStatus': pd.DataFrame([
            dict(PatientId='0001', MonthStart='2019-12-01', AssessmentDate='2019-12-31', IsDRE=1),
            dict(PatientId='0001', MonthStart='2020-01-01', AssessmentDate='2020-01-31', IsDRE=0),
            dict(PatientId='0001', MonthStart='2020-07-01', AssessmentDate='2020-07-31', IsDRE=0),
            dict(PatientId='0001', MonthStart='2021-01-01', AssessmentDate='2021-01-31', IsDRE=None)]),
    }


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


