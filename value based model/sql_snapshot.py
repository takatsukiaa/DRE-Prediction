"""Complete, read-only SQL Server extraction and portable local JSONL snapshots."""
import base64
import collections
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import pandas as pd

TABLES_SQL = """SELECT s.name AS SchemaName,t.name AS TableName FROM sys.tables t
JOIN sys.schemas s ON s.schema_id=t.schema_id WHERE t.is_ms_shipped=0 ORDER BY s.name,t.name"""
COLUMNS_SQL = """SELECT s.name AS SchemaName,t.name AS TableName,c.name AS ColumnName,
c.column_id AS Ordinal,ty.name AS SqlType,c.max_length AS MaxLengthBytes,c.precision AS [Precision],
c.scale AS Scale,c.is_nullable AS IsNullable,c.is_identity AS IsIdentity,c.is_computed AS IsComputed,
c.collation_name AS CollationName,dc.definition AS DefaultDefinition
FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON ty.user_type_id=c.user_type_id
LEFT JOIN sys.default_constraints dc ON dc.object_id=c.default_object_id
WHERE t.is_ms_shipped=0 ORDER BY s.name,t.name,c.column_id"""
KEYS_SQL = """SELECT s.name AS SchemaName,t.name AS TableName,c.name AS ColumnName,
ic.key_ordinal AS KeyOrdinal,k.name AS ConstraintName FROM sys.key_constraints k
JOIN sys.tables t ON t.object_id=k.parent_object_id JOIN sys.schemas s ON s.schema_id=t.schema_id
JOIN sys.index_columns ic ON ic.object_id=t.object_id AND ic.index_id=k.unique_index_id
JOIN sys.columns c ON c.object_id=t.object_id AND c.column_id=ic.column_id
WHERE k.type='PK' AND t.is_ms_shipped=0 ORDER BY s.name,t.name,ic.key_ordinal"""
FOREIGN_KEYS_SQL = """SELECT fk.name AS ConstraintName,cs.name AS ChildSchema,ct.name AS ChildTable,
cc.name AS ChildColumn,ps.name AS ParentSchema,pt.name AS ParentTable,pc.name AS ParentColumn,
fc.constraint_column_id AS ColumnOrdinal,fk.delete_referential_action_desc AS OnDelete,
fk.update_referential_action_desc AS OnUpdate,fk.is_disabled AS IsDisabled
FROM sys.foreign_keys fk JOIN sys.foreign_key_columns fc ON fc.constraint_object_id=fk.object_id
JOIN sys.tables ct ON ct.object_id=fc.parent_object_id JOIN sys.schemas cs ON cs.schema_id=ct.schema_id
JOIN sys.columns cc ON cc.object_id=ct.object_id AND cc.column_id=fc.parent_column_id
JOIN sys.tables pt ON pt.object_id=fc.referenced_object_id JOIN sys.schemas ps ON ps.schema_id=pt.schema_id
JOIN sys.columns pc ON pc.object_id=pt.object_id AND pc.column_id=fc.referenced_column_id
ORDER BY cs.name,ct.name,fk.name,fc.constraint_column_id"""
TEXT_TYPES = {"char","nchar","varchar","nvarchar","text","ntext","uniqueidentifier","xml"}
TEMPORAL_TYPES = {"date","datetime","datetime2","smalldatetime","datetimeoffset","time"}
INTEGER_TYPES = {"tinyint","smallint","int","bigint"}


@dataclass
class DatabaseSnapshot:
    tables: dict
    schema: dict
    inventory: list


def quote_identifier(name):
    return "[" + name.replace("]", "]]") + "]"


def _query(connection, sql):
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        names = [c[0] for c in cursor.description]
        return [dict(zip(names,row)) for row in cursor.fetchall()]
    finally:
        cursor.close()


def _typed_frame(records, columns):
    """Construct nullable columns before pandas can coerce bigint+NULL to float."""
    result = {}
    for column in columns:
        name, kind = column["ColumnName"], column["SqlType"].lower()
        values = [r[name] for r in records]
        dtype = "Int64" if kind in INTEGER_TYPES else "boolean" if kind=="bit" else "Float64" if kind in {"float","real"} else "string" if kind in TEXT_TYPES | TEMPORAL_TYPES else "object"
        result[name] = pd.Series(values,dtype=dtype)
    return pd.DataFrame(result)


def _projection(columns):
    parts = []
    for column in columns:
        name = quote_identifier(column["ColumnName"])
        if column["SqlType"].lower() in TEMPORAL_TYPES:
            # SQL-side conversion preserves datetime2(7)/datetimeoffset(7), avoids
            # pyodbc's unsupported datetimeoffset type, and retains years like 5017.
            parts.append(f"CONVERT(varchar(48),{name},127) AS {name}")
        else:
            parts.append(name)
    return ",".join(parts)


def _patient_paths(foreign_keys):
    edges = collections.defaultdict(dict)
    for fk in foreign_keys:
        child = (fk["ChildSchema"],fk["ChildTable"])
        key = fk["ConstraintName"]
        if key not in edges[child]:
            edges[child][key] = dict(constraint=key,child_schema=child[0],child_table=child[1],
                                    parent_schema=fk["ParentSchema"],parent_table=fk["ParentTable"],columns=[])
        edges[child][key]["columns"].append(dict(child=fk["ChildColumn"],parent=fk["ParentColumn"]))
    def find(start):
        queue = collections.deque([(start,[])])
        seen = set()
        while queue:
            node,path = queue.popleft()
            if node==("dbo","Patients"):
                return path
            if node in seen:
                continue
            seen.add(node)
            for edge in edges[node].values():
                queue.append(((edge["parent_schema"],edge["parent_table"]),path+[edge]))
        return None
    return find


def make_inventory(tables, schema):
    paths = _patient_paths(schema["foreign_keys"])
    inventory = []
    for table in schema["tables"]:
        schema_name,name = table["SchemaName"],table["TableName"]
        key = schema_name+"."+name
        frame = tables[key]
        columns = [c for c in schema["columns"] if (c["SchemaName"],c["TableName"])==(schema_name,name)]
        pk = [c["ColumnName"] for c in schema["primary_keys"] if (c["SchemaName"],c["TableName"])==(schema_name,name)]
        fk = [c["ChildColumn"] for c in schema["foreign_keys"] if (c["ChildSchema"],c["ChildTable"])==(schema_name,name)]
        role = "database_metadata" if name.startswith("__") else "lookup_settings" if name.endswith("Settings") else "derived_labels_and_audit" if name=="PatientMonthlyDreStatus" else "patient_baseline" if name=="Patients" else "clinical_records"
        flags,profile = {},[]
        for column in columns:
            field,kind = column["ColumnName"],column["SqlType"].lower()
            reasons = []
            if field in set(pk+fk) or field in {"PatientId","MedicalRecordNumber"}:
                reasons.append("Identifier/join key; exclude from predictors")
            if role=="derived_labels_and_audit" or field in {"IsDRE","DreStatusDetail","DrugRefractoryDetail"}:
                reasons.append("Outcome or label provenance; do not use as a predictor")
            if field=="Age2026":
                reasons.append("Age at fixed future year; derive age at prediction date instead")
            if field in {"LastSeizure","LastUpdatedAt"} or "Reminder" in field:
                reasons.append("Latest-state/workflow field; verify availability at prediction time")
            if any(token in field.lower() for token in ("note","reason","plan","description")) and kind in TEXT_TYPES:
                reasons.append("Free text; review timing and outcome mentions before encoding")
            if reasons:
                flags[field] = reasons
            series = frame[field]
            profile.append(dict(column=field,sql_type=column["SqlType"],pandas_dtype=str(series.dtype),
                                missing_count=int(series.isna().sum()),
                                missing_fraction=float(series.isna().mean()) if len(frame) else None,
                                distinct_non_null=int(series.nunique(dropna=True))))
        inventory.append(dict(table=key,schema=schema_name,name=name,rows=len(frame),columns=len(columns),role=role,
                              primary_key=pk,foreign_key_columns=fk,patient_join_path=paths((schema_name,name)),
                              temporal_columns=[c["ColumnName"] for c in columns if c["SqlType"].lower() in TEMPORAL_TYPES],
                              numeric_columns_to_review=[c["ColumnName"] for c in columns if c["SqlType"].lower() in INTEGER_TYPES | {"float","real","decimal","numeric","bit","money","smallmoney"} and c["ColumnName"] not in flags and role not in {"derived_labels_and_audit","database_metadata","lookup_settings"}],
                              feature_review_flags=flags,column_profile=profile))
    return inventory


def extract_snapshot(connection, progress=None):
    """Extract every non-system base table under the caller's read transaction.

    SERIALIZABLE holds shared/range locks until the connection context closes.
    Finish extraction and close the context before training or writing files.
    """
    cursor = connection.cursor()
    try:
        cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE; SET LOCK_TIMEOUT 30000;")
    finally:
        cursor.close()
    info = _query(connection,"SELECT DB_NAME() AS DatabaseName,CONVERT(nvarchar(128),SERVERPROPERTY('ServerName')) AS ServerName")[0]
    if info["DatabaseName"]!="EpilepsyDB":
        raise ValueError("Expected EpilepsyDB, got "+str(info["DatabaseName"]))
    schema = dict(format_version=1,database=info["DatabaseName"],server=info["ServerName"],
                  extracted_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),isolation="SERIALIZABLE",
                  temporal_representation="SQL CONVERT(varchar(48),value,127); original SQL types and precision retained in columns metadata",
                  tables=_query(connection,TABLES_SQL),columns=_query(connection,COLUMNS_SQL),
                  primary_keys=_query(connection,KEYS_SQL),foreign_keys=_query(connection,FOREIGN_KEYS_SQL))
    tables = {}
    for table in schema["tables"]:
        schema_name,name = table["SchemaName"],table["TableName"]
        key = schema_name+"."+name
        if key in tables:
            raise ValueError("Ambiguous schema-qualified table name: "+key)
        columns = [c for c in schema["columns"] if (c["SchemaName"],c["TableName"])==(schema_name,name)]
        keys = [c["ColumnName"] for c in schema["primary_keys"] if (c["SchemaName"],c["TableName"])==(schema_name,name)]
        qualified = quote_identifier(schema_name)+"."+quote_identifier(name)
        order = " ORDER BY "+",".join(quote_identifier(c) for c in keys) if keys else ""
        records = _query(connection,"SELECT "+_projection(columns)+" FROM "+qualified+order)
        frame = _typed_frame(records,columns)
        expected = _query(connection,"SELECT COUNT_BIG(*) AS N FROM "+qualified)[0]["N"]
        if len(frame)!=expected:
            raise ValueError("Extracted row count does not match source: "+key)
        tables[key] = frame
        if progress:
            progress(key,len(frame))
    return DatabaseSnapshot(tables,schema,make_inventory(tables,schema))


def _encode(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value,Decimal):
        return {"__sql_decimal__":str(value)}
    if isinstance(value,(bytes,bytearray,memoryview)):
        return {"__sql_binary__":base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value,(dt.date,dt.datetime,dt.time)):
        return value.isoformat()
    raise TypeError("Unsupported snapshot value: "+type(value).__name__)


def _decode(value):
    if set(value)=={"__sql_decimal__"}:
        return Decimal(value["__sql_decimal__"])
    if set(value)=={"__sql_binary__"}:
        return base64.b64decode(value["__sql_binary__"])
    return value


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def write_snapshot(snapshot, directory):
    """Write new snapshot directories only; manifest is written last on success."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    entries = []
    for index,table in enumerate(snapshot.schema["tables"],1):
        schema_name,name = table["SchemaName"],table["TableName"]
        key = schema_name+"."+name
        filename = f"table_{index:03d}__{quote(schema_name,safe='')}__{quote(name,safe='')}.jsonl"
        frame = snapshot.tables[key]
        with (directory/filename).open("w",encoding="utf-8",newline="\n") as stream:
            for record in frame.to_dict(orient="records"):
                stream.write(json.dumps(record,ensure_ascii=False,default=_encode,allow_nan=False,separators=(",", ":"))+"\n")
        entries.append(dict(table=key,schema=schema_name,name=name,file=filename,rows=len(frame),
                            columns=list(frame.columns),sha256=_hash(directory/filename)))
    for filename,data in (("schema.json",snapshot.schema),("inventory.json",snapshot.inventory)):
        (directory/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=_encode),encoding="utf-8")
    manifest = dict(format_version=1,format="jsonl",database=snapshot.schema["database"],server=snapshot.schema["server"],
                    extracted_at_utc=snapshot.schema["extracted_at_utc"],table_count=len(entries),
                    total_rows=sum(t["rows"] for t in entries),tables=entries,
                    metadata_hashes={name:_hash(directory/name) for name in ("schema.json","inventory.json")})
    (directory/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


def read_snapshot(directory):
    """Reload all tables as typed DataFrames and verify every saved checksum."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version")!=1 or manifest.get("format")!="jsonl":
        raise ValueError("Unsupported snapshot format")
    def checked_file(name,expected_hash):
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or _hash(path)!=expected_hash:
            raise ValueError("Snapshot checksum/path verification failed: "+name)
        return path
    schema = json.loads(checked_file("schema.json",manifest["metadata_hashes"]["schema.json"]).read_text(encoding="utf-8"))
    inventory = json.loads(checked_file("inventory.json",manifest["metadata_hashes"]["inventory.json"]).read_text(encoding="utf-8"))
    tables = {}
    for entry in manifest["tables"]:
        path = checked_file(entry["file"],entry["sha256"])
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line,object_hook=_decode) for line in stream if line.strip()]
        columns = [c for c in schema["columns"] if (c["SchemaName"],c["TableName"])==(entry["schema"],entry["name"])]
        names = [c["ColumnName"] for c in columns]
        if len(records)!=entry["rows"] or names!=entry["columns"] or any(list(r)!=names for r in records):
            raise ValueError("Snapshot row/column mismatch: "+entry["table"])
        tables[entry["table"]] = _typed_frame(records,columns)
    expected = {t["SchemaName"]+"."+t["TableName"] for t in schema["tables"]}
    if set(tables)!=expected or len(tables)!=manifest["table_count"] or sum(len(t) for t in tables.values())!=manifest["total_rows"]:
        raise ValueError("Incomplete snapshot")
    return DatabaseSnapshot(tables,schema,inventory)
