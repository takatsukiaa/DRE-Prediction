"""Standalone DOCX parser producing MySQL-ready pandas DataFrames.

The module intentionally does not import any of the project's legacy parser
modules.  ``parse_docx_files`` is the public API; the command-line interface is
primarily a convenient way to inspect or export the resulting tables.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from docx import Document


COMMON_COLUMNS = ["patient_id", "cohort"]

PREGNANCY_COLUMNS = [
    "pregnancy",
    "pregnancy_abnormal_obstetric_check",
    "pregnancy_count",
    "childbirth_count",
    "child_inborn_error",
    "child_developmental_delay",
    "spontaneous_abortion",
    "artificial_abortion",
    "seizure_frequency_during_pregnancy",
    "increased_seizure_frequency_during_pregnancy",
    "decreased_seizure_frequency_during_pregnancy",
]

STATIC_VALUE_COLUMNS = [
    "birthday",
    "gender",
    "first_seizure_age",
    "drug_refractory_epilepsy",
    "systemic_disease",
    "adverse_for_asms",
    "adverse_for_asms_asm_name",
    "adverse_for_asms_description",
    *PREGNANCY_COLUMNS,
    "etiology",
    "eeg_focality",
    "frequency_per_month",
    "focality",
    "unknown_focal_or_generalized",
    "generalized",
    "unclassified",
    "elementary_motor",
    "complex_hypermotor",
    "sensory",
    "cognitive_and_language",
    "autonomic",
    "affective",
    "aura",
    "descriptor",
    "asymptomatic_ied",
    "provoking_factor",
    "diagnosis",
    "mental_retardation",
]

ASM_COLUMNS = COMMON_COLUMNS + [
    "date",
    "observed_date",
    "record_type",
    "drug_label",
    "generic_name",
    "unit_dose",
    "unit_dose_unit",
    "administered_dose",
    "administration_unit",
    "frequency",
    "frequency_per_day",
    "daily_dose_mg",
    "ddd",
    "total_ddd",
    "number_of_drugs",
    "raw_instruction",
    "is_carried_forward",
]

PATIENT_ASM_BASE_COLUMNS = [
    "patient_id",
    "cohort",
    "date",
    "is_carried_forward",
    "total_ddd",
    "number_of_drugs",
]

PATIENT_ASM_DRUG_FIELDS = [
    "drug_label",
    "observed_date",
    "unit_dose",
    "unit_dose_unit",
    "administered_dose",
    "administration_unit",
    "frequency",
    "frequency_per_day",
    "daily_dose_mg",
    "ddd",
    "raw_instruction",
]

PATIENT_ASM_MANIFEST_COLUMNS = [
    "patient_id",
    "cohort",
    "csv_file",
    "status",
    "row_count",
    "first_date",
    "last_date",
    "drug_count",
    "drug_names",
    "asm_error_count",
]

SEIZURE_COLUMNS = COMMON_COLUMNS + [
    "date",
    "seizure_count_min",
    "seizure_count_max",
    "seizure_count_estimate",
    "confidence",
    "evidence_count",
    "methods",
    "evidence",
]

EEG_COLUMNS = COMMON_COLUMNS + ["report_date", "report"]
LAB_COLUMNS = COMMON_COLUMNS + ["date", "test_name", "value"]
COGNITIVE_COLUMNS = COMMON_COLUMNS + ["date", "test_type", "metric", "value"]
BRAIN_IMAGE_COLUMNS = COMMON_COLUMNS + ["date", "field_name", "value"]
ERROR_COLUMNS = [
    "patient_id",
    "cohort",
    "source_file",
    "category",
    "error_type",
    "row_number",
    "evidence",
    "message",
]

DEFAULT_ASM_PER_PATIENT_DIR = Path("mysql_ready_csvs") / "asm_per_patient"

DATAFRAME_COLUMNS = {
    "static": COMMON_COLUMNS + STATIC_VALUE_COLUMNS,
    "asm": ASM_COLUMNS,
    "seizure": SEIZURE_COLUMNS,
    "eeg": EEG_COLUMNS,
    "lab_general": LAB_COLUMNS,
    "lab_routine": LAB_COLUMNS,
    "lab_drug": LAB_COLUMNS,
    "cognitive": COGNITIVE_COLUMNS,
    "brain_image": BRAIN_IMAGE_COLUMNS,
    "parse_errors": ERROR_COLUMNS,
}


def normalize_text(value: object) -> str:
    text = str(value or "").replace("\xa0", " ").replace("→", "->").replace("〜", "~")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_label(value: object) -> str:
    text = normalize_text(value).casefold()
    text = text.replace("★", "").replace("☆", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :：-_")


def get_cell_text(cell) -> str:
    """Read visible and underlying XML text, preferring the more complete form."""
    visible = normalize_text(cell.text.replace("\n", " "))
    nodes = cell._tc.xpath(".//w:t")
    xml_text = normalize_text("".join(node.text for node in nodes if node.text))
    return xml_text if len(xml_text) > len(visible) else visible


def get_unique_row_values(row) -> list[str]:
    values: list[str] = []
    seen = set()
    for cell in row.cells:
        if cell._tc in seen:
            continue
        seen.add(cell._tc)
        values.append(get_cell_text(cell))
    return values


def table_title(table) -> str:
    if not table.rows:
        return ""
    values = get_unique_row_values(table.rows[0])
    return normalize_label(values[0]) if values else ""


def parse_date(value: object) -> pd.Timestamp:
    text = normalize_text(value)
    if not text:
        return pd.NaT
    compact = re.search(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)", text)
    if compact:
        return pd.to_datetime("".join(compact.groups()), format="%Y%m%d", errors="coerce")
    separated = re.search(r"(?<!\d)((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)", text)
    if separated:
        return pd.to_datetime("-".join(separated.groups()), errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def patient_id_from_path(path: Path) -> str:
    match = re.search(r"\d+", path.stem)
    return match.group(0) if match else path.stem


def cohort_from_path(path: Path) -> str:
    parts = [part.upper() for part in path.parts]
    if any("NDRE" in part or part.startswith("NS") for part in parts):
        return "NDRE"
    if any("DRE" in part for part in parts):
        return "DRE"
    return path.parent.name


def collect_docx_files(paths: Sequence[str | Path] | str | Path) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*")
        else:
            continue
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.casefold() == ".docx" and not candidate.name.startswith("~$"):
                resolved = candidate.resolve()
                found[str(resolved).casefold()] = resolved
    return sorted(found.values(), key=lambda item: str(item).casefold())


def add_error(
    errors: list[dict],
    path: Path,
    patient_id: str,
    cohort: str,
    category: str,
    error_type: str,
    message: str,
    row_number: int | None = None,
    evidence: object = "",
) -> None:
    errors.append(
        {
            "patient_id": patient_id,
            "cohort": cohort,
            "source_file": str(path),
            "category": category,
            "error_type": error_type,
            "row_number": row_number,
            "evidence": normalize_text(evidence)[:500],
            "message": str(message),
        }
    )


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

STATIC_ALIASES = {
    "patient id": "patient_id",
    "birthday": "birthday",
    "date of birth": "birthday",
    "gender": "gender",
    "1st seizure age": "first_seizure_age",
    "1st seizure age (ymd)": "first_seizure_age",
    "drug-refractory epilepsy": "drug_refractory_epilepsy",
    "drug-refractory epilepsy (≥3asm & sz in recent 1 yr)": "drug_refractory_epilepsy",
    "systemic disease": "systemic_disease",
    "systemic disease 內科疾病": "systemic_disease",
    "etiology": "etiology",
    "eeg focality": "eeg_focality",
    "frequency (n/month)": "frequency_per_month",
    "unknown whether focal or generalized (u)": "unknown_focal_or_generalized",
    "generalized (g)": "generalized",
    "unclassified": "unclassified",
    "elementary motor (motor)": "elementary_motor",
    "complex hypermotor (automatism)": "complex_hypermotor",
    "sensory": "sensory",
    "cognitive and language": "cognitive_and_language",
    "autonomic (e.g. palpitation, sweating, hot flush, epigastric or abdominal raising)": "autonomic",
    "affective (emotional)": "affective",
    "aura (indescribable)": "aura",
    "descriptor": "descriptor",
    "asymptomatic ied (eeg ied)": "asymptomatic_ied",
    "provoking factor": "provoking_factor",
    "dx": "diagnosis",
    "diagnosis": "diagnosis",
    "mental retardation": "mental_retardation",
    "pregnancy": "pregnancy",
    "pregenacy": "pregnancy",
    "preganancy": "pregnancy",
}

ADVERSE_TITLES = {"adverse of asm", "adverse for asm", "adverse for asms"}
PREGNANCY_TITLES = {
    "pregnancy",
    "pregenacy",
    "preganancy",
    "pregnancy 懷孕狀況",
}
PREGNANCY_FIELD_ALIASES = {
    "pregnancy": "pregnancy",
    "pregenacy": "pregnancy",
    "preganancy": "pregnancy",
    "正在懷孕": "pregnancy",
    "abnormal at obs routine check 產檢異常": "pregnancy_abnormal_obstetric_check",
    "pregnancy number 先前+目前": "pregnancy_count",
    "child birth number先前+目前": "childbirth_count",
    "child birth number 先前+目前": "childbirth_count",
    "inborn error of child 先天性疾病": "child_inborn_error",
    "developmental delay of child": "child_developmental_delay",
    "spontaneous abortion": "spontaneous_abortion",
    "artificial abortion": "artificial_abortion",
    "seizure frequency during pregnancy (t1, t2, t3 n/m)": "seizure_frequency_during_pregnancy",
    "increased seizure frequency during pregnancy": "increased_seizure_frequency_during_pregnancy",
    "decreased seizure frequency during pregnancy": "decreased_seizure_frequency_during_pregnancy",
}
PREGNANCY_NOT_APPLICABLE = "50歲以上不適用"

EEG_ELECTRODES = [
    "Fp1", "F3", "F7", "Fp2", "F4", "F8", "Fz", "T3", "T4", "T5", "T6",
    "S1", "S2", "C3", "C4", "Cz", "P3", "P4", "Pz", "O1", "O2",
]
EEG_ELECTRODE_BY_CASE = {item.casefold(): item for item in EEG_ELECTRODES}
EEG_ELECTRODE_RE = re.compile(
    "|".join(re.escape(item) for item in sorted(EEG_ELECTRODES, key=len, reverse=True)),
    re.IGNORECASE,
)
FOCALITY_CODE_RE = re.compile(r"(?<![\d.])(?:[1-3]\.[1-3])(?![\d.])")


def extract_eeg_electrodes(text: object) -> list[str]:
    result: list[str] = []
    seen = set()
    for match in EEG_ELECTRODE_RE.findall(str(text or "")):
        canonical = EEG_ELECTRODE_BY_CASE[match.casefold()]
        if canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def extract_focality_codes(text: object) -> list[str]:
    return ordered_unique(FOCALITY_CODE_RE.findall(str(text or "")))


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        value = normalize_text(value).strip(" ,;|/")
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def split_focality_value(value: object) -> tuple[str, str]:
    text = normalize_text(value)
    if not text:
        return "", ""

    eeg_match = re.search(r"(?i)\beeg\s*focality\s*[:：-]?", text)
    focal_match = re.search(r"(?i)(?<!eeg\s)\bfocality\s*[:：-]?", text)
    if eeg_match and focal_match:
        markers = sorted([(eeg_match.start(), eeg_match.end(), "eeg"), (focal_match.start(), focal_match.end(), "focal")])
        pieces = {"eeg": "", "focal": ""}
        for index, (_start, end, kind) in enumerate(markers):
            next_start = markers[index + 1][0] if index + 1 < len(markers) else len(text)
            pieces[kind] = text[end:next_start].strip(" ,;|/")
        return pieces["eeg"], pieces["focal"]

    electrodes = extract_eeg_electrodes(text)
    if not electrodes:
        return "", text
    remainder = EEG_ELECTRODE_RE.sub(" ", text)
    remainder = re.sub(r"(?i)\beeg\s*focality\b|\bfocality\b", " ", remainder)
    remainder = re.sub(r"[\s,;|/]+", " ", remainder).strip(" :-")
    return ", ".join(electrodes), remainder


def parse_boolean(value: object) -> bool | pd._libs.missing.NAType:
    text = normalize_label(value)
    if text in {"yes", "y", "true"}:
        return True
    if text in {"no", "n", "false"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return pd.NA
    if numeric == 0:
        return False
    if numeric > 0:
        return True
    return pd.NA


def parse_adverse_table(
    table,
) -> tuple[bool | pd._libs.missing.NAType, str, str, bool]:
    flag: bool | pd._libs.missing.NAType = pd.NA
    asm_name = ""
    description = ""
    flag_labels = ADVERSE_TITLES | {"yes", "has adverse response", "adverse present"}
    asm_name_labels = {"asm name"}
    description_labels = {
        "description",
        "adverse response",
        "adverse effect",
        "adverse effects",
        "side effect",
        "side effects",
    }
    empty_values = {"", "0", "-", "none", "nil", "na", "n/a"}

    for source_row in table.rows:
        values = get_unique_row_values(source_row)
        if not values:
            continue
        label = normalize_label(values[0])
        value = values[1] if len(values) > 1 else ""

        if label in flag_labels:
            candidate = parse_boolean(value)
            if not pd.isna(candidate):
                flag = candidate
        elif label in asm_name_labels and normalize_label(value) not in empty_values:
            asm_name = normalize_text(value)
        elif label in description_labels and normalize_label(value) not in empty_values:
            description = normalize_text(value)

    # In the source template, a false checkbox means there is no adverse ASM
    # response.  Other populated cells can contain stale medication text and
    # must not leak into the clinical description.
    if flag is False:
        asm_name = ""
        description = ""
    else:
        if normalize_label(asm_name) in empty_values:
            asm_name = ""
        if normalize_label(description) in empty_values:
            description = ""
    ambiguous = bool(asm_name or description) and pd.isna(flag)
    return flag, asm_name, description, ambiguous


def normalize_gender(value: object) -> str | pd._libs.missing.NAType:
    normalized = normalize_label(value)
    if normalized in {"0", "female", "f", "woman", "女", "女生"}:
        return "female"
    if normalized in {"1", "male", "m", "man", "男", "男生"}:
        return "male"
    return pd.NA


def clean_first_seizure_age(value: object) -> str | pd._libs.missing.NAType:
    text = normalize_text(value)
    normalized = normalize_label(text)
    if not text or normalized.startswith("未知"):
        return pd.NA

    descriptors = {
        "early childhood": "early childhood",
        "國中": "國中",
    }
    if normalized in descriptors:
        return descriptors[normalized]

    combined = re.match(
        r"(?i)^\s*(\d+(?:\.\d+)?\s*y\s*\d+(?:\.\d+)?\s*m)\b",
        text,
    )
    match = combined or re.match(
        r"(?i)^\s*("
        r"\d+(?:\.\d+)?"
        r"(?:\s*(?:~|-)\s*\d+(?:\.\d+)?)?"
        r"\s*\+?"
        r"(?:\s*(?:y/o|years?|yrs?|y|m/o|months?|mos?|m))?"
        r")",
        text,
    )
    if not match:
        return pd.NA

    cleaned = normalize_text(match.group(1))
    cleaned = re.sub(r"\s*([~-])\s*", r"\1", cleaned)
    cleaned = re.sub(r"\s*\+\s*", "+", cleaned)
    numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", cleaned)]
    has_unit_or_range = bool(re.search(r"(?i)[a-z/~+-]", cleaned))
    if not numbers or (not has_unit_or_range and numbers[0] > 120):
        return pd.NA
    if any(number > 120 for number in numbers):
        return pd.NA
    return cleaned


def clean_etiology_codes(value: object) -> str | pd._libs.missing.NAType:
    text = normalize_text(value)
    if not text:
        return pd.NA
    code = r"(?:1[0-4]|[0-9])(?!\d)"
    match = re.match(rf"^\s*({code}(?:\s*[,、]\s*{code})*)", text)
    if not match:
        return pd.NA
    codes = re.findall(code, match.group(1))
    return ", ".join(ordered_unique(codes)) if codes else pd.NA


def parse_pregnancy_table(table) -> tuple[dict[str, str], bool]:
    """Return pregnancy key/value fields and whether the whole table is ineligible."""
    parsed: dict[str, str] = {}
    value_cells: list[str] = []
    for source_row in table.rows:
        values = get_unique_row_values(source_row)
        if len(values) < 2:
            continue
        value_cells.extend(values[1:])
        field = PREGNANCY_FIELD_ALIASES.get(normalize_label(values[0]))
        if field:
            parsed[field] = normalize_text(values[1])

    ignore_table = any(
        normalize_text(value) == PREGNANCY_NOT_APPLICABLE for value in value_cells
    )
    return ({} if ignore_table else parsed), ignore_table


def static_value_pairs(table) -> Iterable[tuple[str, str]]:
    for row in table.rows:
        values = get_unique_row_values(row)
        for index, label in enumerate(values[:-1]):
            normalized = normalize_label(label)
            if normalized in STATIC_ALIASES:
                yield STATIC_ALIASES[normalized], values[index + 1]


def parse_static(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> dict:
    row = {column: "" for column in COMMON_COLUMNS + STATIC_VALUE_COLUMNS}
    row.update({"patient_id": patient_id, "cohort": cohort, "adverse_for_asms": pd.NA})
    found_static = False
    eeg_values: list[str] = []
    focal_codes: list[str] = []

    for table in doc.tables:
        title = table_title(table)
        is_static = (
            title.startswith("baseline")
            or title in {"semiology", "eeg focality", "etiology of epilepsy"}
            or title in ADVERSE_TITLES
            or title in PREGNANCY_TITLES
        )
        if not is_static:
            continue
        found_static = True

        if title in ADVERSE_TITLES:
            flag, asm_name, description, ambiguous = parse_adverse_table(table)
            row["adverse_for_asms"] = flag
            row["adverse_for_asms_asm_name"] = asm_name
            row["adverse_for_asms_description"] = description
            if ambiguous:
                add_error(
                    errors, path, patient_id, cohort, "static", "ambiguous_adverse_flag",
                    "Adverse ASM name or response was present without an explicit Yes/No flag.",
                    evidence="; ".join(value for value in [asm_name, description] if value),
                )
            continue

        if title == "eeg focality":
            for source_row in table.rows[1:]:
                values = get_unique_row_values(source_row)
                if values:
                    eeg_values.extend(extract_eeg_electrodes(values[-1]))

        if title == "semiology":
            for row_number, source_row in enumerate(table.rows, start=1):
                values = get_unique_row_values(source_row)
                if not values or not normalize_label(values[0]).startswith("focal (f)"):
                    continue
                focal_value = values[1] if len(values) > 1 else ""
                codes = extract_focality_codes(focal_value)
                if codes:
                    focal_codes.extend(codes)
                elif normalize_text(focal_value):
                    add_error(
                        errors,
                        path,
                        patient_id,
                        cohort,
                        "static",
                        "unparseable_focality",
                        "The populated Semiology Focal row contained no recognized classification code.",
                        row_number=row_number,
                        evidence=focal_value,
                    )

        if title == "etiology of epilepsy" and table.rows:
            values = get_unique_row_values(table.rows[0])
            if len(values) > 1:
                row["etiology"] = values[1]
            if len(values) > 3:
                row["mental_retardation"] = values[3]

        if title in PREGNANCY_TITLES:
            pregnancy_values, _ignored = parse_pregnancy_table(table)
            row.update(pregnancy_values)
            continue

        for field, value in static_value_pairs(table):
            if field == "patient_id":
                match = re.search(r"\d+", value)
                if match:
                    row["patient_id"] = match.group(0)
            elif field == "eeg_focality":
                eeg_part, _focal_part = split_focality_value(value)
                if not eeg_part:
                    eeg_part = value
                if eeg_part:
                    eeg_values.append(eeg_part)
            else:
                row[field] = value

    if not found_static:
        add_error(errors, path, patient_id, cohort, "static", "missing_table", "No static-data table was found.")

    row["patient_id"] = row["patient_id"] or patient_id
    row["eeg_focality"] = ", ".join(ordered_unique(
        electrode for value in eeg_values for electrode in (extract_eeg_electrodes(value) or [value])
    ))
    normalized_focality = ", ".join(ordered_unique(focal_codes))
    row["focality"] = normalized_focality or pd.NA

    raw_gender = row["gender"]
    row["gender"] = normalize_gender(raw_gender)
    if normalize_text(raw_gender) and pd.isna(row["gender"]):
        add_error(
            errors,
            path,
            row["patient_id"],
            cohort,
            "static",
            "unrecognized_gender",
            "Gender value was not recognized as male or female.",
            evidence=raw_gender,
        )

    raw_age = row["first_seizure_age"]
    row["first_seizure_age"] = clean_first_seizure_age(raw_age)
    if (
        normalize_text(raw_age)
        and not normalize_label(raw_age).startswith("未知")
        and pd.isna(row["first_seizure_age"])
    ):
        add_error(
            errors,
            path,
            row["patient_id"],
            cohort,
            "static",
            "unparseable_first_seizure_age",
            "First-seizure age contained no recognized age expression.",
            evidence=raw_age,
        )

    raw_etiology = row["etiology"]
    row["etiology"] = clean_etiology_codes(raw_etiology)
    if normalize_text(raw_etiology) and pd.isna(row["etiology"]):
        add_error(
            errors,
            path,
            row["patient_id"],
            cohort,
            "static",
            "unparseable_etiology",
            "Etiology contained no leading code from 0 through 14.",
            evidence=raw_etiology,
        )

    if not pd.isna(row["gender"]) and row["gender"] == "male":
        for field in PREGNANCY_COLUMNS:
            row[field] = pd.NA
    else:
        for field in PREGNANCY_COLUMNS:
            if row[field] == "":
                row[field] = pd.NA
    return row


# ---------------------------------------------------------------------------
# ASM medication details and monthly summaries
# ---------------------------------------------------------------------------

DRUG_FREQUENCY_MAP = [
    ("BIDPC (7 DAYS)", 2.0), ("BIDPC", 2.0), ("BID", 2.0),
    ("TIDPC", 3.0), ("TID", 3.0), ("QID", 4.0), ("QDPC", 1.0), ("QNPC", 1.0),
    ("QDLUNCH", 1.0), ("QO HS", 0.5), ("QOHS", 0.5), ("QOD", 0.5),
    ("HS", 1.0), ("QD", 1.0), ("QN", 1.0), ("AS ORDER", 1.0),
    ("Q12H", 2.0), ("Q8H", 3.0), ("Q6H", 4.0),
]

DDD_MG = {
    "Lamotrigine": 300, "Pregabalin": 300, "Topiramate": 300, "Perampanel": 8,
    "Levetiracetam": 1500, "Oxcarbazepine": 1000, "Carbamazepine": 1000,
    "Valproate": 1500, "Clobazam": 20, "Lacosamide": 300, "Zonisamide": 200,
    "Clonazepam": 8, "Phenytoin": 300, "Phenobarbital": 100, "Risperidone": 5,
    "Gabapentin": 1800, "Diazepam": 10, "Brivaracetam": 100, "Vigabatrin": 2000,
}

DRUG_ALIASES = {
    "toporamate": "Topiramate", "vimpet": "Lacosamide", "lmctal": "Lamotrigine",
    "lamictal": "Lamotrigine", "lyrica": "Pregabalin", "topamax": "Topiramate",
    "fycompa": "Perampanel", "keppra": "Levetiracetam", "trileptal": "Oxcarbazepine",
    "depakine": "Valproate", "frisium": "Clobazam", "vimpat": "Lacosamide",
    "zonegran": "Zonisamide", "carpine": "Carbamazepine", "aclonax": "Clonazepam",
    "rivotril": "Clonazepam", "tegretol": "Carbamazepine",
    "dilantin": "Phenytoin", "risperdal": "Risperidone", "neurontin": "Gabapentin",
    "ufree": "Levetiracetam", "dupin": "Diazepam", "briviact": "Brivaracetam",
}

UNIT_DOSE_RE = re.compile(
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<mass>mg|g)\s*/\s*(?P<form>tab(?:let)?|cap(?:sule)?|ml)",
    re.IGNORECASE,
)

UNIT_DOSE_FALLBACKS = {
    "valproate (depakine)": (500.0, "mg/tab"),
    "carbamazepine c.r": (200.0, "mg/tab"),
    "toporamate (topamax) 25cap/tab": (25.0, "mg/tab"),
    "gabapentin (neurontin)": (300.0, "mg/cap"),
}

INSTRUCTION_RE = re.compile(
    r"(?P<dose>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>tabs?|tablets?|caps?|capsules?|tan|ml|mg|g|#)?\s*"
    r"(?:/\s*|\s+)"
    r"(?P<frequency>[A-Za-z][A-Za-z0-9 ]*(?:\([^)]*\))?)",
    re.IGNORECASE,
)

COMPOUND_INSTRUCTION_BOUNDARY_RE = re.compile(
    r"(?i)(?<=\S)\s+(?="
    r"\d+(?:\.\d+)?\s*(?:tabs?|tablets?|caps?|capsules?|tan|ml|mg|g|#)\s*"
    r"(?:/|\s+(?:BID|TID|QID|QD|QN|HS|Q\d+H|QO\s*HS|QOD|QW\s*[1-7]+|AS\s+OD?ER))"
    r")"
)

ASM_DATE_CORRECTIONS = {("0199", "21090221"): "20190221"}
MIN_ASM_YEAR = 1900


def lookup_frequency(value: str) -> float | None:
    upper = re.sub(r"(?i)\bAS\s+ODER\b", "AS ORDER", normalize_text(value)).upper()
    if "PRN" in upper:
        return None
    weekly = re.search(r"\bQW\s*([1-7]+)\b", upper)
    if weekly:
        return len(set(weekly.group(1))) / 7.0
    for keyword, multiplier in DRUG_FREQUENCY_MAP:
        if keyword in upper:
            return multiplier
    return None


def canonical_drug_name(label: str) -> str | None:
    candidates: list[str] = []
    candidates.extend(re.findall(r"\(([^)]+)\)", label))
    candidates.extend(re.findall(r"[A-Za-z][A-Za-z-]+", re.sub(r"\([^)]*\)", " ", label)))
    known = list(DDD_MG)
    for raw in reversed(candidates):
        for part in re.split(r"\s*/\s*", raw):
            candidate = normalize_text(part).strip(" .")
            if not candidate:
                continue
            for name in known:
                if candidate.casefold() == name.casefold():
                    return name
            if candidate.casefold() in DRUG_ALIASES:
                return DRUG_ALIASES[candidate.casefold()]
            match = get_close_matches(candidate, known, n=1, cutoff=0.78)
            if match:
                return match[0]
    return None


def unit_dose_from_label(label: str) -> tuple[float | None, str | None]:
    match = UNIT_DOSE_RE.search(label)
    if match:
        value = float(match.group("dose"))
        if match.group("mass").casefold() == "g":
            value *= 1000
        form = match.group("form").casefold()
        form = "mL" if form == "ml" else ("tab" if form.startswith("tab") else "cap")
        return value, f"mg/{form}"

    mass_only = re.search(r"(?i)(?P<dose>\d+(?:\.\d+)?)\s*(?P<mass>mg|g)\b", label)
    if mass_only:
        value = float(mass_only.group("dose"))
        if mass_only.group("mass").casefold() == "g":
            value *= 1000
        return value, "mg"
    return UNIT_DOSE_FALLBACKS.get(normalize_label(label), (None, None))


def split_instructions(value: str) -> list[str]:
    text = normalize_text(value).replace("\n", ";")
    text = COMPOUND_INSTRUCTION_BOUNDARY_RE.sub(";", text)
    pieces = [part.strip() for part in re.split(r"[;,]+", text) if part.strip()]
    return pieces or ([text] if text else [])


def parse_instruction(value: str) -> tuple[float | None, str | None, str, float | None]:
    match = INSTRUCTION_RE.search(value)
    if match:
        dose = float(match.group("dose"))
        unit = (match.group("unit") or "").casefold()
        if unit.startswith("tab") or unit == "tan":
            unit = "tab"
        elif unit.startswith("cap"):
            unit = "cap"
        elif unit == "ml":
            unit = "mL"
        unit = unit or None
        frequency = re.sub(
            r"(?i)\bAS\s+ODER\b", "as order", normalize_text(match.group("frequency"))
        )
        return dose, unit, frequency, lookup_frequency(frequency)

    dose_match = re.search(r"\d+(?:\.\d+)?", value)
    frequency = normalize_text(value.split("/", 1)[1]) if "/" in value else ""
    frequency = re.sub(r"(?i)\bAS\s+ODER\b", "as order", frequency)
    dose = float(dose_match.group(0)) if dose_match else None
    unit_match = re.search(r"(?i)\b(tab(?:let)?s?|tan|cap(?:sule)?s?|ml|mg|g)\b|#", value)
    unit = unit_match.group(0) if unit_match else None
    if unit:
        lowered = unit.casefold()
        if lowered.startswith("tab") or lowered == "tan":
            unit = "tab"
        elif lowered.startswith("cap"):
            unit = "cap"
        elif lowered == "ml":
            unit = "mL"
        else:
            unit = lowered
    return dose, unit, frequency, lookup_frequency(frequency) if frequency else None


def calculate_daily_dose_mg(
    unit_dose: float | None,
    unit_dose_unit: str | None,
    administered_dose: float | None,
    administration_unit: str | None,
    frequency_per_day: float | None,
) -> float | None:
    if administered_dose is None or frequency_per_day is None:
        return None
    administration = normalize_text(administration_unit).casefold()
    if administration == "mg":
        return administered_dose * frequency_per_day
    if administration == "g":
        return administered_dose * 1000.0 * frequency_per_day
    if unit_dose is None:
        return None
    if administration == "ml" and normalize_text(unit_dose_unit).casefold() != "mg/ml":
        return None
    return unit_dose * administered_dose * frequency_per_day


def parse_asm_date(value: object, patient_id: str) -> tuple[pd.Timestamp, str | None]:
    text = normalize_text(value)
    corrected = ASM_DATE_CORRECTIONS.get((patient_id, text))
    parsed = parse_date(corrected or text)
    if pd.isna(parsed):
        return pd.NaT, corrected
    maximum_year = pd.Timestamp.today().year + 1
    if not MIN_ASM_YEAR <= parsed.year <= maximum_year:
        return pd.NaT, corrected
    return parsed, corrected


def month_end(value: pd.Timestamp) -> pd.Timestamp:
    return value.to_period("M").to_timestamp("M")


def parse_asm(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> list[dict]:
    target = next((table for table in doc.tables if table_title(table) == "asm狀況"), None)
    if target is None:
        add_error(errors, path, patient_id, cohort, "asm", "missing_table", "No ASM狀況 table was found.")
        return []

    header_index = None
    headers: list[str] = []
    for index, source_row in enumerate(target.rows):
        values = get_unique_row_values(source_row)
        if values and normalize_label(values[0]) == "date":
            header_index = index
            headers = values
            break
    if header_index is None:
        add_error(errors, path, patient_id, cohort, "asm", "missing_header", "ASM table has no Date header.")
        return []

    observed: dict[tuple[str, pd.Period], tuple[pd.Timestamp, list[dict]]] = {}
    all_dates: list[pd.Timestamp] = []
    dated_states: dict[pd.Timestamp, dict[str, bool]] = {}
    ignored_headers = {"date", "備註", "是否為吳醫師看診(不是：0；是：1；無看診紀錄：-)"}

    for row_number, source_row in enumerate(target.rows[header_index + 1 :], start=header_index + 2):
        values = get_unique_row_values(source_row)
        if not values or not any(values):
            continue
        values += [""] * max(0, len(headers) - len(values))
        observed_date, corrected_date = parse_asm_date(values[0], patient_id)
        if corrected_date is not None:
            add_error(
                errors, path, patient_id, cohort, "asm", "corrected_source_date",
                f"Corrected ASM date {values[0]} to {corrected_date}.",
                row_number, f"{values[0]} -> {corrected_date}",
            )
        if pd.isna(observed_date):
            if values[0]:
                add_error(
                    errors, path, patient_id, cohort, "asm", "unparseable_date",
                    "ASM row has an unparseable date.", row_number, values[0],
                )
            continue
        observed_date = observed_date.normalize()
        all_dates.append(observed_date)
        period = observed_date.to_period("M")
        row_state = dated_states.setdefault(observed_date, {})

        for drug_label, cell_value in zip(headers[1:], values[1:]):
            if normalize_text(drug_label) in ignored_headers:
                continue
            generic_name = canonical_drug_name(drug_label)
            if generic_name is not None:
                row_state[generic_name] = row_state.get(generic_name, False) or bool(
                    normalize_text(cell_value)
                )
            if not normalize_text(cell_value):
                continue
            if generic_name is None:
                add_error(
                    errors, path, patient_id, cohort, "asm", "unknown_drug",
                    "Could not identify the generic drug name.", row_number, drug_label,
                )
                continue
            unit_dose, unit_dose_unit = unit_dose_from_label(drug_label)
            if unit_dose is None:
                add_error(
                    errors, path, patient_id, cohort, "asm", "missing_unit_dose",
                    "Could not parse unit dose from the drug label.", row_number, drug_label,
                )

            regimen: list[dict] = []
            for instruction in split_instructions(cell_value):
                administered, administration_unit, frequency, frequency_per_day = parse_instruction(instruction)
                daily_dose = calculate_daily_dose_mg(
                    unit_dose,
                    unit_dose_unit,
                    administered,
                    administration_unit,
                    frequency_per_day,
                )
                ddd = daily_dose / DDD_MG[generic_name] if daily_dose is not None else None
                regimen.append(
                    {
                        "drug_label": drug_label,
                        "generic_name": generic_name,
                        "unit_dose": unit_dose,
                        "unit_dose_unit": unit_dose_unit,
                        "administered_dose": administered,
                        "administration_unit": administration_unit,
                        "frequency": frequency,
                        "frequency_per_day": frequency_per_day,
                        "daily_dose_mg": daily_dose,
                        "ddd": ddd,
                        "raw_instruction": instruction,
                    }
                )
                if administered is None or not frequency:
                    add_error(
                        errors, path, patient_id, cohort, "asm", "unparsed_instruction",
                        "Dose or frequency could not be completely parsed.", row_number, instruction,
                    )
            if regimen:
                key = (generic_name, period)
                previous = observed.get(key)
                if previous is None or observed_date >= previous[0]:
                    observed[key] = (observed_date, regimen)

    if not all_dates or not observed:
        return []

    stop_events: dict[tuple[str, pd.Period], pd.Timestamp] = {}
    for state_date in sorted(dated_states):
        state = dated_states[state_date]
        for generic_name, has_instruction in state.items():
            if has_instruction:
                continue
            key = (generic_name, state_date.to_period("M"))
            previous = stop_events.get(key)
            if previous is None or state_date > previous:
                stop_events[key] = state_date

    first_period = min(all_dates).to_period("M")
    last_period = max(all_dates).to_period("M")
    months = pd.period_range(first_period, last_period, freq="M")
    drugs = sorted({key[0] for key in observed})
    result: list[dict] = []

    for generic_name in drugs:
        latest: tuple[pd.Timestamp, list[dict]] | None = None
        for period in months:
            current = observed.get((generic_name, period))
            stop_date = stop_events.get((generic_name, period))
            if stop_date is not None and (current is None or stop_date >= current[0]):
                latest = None
                current = None
            elif current is not None:
                latest = current
            if latest is None:
                continue
            observed_date, regimen = latest
            is_carried_forward = current is None
            for item in regimen:
                result.append(
                    {
                        "patient_id": patient_id,
                        "cohort": cohort,
                        "date": period.to_timestamp("M"),
                        "observed_date": observed_date,
                        "record_type": "prescription",
                        **item,
                        "total_ddd": np.nan,
                        "number_of_drugs": pd.NA,
                        "is_carried_forward": is_carried_forward,
                    }
                )

    by_month: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for item in result:
        by_month[item["date"]].append(item)
    visit_periods = {date.to_period("M") for date in all_dates}
    for period in months:
        date = period.to_timestamp("M")
        prescriptions = by_month.get(date, [])
        total = sum(item["ddd"] for item in prescriptions if item["ddd"] is not None)
        count = len({item["generic_name"] for item in prescriptions if item["ddd"] is not None})
        summary = {column: pd.NA for column in ASM_COLUMNS}
        summary.update(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "date": date,
                "record_type": "monthly_summary",
                "total_ddd": round(total, 2),
                "number_of_drugs": count,
                "is_carried_forward": (
                    False
                    if period in visit_periods
                    else all(bool(item["is_carried_forward"]) for item in prescriptions)
                ),
            }
        )
        result.append(summary)
    return sorted(result, key=lambda item: (item["date"], item["record_type"], str(item.get("generic_name", ""))))


# ---------------------------------------------------------------------------
# Lab, cognitive, EEG, and brain-image tables
# ---------------------------------------------------------------------------

GENERAL_REPORT = "一般生化檢驗報告"
ROUTINE_REPORT = "常規生化檢驗報告"
ANTIGEN_REPORT = "抗原檢驗報告"
IMMUNE_REPORT = "免疫生化檢驗報告"
DRUG_REPORT = "藥物濃度（特殊生化檢驗報告）"
LAB_OUTPUT_MAP = {
    GENERAL_REPORT: "lab_general",
    ROUTINE_REPORT: "lab_routine",
    DRUG_REPORT: "lab_drug",
}
LAB_SECTION_TITLES = set(LAB_OUTPUT_MAP) | {ANTIGEN_REPORT, IMMUNE_REPORT}
LAB_THRESHOLD_PATTERN = re.compile(
    r"^\s*(?:>=|<=|>|<|\u2265|\u2267|\u2264|\u2266)\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\s*\([^)]*\))?\s*$"
)


def clean_numeric_value(value: object) -> float:
    text = normalize_text(value)
    if not text or text == "-":
        return np.nan
    threshold_match = LAB_THRESHOLD_PATTERN.fullmatch(text)
    if threshold_match:
        return float(threshold_match.group(1))
    text = re.sub(r"^[<>]=?", "", text)
    text = re.sub(r"(\d+)\.\.+(\d+)", r"\1.\2", text)
    return pd.to_numeric(text, errors="coerce")


def parse_labs(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> dict[str, list[dict]]:
    outputs = {key: [] for key in LAB_OUTPUT_MAP.values()}
    target = next((table for table in doc.tables if table_title(table) == "lab"), None)
    if target is None:
        add_error(errors, path, patient_id, cohort, "lab", "missing_table", "No Lab table was found.")
        return outputs

    sections: dict[str, tuple[list[str], list[list[str]]]] = {}
    current_section: str | None = None
    headers: list[str] | None = None
    rows: list[list[str]] = []

    def flush() -> None:
        nonlocal headers, rows
        if current_section and headers:
            sections[current_section] = (headers, rows)
        headers = None
        rows = []

    for row_number, source_row in enumerate(target.rows, start=1):
        values = get_unique_row_values(source_row)
        if not values or not values[0]:
            continue
        first = normalize_text(values[0])
        if first in LAB_SECTION_TITLES:
            flush()
            current_section = first
            continue
        if current_section and normalize_label(first) == "date":
            headers = values
            rows = []
            continue
        if current_section and headers:
            rows.append(values)
    flush()

    for section, output_key in LAB_OUTPUT_MAP.items():
        if section not in sections:
            continue
        section_headers, section_rows = sections[section]
        normalized_rows = [row + [""] * max(0, len(section_headers) - len(row)) for row in section_rows]
        frame = pd.DataFrame([row[: len(section_headers)] for row in normalized_rows], columns=section_headers)
        if "Date" not in frame.columns:
            add_error(errors, path, patient_id, cohort, output_key, "missing_header", "Lab section has no Date column.")
            continue
        frame["Date"] = frame["Date"].map(parse_date)
        invalid = frame["Date"].isna()
        for index in frame.index[invalid]:
            add_error(
                errors, path, patient_id, cohort, output_key, "unparseable_date",
                "Lab row has an unparseable date.", int(index) + 1, section_rows[int(index)][0],
            )
        frame = frame.loc[~invalid].set_index("Date")
        if frame.empty:
            continue
        for column in frame.columns:
            frame[column] = frame[column].map(clean_numeric_value)
        frame = frame.resample("ME").mean().dropna(how="all")
        for date, source_values in frame.iterrows():
            for test_name, value in source_values.items():
                if pd.notna(value):
                    outputs[output_key].append(
                        {
                            "patient_id": patient_id,
                            "cohort": cohort,
                            "date": date,
                            "test_name": str(test_name),
                            "value": float(value),
                        }
                    )
    return outputs


TEST_COLUMNS = {
    "MMSE": ["MMSE"] + [f"MMSE_Q{i}" for i in range(1, 12)],
    "CASI": ["CASI"] + [f"CASI_Q{i}" for i in range(1, 10)],
    "WAIS": ["WAIS"] + [f"WAIS_Q{i}" for i in range(1, 15)],
}


def identify_test_type(values: Sequence[str]) -> str | None:
    joined = " ".join(values).upper()
    return next((test for test in TEST_COLUMNS if test in joined), None)


def cognitive_metrics_from_header(test_type: str, values: Sequence[str]) -> list[str]:
    """Map DOCX header labels such as Q1 or Q 1 to stable metric names."""
    metrics: list[str] = []
    fallback = TEST_COLUMNS[test_type]
    for index, header in enumerate(values[1:]):
        if index == 0:
            metrics.append(test_type)
            continue
        match = re.search(r"(?i)\bq\s*(\d+)\b", normalize_text(header))
        if match:
            question = int(match.group(1))
            metric = f"{test_type}_Q{question}"
            if metric in fallback:
                metrics.append(metric)
                continue
        metrics.append(fallback[index] if index < len(fallback) else f"{test_type}_Q{index}")
    return metrics


def parse_cognitive(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> list[dict]:
    target = next((table for table in doc.tables if table_title(table) == "mental retardation"), None)
    if target is None:
        add_error(errors, path, patient_id, cohort, "cognitive", "missing_table", "No Mental Retardation table was found.")
        return []

    result: list[dict] = []
    current_test: str | None = None
    current_metrics: list[str] = []
    for row_number, source_row in enumerate(target.rows, start=1):
        values = get_unique_row_values(source_row)
        if not values:
            continue
        first = normalize_text(values[0])
        if normalize_label(first) == "mental retardation":
            continue
        if normalize_label(first) == "date":
            current_test = identify_test_type(values)
            current_metrics = cognitive_metrics_from_header(current_test, values) if current_test else []
            continue
        if current_test is None:
            continue
        date = parse_date(first)
        if pd.isna(date):
            if first:
                add_error(
                    errors, path, patient_id, cohort, "cognitive", "unparseable_date",
                    "Cognitive-test row has an unparseable date.", row_number, first,
                )
            continue
        raw_values = list(values[1:])
        raw_values += [""] * max(0, len(current_metrics) - len(raw_values))
        normalized_values = [normalize_text(value) for value in raw_values[: len(current_metrics)]]
        present_values = [value not in {"", "無", "NA", "N/A", "nan", "None"} for value in normalized_values]
        if not any(present_values):
            continue

        for metric, value, present in zip(current_metrics, normalized_values, present_values):
            # MMSE question cells are part of the database contract even when
            # the current source documents leave them blank.  Future populated
            # scores are captured into the same metrics without a schema change.
            if present or current_test == "MMSE":
                result.append(
                    {
                        "patient_id": patient_id,
                        "cohort": cohort,
                        "date": date.normalize(),
                        "test_type": current_test,
                        "metric": metric,
                        "value": value if present else pd.NA,
                    }
                )
    return result


def parse_eeg(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> list[dict]:
    target = next((table for table in doc.tables if table_title(table) == "eeg"), None)
    if target is None:
        add_error(errors, path, patient_id, cohort, "eeg", "missing_table", "No EEG table was found.")
        return []
    result: list[dict] = []
    for row_number, source_row in enumerate(target.rows[1:], start=2):
        values = get_unique_row_values(source_row)
        if not values or not any(values):
            continue
        if normalize_label(values[0]) == "date":
            continue
        report_date = parse_date(values[0])
        report = normalize_text(" ".join(value for value in values[1:] if value))
        if pd.isna(report_date):
            add_error(
                errors, path, patient_id, cohort, "eeg", "unparseable_date",
                "EEG row has an unparseable date.", row_number, " | ".join(values),
            )
            continue
        if report:
            result.append(
                {
                    "patient_id": patient_id,
                    "cohort": cohort,
                    "report_date": report_date.normalize(),
                    "report": report,
                }
            )
    return result


BRAIN_IMAGE_CATEGORY_ALIASES = {
    "mri": "MRI",
    "mri (核磁共振)": "MRI",
    "mri（核磁共振）": "MRI",
    "ct": "CT",
    "ct (電腦斷層)": "CT",
    "ct（電腦斷層）": "CT",
    "spect": "SPECT",
    "spect (核醫科影像)": "SPECT",
    "spect（核醫科影像）": "SPECT",
}


def parse_brain_image(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> list[dict]:
    target = next((table for table in doc.tables if table_title(table) == "brain image"), None)
    if target is None:
        add_error(errors, path, patient_id, cohort, "brain_image", "missing_table", "No Brain Image table was found.")
        return []

    result: list[dict] = []
    current_category: str | None = None
    for row_number, source_row in enumerate(target.rows[1:], start=2):
        values = get_unique_row_values(source_row)
        if not values or not any(values):
            continue

        first = normalize_text(values[0])
        label = normalize_label(first)
        category = BRAIN_IMAGE_CATEGORY_ALIASES.get(label)
        if category is not None:
            current_category = category
            continue
        if label == "date":
            continue

        report = normalize_text(" ".join(value for value in values[1:] if value))
        if not report:
            continue
        if current_category is None:
            add_error(
                errors, path, patient_id, cohort, "brain_image", "missing_category",
                "Brain-image report appeared before a recognized MRI, CT, or SPECT section.",
                row_number, " | ".join(values),
            )
            continue

        report_date = parse_date(first)
        if pd.isna(report_date):
            add_error(
                errors, path, patient_id, cohort, "brain_image", "unparseable_date",
                "Brain-image report has an unparseable date.",
                row_number, " | ".join(values),
            )
            continue
        result.append(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "date": report_date.normalize(),
                "field_name": current_category,
                "value": report,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Seizure timeline (consolidated from the legacy SOAP parser)
# ---------------------------------------------------------------------------

MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

SZ_KEYWORDS = re.compile(
    r"(?i)\b(sz|seizure|seizures|convulsive|convulsion|convulsions|gtc|cps|sps|aura)\b"
    r"|vocalization|發作|抽搐|抽蓄|臉部扭曲|微放電|放電|失神|恍神"
)
SZ_NO_EVENT_RE = re.compile(
    r"(?i)(?<![A-Za-z])(no seizure|no seizures|no clinical seizure|no event|no further seizure|seizure free|seizure-free)\b"
    r"|無發作|沒有發作|未發作|沒有大發作|沒有小發作"
)
SZ_STATIONARY_RE = re.compile(r"(?i)\bstationary\b|穩定")
SZ_DIRECT_CONTEXT_RE = re.compile(
    r"(?i)\b(sz|seizure|seizures|convulsive|convulsion|convulsions|gtc|cps|sps|aura|"
    r"tonic|clonic|staring|unresponsive|loss of consciousness|loc|focal motor)\b"
    r"|發作|抽搐|抽蓄|臉部扭曲|微放電|放電|失神|恍神|僵硬|眼神呆滯"
)
SZ_NON_EVENT_CONTEXT_RE = re.compile(
    r"(?i)\b(headache|migraine|dizziness|pain|fever|cough|nausea|vomit|sleepiness|"
    r"insomnia|memory|diarrhea|constipation|mood|menstruation|menses|period)\b"
    r"|頭痛|頭暈|發燒|咳|感冒|噁心|月經|生理期|情緒"
)
SZ_EEG_CONTEXT_RE = re.compile(r"(?i)\b(eeg|spike|spikes|sharp waves?|slow waves?|sleep stage)\b")
SZ_HISTORICAL_RE = re.compile(
    r"(?i)\b(before\s*(?:aed|asm|med|treatment)|pre[- ]?(?:aed|asm|op)|prior to|previous(?:ly)?|"
    r"used to|baseline|history|historical|since\s+\d{4}|last seizure)\b|以前|過去|病史|術後"
)

SZ_ABS_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})[./-](?P<month>\d{1,2})(?:[./-](?P<day>\d{1,2}))?(?!\d)"
)
SZ_COMPACT_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<month>\d{2})(?P<day>\d{2})?(?!\d)"
)
SZ_YEAR_MONTH_NAME_RE = re.compile(
    r"(?i)(?<!\d)(?P<year>(?:19|20)\d{2})[./\-\s]*(?P<month_name>"
    + "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
    + r")\b"
)
SZ_MONTH_DAY_RE = re.compile(r"(?<!\d)(?P<month>\d{1,2})/(?P<day>\d{1,2})(?!\d)")
SZ_CONTEXT_WINDOW = 72
SZ_MAX_COUNT = 365
SZ_MAX_RANGE_CARRY_MONTHS = 3


@dataclass
class SeizureEvidence:
    month: str
    count_min: float
    count_max: float
    count_estimate: float
    confidence: str
    method: str
    evidence: str


def sz_month_key(year: int | str, month: int | str) -> str | None:
    year, month = int(year), int(month)
    return f"{year:04d}-{month:02d}" if 1900 <= year <= 2099 and 1 <= month <= 12 else None


def sz_parse_absolute_month(token: str) -> str | None:
    token = token.strip("() ")
    match = SZ_ABS_DATE_RE.search(token) or SZ_COMPACT_DATE_RE.search(token)
    if match:
        return sz_month_key(match.group("year"), match.group("month"))
    named = SZ_YEAR_MONTH_NAME_RE.search(token)
    if named:
        return sz_month_key(named.group("year"), MONTH_NAMES[named.group("month_name").lower()])
    try:
        serial = int(float(token))
    except (TypeError, ValueError):
        return None
    if 20000 <= serial <= 100000:
        date = pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D")
        return f"{date.year:04d}-{date.month:02d}"
    return None


def sz_date_markers(text: str) -> list[tuple[int, int, str, str]]:
    markers: list[tuple[int, int, str, str]] = []
    for regex in (SZ_ABS_DATE_RE, SZ_COMPACT_DATE_RE, SZ_YEAR_MONTH_NAME_RE):
        for match in regex.finditer(text):
            month = sz_parse_absolute_month(match.group(0))
            if month:
                markers.append((match.start(), match.end(), month, match.group(0)))
    markers.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[tuple[int, int, str, str]] = []
    last_end = -1
    for marker in markers:
        if marker[0] >= last_end:
            result.append(marker)
            last_end = marker[1]
    return result


def sz_split_segments(text: str) -> list[dict]:
    markers = sz_date_markers(text)
    result = []
    for index, marker in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        segment = text[marker[0] : end].strip(" ;,\n")
        if segment:
            result.append({"month": marker[2], "text": segment})
    return result


def sz_match_overlaps_date(text: str, start: int, end: int) -> bool:
    for regex in (SZ_ABS_DATE_RE, SZ_COMPACT_DATE_RE):
        for match in regex.finditer(text):
            if match.start() <= start < match.end() or match.start() < end <= match.end():
                return True
    return False


def sz_count_match_valid(text: str, match) -> bool:
    start, end = match.span()
    if sz_match_overlaps_date(text, start, end):
        return False
    before_char = text[:start].rstrip()[-1:] or ""
    if before_char in {"-", "~"}:
        return False
    context = text[max(0, start - SZ_CONTEXT_WINDOW) : min(len(text), end + SZ_CONTEXT_WINDOW)]
    before = text[max(0, start - SZ_CONTEXT_WINDOW) : start]
    if SZ_HISTORICAL_RE.search(before):
        return False
    direct = SZ_DIRECT_CONTEXT_RE.search(context)
    if SZ_EEG_CONTEXT_RE.search(context) and not direct:
        return False
    if SZ_NON_EVENT_CONTEXT_RE.search(context) and not direct:
        return False
    return True


def sz_parse_frequency(text: str) -> tuple[float, float, float, str] | None:
    patterns = [
        (re.compile(r"(?i)(?P<min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<max>\d+(?:\.\d+)?)\s*/\s*(?P<m1>\d+(?:\.\d+)?)\s*[-~]\s*(?P<m2>\d+(?:\.\d+)?)\s*(?:m\b|months?\b)"), "count_range_per_n_months"),
        (re.compile(r"(?i)(?P<min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<max>\d+(?:\.\d+)?)\s*(?:times?|x|episodes?|seizures?|sz)?\s*(?:of seizure)?\s*(?:/\s*|per\s*)month\b"), "monthly_range"),
        (re.compile(r"(?i)(?P<count>\d+(?:\.\d+)?)\s*\+\s*(?:times?|x|episodes?|seizures?|sz)?\s*(?:/\s*|per\s*)month\b"), "monthly_lower_bound"),
        (re.compile(r"(?i)(?P<count>\d+(?:\.\d+)?)\s*(?:times?|x|episodes?|seizures?|sz)?\s*(?:of seizure)?\s*(?:/\s*|per\s*)month\b"), "monthly_count"),
        (re.compile(r"(?i)<\s*(?P<count>\d+(?:\.\d+)?)\s*/\s*(?P<months>\d+(?:\.\d+)?)\s*(?:m\b|months?\b)"), "upper_bound_per_n_months"),
        (re.compile(r"(?i)(?P<count>\d+(?:\.\d+)?)\s*/\s*(?P<months>\d+(?:\.\d+)?)\s*(?:m\b|months?\b)"), "count_per_n_months"),
    ]
    for regex, method in patterns:
        for match in regex.finditer(text):
            if not sz_count_match_valid(text, match):
                continue
            if method == "count_range_per_n_months":
                low, high = float(match.group("min")), float(match.group("max"))
                m1, m2 = float(match.group("m1")), float(match.group("m2"))
                if low <= high and m1 > 0 and m2 > 0:
                    monthly_low, monthly_high = low / max(m1, m2), high / min(m1, m2)
                    return monthly_low, monthly_high, (monthly_low + monthly_high) / 2, method
            elif method == "monthly_range":
                low, high = float(match.group("min")), float(match.group("max"))
                if low <= high:
                    return low, high, (low + high) / 2, method
            elif method in {"monthly_lower_bound", "monthly_count"}:
                count = float(match.group("count"))
                return count, count, count, method
            else:
                count, months = float(match.group("count")), float(match.group("months"))
                if months > 0:
                    monthly = count / months
                    return (0.0, monthly, monthly, method) if method.startswith("upper") else (monthly, monthly, monthly, method)
    return None


def sz_explicit_counts(text: str) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    patterns = [
        re.compile(r"(?i)\b(?:seizures?|sz)\s+(?:noted\s+)?(?:for\s+)?(?P<count>\d+(?:\.\d+)?)\s+times?\b"),
        re.compile(r"(?i)(?<![\w.])(?P<count>\d+(?:\.\d+)?)\s*x\b"),
        re.compile(r"(?i)\bx\s*(?P<count>\d+(?:\.\d+)?)\b"),
        re.compile(r"(?i)\b(?P<count>\d+)\s*(?:seizures|episodes|events)\b"),
    ]
    for regex in patterns:
        for match in regex.finditer(text):
            count = float(match.group("count"))
            if 0 <= count <= SZ_MAX_COUNT and sz_count_match_valid(text, match):
                result.append((count, match.group(0)))
    word_values = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    word_re = re.compile(r"(?i)\b(?P<count>" + "|".join(word_values) + r")\s+(?:seizure|seizures|episode|episodes|events?|gtc|cps|sps)\b")
    for match in word_re.finditer(text):
        if sz_count_match_valid(text, match):
            result.append((float(word_values[match.group("count").lower()]), match.group(0)))
    return result


def sz_month_specific_counts(segment_text: str, default_month: str) -> list[tuple[str, float, str]]:
    result: list[tuple[str, float, str]] = []
    month_names = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
    count_expression = r"x\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*x|one seizure|once|two seizures|twice"

    for match in re.finditer(
        r"(?i)(?P<year>(?:19|20)\d{2})[./\-\s]*(?P<month_name>"
        + month_names
        + r")\.?\s*(?P<count>"
        + count_expression
        + r")",
        segment_text,
    ):
        month = sz_month_key(match.group("year"), MONTH_NAMES[match.group("month_name").lower()])
        counts = sz_explicit_counts(match.group("count"))
        if month and counts:
            result.append((month, counts[0][0], match.group(0)))

    default_year = int(default_month[:4]) if default_month else None
    if default_year:
        for match in re.finditer(
            r"(?i)\b(?P<month_name>"
            + month_names
            + r")\.?\s*(?P<count>"
            + count_expression
            + r")",
            segment_text,
        ):
            month = sz_month_key(default_year, MONTH_NAMES[match.group("month_name").lower()])
            counts = sz_explicit_counts(match.group("count"))
            if month and counts:
                result.append((month, counts[0][0], match.group(0)))

        for match in re.finditer(
            r"(?<!\d)(?P<month>\d{1,2})/(?P<day>\d{1,2})\s*[:;,]?\s*"
            r"(?P<count>"
            + count_expression
            + r")?",
            segment_text,
        ):
            month = sz_month_key(default_year, match.group("month"))
            count_text = match.group("count")
            counts = sz_explicit_counts(count_text) if count_text else []
            if month and counts:
                result.append((month, counts[0][0], match.group(0)))
    return result


def sz_extract_segment(segment: dict) -> tuple[list[SeizureEvidence], list[tuple[str, str, str]]]:
    text, month = segment["text"], segment["month"]
    evidence: list[SeizureEvidence] = []
    reviews: list[tuple[str, str, str]] = []
    specific_counts = sz_month_specific_counts(text, month)
    for specific_month, count, snippet in specific_counts:
        evidence.append(SeizureEvidence(specific_month, count, count, count, "high", "month_specific_count", snippet))

    frequency = sz_parse_frequency(text)
    if frequency and not specific_counts:
        low, high, estimate, method = frequency
        confidence = "low" if method == "upper_bound_per_n_months" else "medium"
        evidence.append(SeizureEvidence(month, low, high, estimate, confidence, method, text))
        if method in {"upper_bound_per_n_months", "count_per_n_months", "count_range_per_n_months"}:
            reviews.append(("period_frequency", f"Converted period frequency with method={method}.", text))
    counts = sz_explicit_counts(text)
    if counts and not frequency and not specific_counts:
        total = sum(value for value, _snippet in counts)
        evidence.append(SeizureEvidence(month, total, total, total, "high", "explicit_count_sum", text))
    if SZ_NO_EVENT_RE.search(text) and not evidence:
        evidence.append(SeizureEvidence(month, 0.0, 0.0, 0.0, "high", "explicit_no_seizure", text))
    elif SZ_NO_EVENT_RE.search(text) and evidence:
        reviews.append(("mixed_no_seizure_and_count", "Segment contains both no-seizure language and a count.", text))
    if not evidence:
        historical = SZ_HISTORICAL_RE.search(text) and not re.search(r"(?i)\b(recur|recurrent|visited|er|today|this month)\b", text)
        inferred = not SZ_NO_EVENT_RE.search(text) and not historical and not (SZ_STATIONARY_RE.search(text) and not frequency) and SZ_KEYWORDS.search(text)
        if inferred:
            evidence.append(SeizureEvidence(month, 1.0, 1.0, 1.0, "low", "inferred_single_event", text))
            reviews.append(("inferred_count", "Seizure event had no explicit count; inferred 1.", text))
        elif SZ_KEYWORDS.search(text):
            reviews.append(("uninterpretable_seizure_text", "Seizure-related text had no interpretable count.", text))
    return evidence, reviews


def sz_value_after_label(table, label: str) -> str:
    label_key = normalize_label(label)
    for source_row in table.rows:
        values = get_unique_row_values(source_row)
        for index, value in enumerate(values):
            if label_key in normalize_label(value):
                return values[index + 1] if index + 1 < len(values) else ""
    return ""


def parse_seizure(doc, path: Path, patient_id: str, cohort: str, errors: list[dict]) -> list[dict]:
    asm_table = next((table for table in doc.tables if table_title(table) == "asm狀況"), None)
    soap = sz_value_after_label(asm_table, "原因 SOAP") if asm_table is not None else ""
    if not soap:
        add_error(
            errors, path, patient_id, cohort, "seizure", "missing_soap_section",
            "Could not find ASM狀況 -> 原因 SOAP content.",
        )
        return []
    segments = sz_split_segments(soap)
    if not segments:
        add_error(
            errors, path, patient_id, cohort, "seizure", "no_dated_segments",
            "SOAP content had no parseable absolute dates.", evidence=soap,
        )
        return []

    all_evidence: list[SeizureEvidence] = []
    for segment in segments:
        segment_evidence, reviews = sz_extract_segment(segment)
        all_evidence.extend(segment_evidence)
        for error_type, message, raw in reviews:
            add_error(errors, path, patient_id, cohort, "seizure", error_type, message, evidence=raw)

    grouped: dict[str, list[SeizureEvidence]] = defaultdict(list)
    for evidence in all_evidence:
        grouped[evidence.month].append(evidence)
    frequency_methods = {
        "monthly_range", "monthly_count", "monthly_lower_bound", "count_per_n_months",
        "count_range_per_n_months", "upper_bound_per_n_months",
    }
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    rows: list[dict] = []
    for month, items in sorted(grouped.items()):
        selected = [item for item in items if item.method in frequency_methods] or items
        if any(item.method in frequency_methods for item in items):
            adjusted = [
                (item.count_min, item.count_min, item.count_min)
                if item.method == "monthly_range"
                else (item.count_min, item.count_max, item.count_estimate)
                for item in selected
            ]
            count_min = max(item[0] for item in adjusted)
            count_max = max(item[1] for item in adjusted)
            estimate = max(item[2] for item in adjusted)
        else:
            count_min = sum(item.count_min for item in selected)
            count_max = sum(item.count_max for item in selected)
            estimate = sum(item.count_estimate for item in selected)
        confidence = min(selected, key=lambda item: confidence_rank.get(item.confidence, 0)).confidence
        rows.append(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "date": pd.to_datetime(month, format="%Y-%m"),
                "seizure_count_min": count_min,
                "seizure_count_max": count_max,
                "seizure_count_estimate": estimate,
                "confidence": confidence,
                "evidence_count": len(items),
                "methods": "; ".join(sorted({item.method for item in items})),
                "evidence": " || ".join(normalize_text(item.evidence)[:260] for item in items[:5]),
            }
        )

    carried: list[dict] = []
    for index, row in enumerate(rows[:-1]):
        methods = {method.strip() for method in row["methods"].split(";")}
        if "monthly_range" not in methods:
            continue
        current = row["date"].to_period("M")
        following = rows[index + 1]["date"].to_period("M")
        gap = following.ordinal - current.ordinal
        for offset in range(1, min(gap, SZ_MAX_RANGE_CARRY_MONTHS + 1)):
            filled = dict(row)
            filled["date"] = (current + offset).to_timestamp()
            filled["seizure_count_max"] = filled["seizure_count_min"]
            filled["seizure_count_estimate"] = filled["seizure_count_min"]
            filled["evidence_count"] = 0
            filled["methods"] = "monthly_range_lower_bound_carry_forward"
            carried.append(filled)
    return sorted(rows + carried, key=lambda row: row["date"])


# ---------------------------------------------------------------------------
# Batch API, schema normalization, and CLI
# ---------------------------------------------------------------------------


def empty_result_rows() -> dict[str, list[dict]]:
    return {key: [] for key in DATAFRAME_COLUMNS}


def _run_category(
    name: str,
    parser,
    doc,
    path: Path,
    patient_id: str,
    cohort: str,
    errors: list[dict],
):
    try:
        return parser(doc, path, patient_id, cohort, errors)
    except Exception as exc:  # category isolation is part of the public contract
        add_error(errors, path, patient_id, cohort, name, "parser_exception", str(exc))
        return []


def _build_dataframe(name: str, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=DATAFRAME_COLUMNS[name])
    if name == "parse_errors":
        frame["row_number"] = pd.to_numeric(frame["row_number"], errors="coerce").astype("Int64")
        return frame

    for column in COMMON_COLUMNS:
        frame[column] = frame[column].astype("string")

    date_columns = {
        "asm": ["date", "observed_date"],
        "seizure": ["date"],
        "eeg": ["report_date"],
        "lab_general": ["date"],
        "lab_routine": ["date"],
        "lab_drug": ["date"],
        "cognitive": ["date"],
        "brain_image": ["date"],
    }
    for column in date_columns.get(name, []):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    if name == "static":
        frame["adverse_for_asms"] = frame["adverse_for_asms"].astype("boolean")
    elif name == "asm":
        numeric = [
            "unit_dose", "administered_dose", "frequency_per_day", "daily_dose_mg",
            "ddd", "total_ddd",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        frame["number_of_drugs"] = pd.to_numeric(frame["number_of_drugs"], errors="coerce").astype("Int64")
        frame["is_carried_forward"] = frame["is_carried_forward"].astype("boolean")
    elif name == "seizure":
        for column in ["seizure_count_min", "seizure_count_max", "seizure_count_estimate"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        frame["evidence_count"] = pd.to_numeric(frame["evidence_count"], errors="coerce").astype("Int64")
    elif name.startswith("lab_"):
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("Float64")
    return frame


def parse_docx_files(paths: Sequence[str | Path] | str | Path) -> dict[str, pd.DataFrame]:
    """Parse DOCX files into a stable dictionary of combined DataFrames.

    Directories are searched recursively.  A malformed file or category is
    recorded in ``parse_errors`` and does not prevent other outputs.
    """
    rows = empty_result_rows()
    files = collect_docx_files(paths)

    for path in files:
        initial_patient_id = patient_id_from_path(path)
        cohort = cohort_from_path(path)
        try:
            doc = Document(str(path))
        except Exception as exc:
            add_error(
                rows["parse_errors"], path, initial_patient_id, cohort, "document",
                "open_error", str(exc),
            )
            continue

        try:
            static_row = parse_static(doc, path, initial_patient_id, cohort, rows["parse_errors"])
        except Exception as exc:
            add_error(
                rows["parse_errors"], path, initial_patient_id, cohort, "static",
                "parser_exception", str(exc),
            )
            static_row = {column: "" for column in COMMON_COLUMNS + STATIC_VALUE_COLUMNS}
            static_row.update(
                {
                    "patient_id": initial_patient_id,
                    "cohort": cohort,
                    "adverse_for_asms": pd.NA,
                    "pregnancy": pd.NA,
                }
            )
        patient_id = str(static_row.get("patient_id") or initial_patient_id)
        static_row["patient_id"] = patient_id
        static_row["cohort"] = cohort
        rows["static"].append(static_row)

        rows["asm"].extend(
            _run_category("asm", parse_asm, doc, path, patient_id, cohort, rows["parse_errors"])
        )
        rows["seizure"].extend(
            _run_category("seizure", parse_seizure, doc, path, patient_id, cohort, rows["parse_errors"])
        )
        rows["eeg"].extend(
            _run_category("eeg", parse_eeg, doc, path, patient_id, cohort, rows["parse_errors"])
        )
        lab_outputs = _run_category("lab", parse_labs, doc, path, patient_id, cohort, rows["parse_errors"])
        if isinstance(lab_outputs, dict):
            for key in ("lab_general", "lab_routine", "lab_drug"):
                rows[key].extend(lab_outputs.get(key, []))
        rows["cognitive"].extend(
            _run_category("cognitive", parse_cognitive, doc, path, patient_id, cohort, rows["parse_errors"])
        )
        rows["brain_image"].extend(
            _run_category("brain_image", parse_brain_image, doc, path, patient_id, cohort, rows["parse_errors"])
        )

    return {name: _build_dataframe(name, table_rows) for name, table_rows in rows.items()}


def export_dataframes(dataframes: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name in DATAFRAME_COLUMNS:
        path = output_path / f"{name}.csv"
        dataframes[name].to_csv(path, index=False, encoding="utf-8-sig")
        saved[name] = path
    return saved


def mysql_identifier(value: object) -> str:
    """Create a stable lowercase identifier suitable for dynamic MySQL columns."""
    identifier = re.sub(r"[^a-z0-9]+", "_", normalize_text(value).casefold()).strip("_")
    if not identifier:
        identifier = "drug"
    if identifier[0].isdigit():
        identifier = f"drug_{identifier}"
    return identifier


def safe_filename_component(value: object) -> str:
    component = re.sub(r"[^\w.-]+", "_", normalize_text(value), flags=re.UNICODE).strip("_")
    return component or "unknown"


def patient_asm_key(patient_id: object, cohort: object) -> str:
    return f"{normalize_text(patient_id)}_{normalize_text(cohort)}"


def _format_join_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:g}"
    return normalize_text(value)


def _join_asm_values(values: Iterable[object]) -> str | pd._libs.missing.NAType:
    joined: list[str] = []
    seen = set()
    for value in values:
        formatted = _format_join_value(value)
        if formatted and formatted not in seen:
            joined.append(formatted)
            seen.add(formatted)
    return "; ".join(joined) if joined else pd.NA


def _scalar_or_join_asm_values(values: Iterable[object]):
    originals: list[object] = []
    formatted_values: list[str] = []
    seen = set()
    for value in values:
        formatted = _format_join_value(value)
        if formatted and formatted not in seen:
            originals.append(value)
            formatted_values.append(formatted)
            seen.add(formatted)
    if not originals:
        return pd.NA
    if len(originals) == 1:
        return originals[0]
    return "; ".join(formatted_values)


def _sum_with_missing(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.sum()) if numeric.notna().any() else np.nan


def build_patient_asm_dataframes(asm_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Transform the authoritative long ASM table into patient-specific wide tables."""
    missing_columns = [column for column in ASM_COLUMNS if column not in asm_df.columns]
    if missing_columns:
        raise ValueError(f"ASM DataFrame is missing required columns: {', '.join(missing_columns)}")
    if asm_df.empty:
        return {}

    working = asm_df.copy()
    working["patient_id"] = working["patient_id"].astype("string")
    working["cohort"] = working["cohort"].astype("string")
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["observed_date"] = pd.to_datetime(working["observed_date"], errors="coerce")

    frames: dict[str, pd.DataFrame] = {}
    identity_rows = working[["patient_id", "cohort"]].dropna().drop_duplicates()
    for identity in identity_rows.sort_values(["cohort", "patient_id"]).itertuples(index=False):
        patient_id = str(identity.patient_id)
        cohort = str(identity.cohort)
        patient = working[
            working["patient_id"].eq(patient_id) & working["cohort"].eq(cohort)
        ].copy()
        prescriptions = patient[
            patient["record_type"].eq("prescription") & patient["date"].notna()
        ].copy()
        summaries = patient[
            patient["record_type"].eq("monthly_summary") & patient["date"].notna()
        ].copy()

        all_dates = sorted(set(prescriptions["date"].tolist()) | set(summaries["date"].tolist()))
        if not all_dates:
            continue
        wide = pd.DataFrame({"date": pd.to_datetime(all_dates)})
        wide.insert(0, "cohort", cohort)
        wide.insert(0, "patient_id", patient_id)

        summary_values = (
            summaries.sort_values("date")
            .drop_duplicates("date", keep="last")
            .set_index("date")[["is_carried_forward", "total_ddd", "number_of_drugs"]]
        )
        wide = wide.merge(summary_values, how="left", left_on="date", right_index=True)

        if not prescriptions.empty:
            calculated_total = prescriptions.groupby("date", sort=False)["ddd"].apply(_sum_with_missing)
            calculable = prescriptions[pd.to_numeric(prescriptions["ddd"], errors="coerce").notna()]
            calculated_count = calculable.groupby("date", sort=False)["generic_name"].nunique()
            calculated_carried_forward = prescriptions.groupby("date", sort=False)[
                "is_carried_forward"
            ].apply(
                lambda values: (
                    bool(values.dropna().astype(bool).all())
                    if values.notna().any()
                    else pd.NA
                )
            )
            wide["total_ddd"] = pd.to_numeric(wide["total_ddd"], errors="coerce").fillna(
                wide["date"].map(calculated_total)
            )
            wide["number_of_drugs"] = pd.to_numeric(wide["number_of_drugs"], errors="coerce").fillna(
                wide["date"].map(calculated_count).fillna(0)
            )
            wide["is_carried_forward"] = wide["is_carried_forward"].astype("boolean").fillna(
                wide["date"].map(calculated_carried_forward).astype("boolean")
            )

        drug_names = sorted(
            (name for name in prescriptions["generic_name"].dropna().astype(str).unique() if name),
            key=str.casefold,
        )
        used_prefixes: set[str] = set()
        dynamic_columns: list[str] = []
        for generic_name in drug_names:
            prefix_base = mysql_identifier(generic_name)
            prefix = prefix_base
            suffix = 2
            while prefix in used_prefixes:
                prefix = f"{prefix_base}_{suffix}"
                suffix += 1
            used_prefixes.add(prefix)

            drug_rows = prescriptions[prescriptions["generic_name"].eq(generic_name)]
            monthly_records: list[dict] = []
            for date, month_rows in drug_rows.groupby("date", sort=True):
                observed_dates = month_rows["observed_date"].dropna()
                record = {
                    "date": date,
                    f"{prefix}__drug_label": _join_asm_values(month_rows["drug_label"]),
                    f"{prefix}__observed_date": observed_dates.max() if not observed_dates.empty else pd.NaT,
                    f"{prefix}__unit_dose": _scalar_or_join_asm_values(month_rows["unit_dose"]),
                    f"{prefix}__unit_dose_unit": _join_asm_values(month_rows["unit_dose_unit"]),
                    f"{prefix}__administered_dose": _join_asm_values(month_rows["administered_dose"]),
                    f"{prefix}__administration_unit": _join_asm_values(month_rows["administration_unit"]),
                    f"{prefix}__frequency": _join_asm_values(month_rows["frequency"]),
                    f"{prefix}__frequency_per_day": _join_asm_values(month_rows["frequency_per_day"]),
                    f"{prefix}__daily_dose_mg": _sum_with_missing(month_rows["daily_dose_mg"]),
                    f"{prefix}__ddd": _sum_with_missing(month_rows["ddd"]),
                    f"{prefix}__raw_instruction": _join_asm_values(month_rows["raw_instruction"]),
                }
                monthly_records.append(record)

            drug_frame = pd.DataFrame(monthly_records)
            wide = wide.merge(drug_frame, on="date", how="left")
            dynamic_columns.extend(f"{prefix}__{field}" for field in PATIENT_ASM_DRUG_FIELDS)

        wide = wide.reindex(columns=PATIENT_ASM_BASE_COLUMNS + dynamic_columns)
        wide["patient_id"] = wide["patient_id"].astype("string")
        wide["cohort"] = wide["cohort"].astype("string")
        wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
        wide["is_carried_forward"] = wide["is_carried_forward"].astype("boolean")
        wide["total_ddd"] = pd.to_numeric(wide["total_ddd"], errors="coerce").astype("Float64")
        wide["number_of_drugs"] = pd.to_numeric(wide["number_of_drugs"], errors="coerce").astype("Int64")
        frames[patient_asm_key(patient_id, cohort)] = wide.sort_values("date").reset_index(drop=True)
    return frames


def export_asm_per_patient(
    dataframes: dict[str, pd.DataFrame], output_dir: str | Path
) -> pd.DataFrame:
    """Write one wide ASM CSV per patient plus manifest and ASM error report."""
    if "asm" not in dataframes or "static" not in dataframes or "parse_errors" not in dataframes:
        raise ValueError("dataframes must contain asm, static, and parse_errors tables")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    patient_frames = build_patient_asm_dataframes(dataframes["asm"])

    static_identities = dataframes["static"][["patient_id", "cohort"]].dropna().drop_duplicates()
    asm_identities = dataframes["asm"][["patient_id", "cohort"]].dropna().drop_duplicates()
    roster = pd.concat([static_identities, asm_identities], ignore_index=True).drop_duplicates()
    if not roster.empty:
        roster["patient_id"] = roster["patient_id"].astype("string")
        roster["cohort"] = roster["cohort"].astype("string")
        roster = roster.sort_values(["cohort", "patient_id"])

    errors = dataframes["parse_errors"].copy()
    if errors.empty:
        asm_errors = pd.DataFrame(columns=ERROR_COLUMNS)
    else:
        asm_errors = errors[errors["category"].astype(str).str.casefold().eq("asm")].copy()
        asm_errors = asm_errors.reindex(columns=ERROR_COLUMNS)
    asm_errors.to_csv(output_path / "asm_parse_errors.csv", index=False, encoding="utf-8-sig")

    manifest_rows: list[dict] = []
    used_filenames: set[str] = set()
    for identity in roster.itertuples(index=False):
        patient_id = str(identity.patient_id)
        cohort = str(identity.cohort)
        key = patient_asm_key(patient_id, cohort)
        filename = (
            f"{safe_filename_component(patient_id)}_"
            f"{safe_filename_component(cohort)}_asm.csv"
        )
        filename_key = filename.casefold()
        if filename_key in used_filenames:
            raise ValueError(f"Patient ASM filename collision: {filename}")
        used_filenames.add(filename_key)

        frame = patient_frames.get(key)
        if frame is None:
            frame = pd.DataFrame(columns=PATIENT_ASM_BASE_COLUMNS)
            status = "no_asm_rows"
        else:
            status = "ok"
        frame.to_csv(output_path / filename, index=False, encoding="utf-8-sig")

        patient_asm = dataframes["asm"][
            dataframes["asm"]["patient_id"].astype(str).eq(patient_id)
            & dataframes["asm"]["cohort"].astype(str).eq(cohort)
            & dataframes["asm"]["record_type"].eq("prescription")
        ]
        drug_names = sorted(
            (name for name in patient_asm["generic_name"].dropna().astype(str).unique() if name),
            key=str.casefold,
        )
        patient_errors = asm_errors[
            asm_errors["patient_id"].astype(str).eq(patient_id)
            & asm_errors["cohort"].astype(str).eq(cohort)
        ]
        valid_dates = pd.to_datetime(frame["date"], errors="coerce").dropna() if "date" in frame else pd.Series(dtype="datetime64[ns]")
        manifest_rows.append(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "csv_file": filename,
                "status": status,
                "row_count": len(frame),
                "first_date": valid_dates.min().date().isoformat() if not valid_dates.empty else "",
                "last_date": valid_dates.max().date().isoformat() if not valid_dates.empty else "",
                "drug_count": len(drug_names),
                "drug_names": "; ".join(drug_names),
                "asm_error_count": len(patient_errors),
            }
        )

    manifest = pd.DataFrame(manifest_rows, columns=PATIENT_ASM_MANIFEST_COLUMNS)
    manifest.to_csv(output_path / "manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse patient DOCX files into database-ready clinical DataFrames."
    )
    parser.add_argument("paths", nargs="+", help="DOCX files or directories searched recursively.")
    parser.add_argument("-o", "--output-dir", help="Optionally export one CSV per DataFrame.")
    parser.add_argument(
        "--asm-per-patient-dir",
        help=(
            "Export one wide monthly ASM CSV per patient plus a manifest. "
            "Defaults to OUTPUT_DIR/asm_per_patient when --output-dir is used, "
            f"otherwise {DEFAULT_ASM_PER_PATIENT_DIR}."
        ),
    )
    args = parser.parse_args(argv)

    dataframes = parse_docx_files(args.paths)
    print("Parsed DataFrames:")
    for name, frame in dataframes.items():
        print(f"  {name}: {len(frame)} row(s)")

    if args.output_dir:
        saved = export_dataframes(dataframes, args.output_dir)
        print(f"Exported {len(saved)} CSV files to {Path(args.output_dir).resolve()}")

    asm_per_patient_dir = (
        Path(args.asm_per_patient_dir)
        if args.asm_per_patient_dir
        else Path(args.output_dir) / "asm_per_patient"
        if args.output_dir
        else DEFAULT_ASM_PER_PATIENT_DIR
    )
    if asm_per_patient_dir:
        manifest = export_asm_per_patient(dataframes, asm_per_patient_dir)
        exported_count = int(manifest["csv_file"].notna().sum()) if not manifest.empty else 0
        print(
            f"Exported {exported_count} patient ASM CSV files plus manifest/errors to "
            f"{asm_per_patient_dir.resolve()}"
        )

    errors = dataframes["parse_errors"]
    if not errors.empty:
        print("\nParser issues:")
        for row in errors.itertuples(index=False):
            print(f"  [{row.category}/{row.error_type}] {row.source_file}: {row.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
