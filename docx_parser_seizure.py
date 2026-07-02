import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from docx import Document

DEFAULT_INPUT_DIRS = ("DRE", "NDRE", "DRE_batch2", "NDRE_batch2")
DEFAULT_OUTPUT_DIR = "new_csvs"

MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

SEIZURE_KEYWORDS = re.compile(
    r"(?i)\b(sz|seizure|seizures|convulsive|convulsion|convulsions|gtc|cps|sps|aura)\b"
    r"|發作|抽蓄|抽搐|臉部扭曲|vocalization"
)
NO_SEIZURE_RE = re.compile(
    r"(?i)\b(no seizure|no seizures|no clinical seizure|no event|no further seizure|seizure free|seizure-free|stationary)\b"
    r"|無發作|沒有發作|未發作"
)
STATIONARY_RE = re.compile(r"(?i)\bstationary\b|穩定")

ASM_TABLE = "ASM\u72c0\u6cc1"
ASM_SOAP_LABEL = "\u539f\u56e0 SOAP"
ASM_PLAN_LABEL = "\u8a3a\u65b7\u8a08\u756b"
DRE_LABEL = "DRE\u2606"

SEIZURE_KEYWORDS = re.compile(
    r"(?i)\b(sz|seizure|seizures|convulsive|convulsion|convulsions|gtc|cps|sps|aura)\b"
    r"|vocalization|\u767c\u4f5c|\u62bd\u6410|\u62bd\u84c4|\u81c9\u90e8\u626d\u66f2"
    r"|\u5fae\u653e\u96fb|\u653e\u96fb|\u5931\u795e|\u604d\u795e"
)
NO_SEIZURE_RE = re.compile(
    r"(?i)(?<![A-Za-z])(no seizure|no seizures|no clinical seizure|no event|no further seizure|seizure free|seizure-free)\b"
    r"|\u7121\u767c\u4f5c|\u6c92\u6709\u767c\u4f5c|\u672a\u767c\u4f5c|\u6c92\u6709\u5927\u767c\u4f5c|\u6c92\u6709\u5c0f\u767c\u4f5c"
)
STATIONARY_RE = re.compile(r"(?i)\bstationary\b|\u7a69\u5b9a")

DIRECT_SEIZURE_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"sz|seizure|seizures|convulsive|convulsion|convulsions|gtc|cps|sps|aura|"
    r"tonic|clonic|staring|unresponsive|unresponsiveness|loss of consciousness|loc|"
    r"blank minded|grimacing|laughing|focal motor"
    r")\b"
    r"|\u767c\u4f5c|\u62bd\u6410|\u62bd\u84c4|\u81c9\u90e8\u626d\u66f2|\u5fae\u653e\u96fb|\u653e\u96fb"
    r"|\u5931\u795e|\u604d\u795e|\u50f5\u786c|\u773c\u795e\u5446\u6eef"
)
NON_SEIZURE_COUNT_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"headache|migraine|dizziness|dizzy|pain|fever|cough|cold|rhinorrhea|nausea|"
    r"vomit|sleepiness|sleepy|insomnia|memory|forget|diarrhea|constipation|mood|"
    r"irritability|menstruation|menses|period"
    r")\b"
    r"|\u982d\u75db|\u982d\u6688|\u6688|\u75bc|\u75db|\u767c\u71d2|\u54b3|\u611f\u5192"
    r"|\u60e1\u5fc3|\u5410|\u60f3\u7761|\u55dc\u7761|\u8a18\u61b6|\u5fd8|\u6708\u7d93|\u751f\u7406\u671f|\u60c5\u7dd2"
)
EEG_COUNT_CONTEXT_RE = re.compile(
    r"(?i)\b(eeg|spike|spikes|sharp waves?|slow waves?|ictal-like|hv|sleep stage|drowsy state)\b"
)
HISTORICAL_COUNT_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"before\s*(?:aed|asm|med|medication|treatment|drug)|pre[- ]?(?:aed|asm|op)|"
    r"prior to|previous(?:ly)?|used to|baseline|history|historical|diagnosed since|"
    r"since\s+\d{4}|before add(?:ing)?|before taper|post-op|last seizure"
    r")\b"
    r"|\u4ee5\u524d|\u904e\u53bb|\u75c5\u53f2|\u8853\u5f8c"
)

COUNT_CONTEXT_WINDOW = 72
MAX_PLAUSIBLE_EXPLICIT_COUNT = 365
MAX_MONTHLY_RANGE_CARRY_FORWARD_MONTHS = 3

ABS_DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?P<year>(?:19|20)\d{2})"
    r"[./-]"
    r"(?P<month>\d{1,2})"
    r"(?:[./-](?P<day>\d{1,2}))?"
    r"(?!\d)"
)
COMPACT_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<month>\d{2})(?P<day>\d{2})?(?!\d)"
)
YEAR_MONTH_NAME_RE = re.compile(
    r"(?i)(?<!\d)(?P<year>(?:19|20)\d{2})[./\-\s]*(?P<month_name>"
    + "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
    + r")\b"
)
MONTH_NAME_RE = re.compile(
    r"(?i)\b(?P<month_name>" + "|".join(sorted(MONTH_NAMES, key=len, reverse=True)) + r")\b"
)
MONTH_DAY_RE = re.compile(r"(?<!\d)(?P<month>\d{1,2})/(?P<day>\d{1,2})(?!\d)")


@dataclass
class Evidence:
    patient_id: str
    cohort: str
    source_file: str
    month: str
    count_min: float
    count_max: float
    count_estimate: float
    confidence: str
    source_section: str
    method: str
    evidence: str


@dataclass
class ReviewIssue:
    patient_id: str
    cohort: str
    source_file: str
    month: str
    issue_type: str
    severity: str
    reason: str
    evidence: str


def normalize_text(text):
    text = text.replace("\xa0", " ")
    text = text.replace("→", "->")
    text = text.replace("〜", "~")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cell_text(cell):
    clean_text = cell.text.strip().replace("\n", " ")
    all_text_nodes = cell._tc.xpath(".//w:t")
    xpath_text = "".join([node.text for node in all_text_nodes if node.text]).strip()
    if len(xpath_text) > len(clean_text):
        return normalize_text(xpath_text)
    return normalize_text(clean_text)


def row_values(row):
    values = []
    visited_cells = set()
    for cell in row.cells:
        if cell._tc in visited_cells:
            continue
        visited_cells.add(cell._tc)
        values.append(cell_text(cell))
    return values


def read_docx_tables(path):
    doc = Document(path)
    for table in doc.tables:
        yield [row_values(row) for row in table.rows]


def first_cell(rows):
    if rows and rows[0]:
        return normalize_text(rows[0][0].replace("\n", " "))
    return ""


def flatten_table(rows):
    cells = []
    for row in rows:
        for cell in row:
            if cell:
                cells.append(cell)
    return cells


def value_after_label(rows, label):
    for row in rows:
        for idx, cell in enumerate(row):
            if label in cell:
                if idx + 1 < len(row):
                    return row[idx + 1]
                return ""
    return ""


def extract_sections_legacy(path):
    sections = {}
    for rows in read_docx_tables(path):
        header = first_cell(rows)
        if header == "ASM狀況":
            sections["asm_soap"] = value_after_label(rows, "原因 SOAP")
            sections["asm_plan"] = value_after_label(rows, "診斷計畫")
        elif header == "Semiology":
            flat = flatten_table(rows)
            for idx, cell in enumerate(flat[:-1]):
                if cell.startswith("Frequency"):
                    sections["semiology_frequency"] = flat[idx + 1]
                elif cell == "Descriptor":
                    sections["semiology_descriptor"] = flat[idx + 1]
                elif cell == "Provoking factor":
                    sections["semiology_provoking_factor"] = flat[idx + 1]
        elif header.startswith("Baseline"):
            flat = flatten_table(rows)
            for idx, cell in enumerate(flat[:-1]):
                if cell == "Patient ID":
                    sections["patient_id"] = flat[idx + 1]
                elif cell == "DRE☆":
                    sections["dre_label"] = flat[idx + 1]
                elif cell == "Last Seizure":
                    sections["baseline_last_seizure"] = flat[idx + 1]
    return {key: normalize_text(value) for key, value in sections.items() if value}


def extract_sections(path):
    sections = {}
    for rows in read_docx_tables(path):
        header = first_cell(rows)
        if header == ASM_TABLE:
            sections["asm_soap"] = value_after_label(rows, ASM_SOAP_LABEL)
            sections["asm_plan"] = value_after_label(rows, ASM_PLAN_LABEL)
        elif header == "Semiology":
            flat = flatten_table(rows)
            for idx, cell in enumerate(flat[:-1]):
                if cell.startswith("Frequency"):
                    sections["semiology_frequency"] = flat[idx + 1]
                elif cell == "Descriptor":
                    sections["semiology_descriptor"] = flat[idx + 1]
                elif cell == "Provoking factor":
                    sections["semiology_provoking_factor"] = flat[idx + 1]
        elif header.startswith("Baseline"):
            flat = flatten_table(rows)
            for idx, cell in enumerate(flat[:-1]):
                if cell == "Patient ID":
                    sections["patient_id"] = flat[idx + 1]
                elif cell == DRE_LABEL:
                    sections["dre_label"] = flat[idx + 1]
                elif cell == "Last Seizure":
                    sections["baseline_last_seizure"] = flat[idx + 1]
    return {key: normalize_text(value) for key, value in sections.items() if value}


def patient_id_from_path(path, sections):
    if sections.get("patient_id"):
        match = re.search(r"\d+", sections["patient_id"])
        if match:
            return match.group(0)
        return sections["patient_id"]
    match = re.search(r"\d+", Path(path).stem)
    return match.group(0) if match else Path(path).stem


def cohort_from_path(path):
    return Path(path).parent.name


def clean_snippet(text, max_len=260):
    text = normalize_text(text).replace("\n", " | ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def month_key(year, month):
    year = int(year)
    month = int(month)
    if 1900 <= year <= 2099 and 1 <= month <= 12:
        return f"{year:04d}-{month:02d}"
    return None


def parse_excel_serial_month(value):
    try:
        serial = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    if serial < 20000 or serial > 100000:
        return None
    # Excel's 1900 date system includes the historical leap-year bug.
    import datetime as _dt

    date = _dt.date(1899, 12, 30) + _dt.timedelta(days=serial)
    return f"{date.year:04d}-{date.month:02d}"


def parse_absolute_month(token):
    token = token.strip("() ")
    match = ABS_DATE_RE.search(token)
    if match:
        return month_key(match.group("year"), match.group("month"))
    match = COMPACT_DATE_RE.search(token)
    if match:
        return month_key(match.group("year"), match.group("month"))
    match = YEAR_MONTH_NAME_RE.search(token)
    if match:
        return month_key(match.group("year"), MONTH_NAMES[match.group("month_name").lower()])
    serial_month = parse_excel_serial_month(token)
    if serial_month:
        return serial_month
    return None


def parse_any_month(value):
    value = normalize_text(str(value or ""))
    if not value:
        return None
    month = parse_absolute_month(value)
    if month:
        return month
    match = MONTH_DAY_RE.search(value)
    if match:
        return None
    return None


def iter_date_markers(text):
    markers = []
    for regex in (ABS_DATE_RE, COMPACT_DATE_RE, YEAR_MONTH_NAME_RE):
        for match in regex.finditer(text):
            month = parse_absolute_month(match.group(0))
            if month:
                markers.append((match.start(), match.end(), month, match.group(0), "absolute"))

    markers.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    filtered = []
    last_end = -1
    for marker in markers:
        if marker[0] < last_end:
            continue
        filtered.append(marker)
        last_end = marker[1]
    return filtered


def split_into_dated_segments(text):
    markers = iter_date_markers(text)
    segments = []
    for idx, marker in enumerate(markers):
        start = marker[0]
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        segment = text[start:end].strip(" ;,\n")
        if segment:
            segments.append({"month": marker[2], "date_token": marker[3], "text": segment})
    return segments


def number_word_to_int(word):
    words = {
        "zero": 0,
        "one": 1,
        "once": 1,
        "two": 2,
        "twice": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return words.get(word.lower())


def local_context(text, start, end, window=COUNT_CONTEXT_WINDOW):
    return text[max(0, start - window) : min(len(text), end + window)]


def previous_nonspace_char(text, start):
    index = start - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return text[index] if index >= 0 else ""


def match_overlaps_date_token(text, start, end):
    for regex in (ABS_DATE_RE, COMPACT_DATE_RE):
        for match in regex.finditer(text):
            if match.start() <= start < match.end() or match.start() < end <= match.end():
                return True
    return False


def count_match_is_valid(text, match):
    start, end = match.span()
    if match_overlaps_date_token(text, start, end):
        return False
    if previous_nonspace_char(text, start) in {"-", "~"}:
        # Avoid counting the right side of ranges when a more specific range
        # pattern was rejected, for example "before AED20-25/month".
        return False

    context = local_context(text, start, end)
    before_context = text[max(0, start - COUNT_CONTEXT_WINDOW) : start]
    if HISTORICAL_COUNT_CONTEXT_RE.search(before_context):
        return False

    has_negated_seizure = NO_SEIZURE_RE.search(context)
    has_non_seizure_context = NON_SEIZURE_COUNT_CONTEXT_RE.search(context)
    has_direct_seizure_context = DIRECT_SEIZURE_CONTEXT_RE.search(context)
    has_eeg_context = EEG_COUNT_CONTEXT_RE.search(context)
    if has_negated_seizure and has_non_seizure_context:
        return False
    if has_eeg_context and not has_direct_seizure_context:
        return False
    if has_non_seizure_context and not has_direct_seizure_context:
        return False
    return True


def count_value_is_plausible(value):
    return 0 <= float(value) <= MAX_PLAUSIBLE_EXPLICIT_COUNT


def parse_frequency_count(text):
    patterns = [
        (
            re.compile(
                r"(?i)(?P<min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<max>\d+(?:\.\d+)?)\s*/\s*"
                r"(?P<months_min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<months_max>\d+(?:\.\d+)?)\s*"
                r"(?:m\b|months?\b)"
            ),
            "count_range_per_n_months",
        ),
        (
            re.compile(
                r"(?i)(?P<min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<max>\d+(?:\.\d+)?)\s*"
                r"(?:times?|x|episodes?|seizures?|sz)?\s*(?:of seizure)?\s*(?:/\s*|per\s*)month\b"
            ),
            "monthly_range",
        ),
        (
            re.compile(
                r"(?i)(?P<count>\d+(?:\.\d+)?)\s*\+\s*"
                r"(?:times?|x|episodes?|seizures?|sz)?\s*(?:of seizure)?\s*(?:/\s*|per\s*)month\b"
            ),
            "monthly_lower_bound",
        ),
        (
            re.compile(
                r"(?i)(?P<count>\d+(?:\.\d+)?)\s*(?:times?|x|episodes?|seizures?|sz)?\s*"
                r"(?:of seizure)?\s*(?:/\s*|per\s*)month\b"
            ),
            "monthly_count",
        ),
        (
            re.compile(
                r"(?i)<\s*(?P<count>\d+(?:\.\d+)?)\s*/\s*(?P<months>\d+(?:\.\d+)?)\s*(?:m\b|months?\b)"
            ),
            "upper_bound_per_n_months",
        ),
        (
            re.compile(r"(?i)(?P<count>\d+(?:\.\d+)?)\s*/\s*(?P<months>\d+(?:\.\d+)?)\s*(?:m\b|months?\b)"),
            "count_per_n_months",
        ),
    ]
    for regex, method in patterns:
        for match in regex.finditer(text):
            if not count_match_is_valid(text, match):
                continue
            if method == "count_range_per_n_months":
                low_count = float(match.group("min"))
                high_count = float(match.group("max"))
                low_months = float(match.group("months_min"))
                high_months = float(match.group("months_max"))
                if low_count > high_count or low_months <= 0 or high_months <= 0:
                    continue
                monthly_low = low_count / max(low_months, high_months)
                monthly_high = high_count / min(low_months, high_months)
                return monthly_low, monthly_high, (monthly_low + monthly_high) / 2, method
            if method == "monthly_range":
                low = float(match.group("min"))
                high = float(match.group("max"))
                if low > high:
                    continue
                return low, high, (low + high) / 2, method
            if method == "monthly_lower_bound":
                count = float(match.group("count"))
                return count, count, count, method
            if method == "monthly_count":
                count = float(match.group("count"))
                return count, count, count, method
            count = float(match.group("count"))
            months = float(match.group("months"))
            if months > 0:
                monthly = count / months
                if method == "upper_bound_per_n_months":
                    return 0.0, monthly, monthly, method
                return monthly, monthly, monthly, method
    return None


def explicit_count_patterns(text):
    counts = []

    for match in re.finditer(
        r"(?i)\b(?:seizures?|sz)\s+(?:noted\s+)?(?:for\s+)?(?P<count>\d+(?:\.\d+)?)\s+times?\b",
        text,
    ):
        count = float(match.group("count"))
        if count_value_is_plausible(count) and count_match_is_valid(text, match):
            counts.append((count, match.group(0)))
    for match in re.finditer(r"(?i)\bx\s*(?P<count>\d{1,3})(?=(?:19|20)\d{2}\b)", text):
        count = float(match.group("count"))
        if count_value_is_plausible(count) and count_match_is_valid(text, match):
            counts.append((count, f"x{match.group('count')}"))
    for match in re.finditer(r"(?i)(?<![\w.])(?P<count>\d+(?:\.\d+)?)\s*x\b", text):
        count = float(match.group("count"))
        if count_value_is_plausible(count) and count_match_is_valid(text, match):
            counts.append((count, match.group(0)))
    for match in re.finditer(r"(?i)\bx\s*(?P<count>\d+(?:\.\d+)?)\b", text):
        count = float(match.group("count"))
        if count_value_is_plausible(count) and count_match_is_valid(text, match):
            counts.append((count, match.group(0)))
    for match in re.finditer(
        r"(?i)\b(?P<count>one|once|two|twice|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:seizure|seizures|episode|episodes|event|events|gtc|cps|sps)\b",
        text,
    ):
        count = number_word_to_int(match.group("count"))
        if count is not None and count_match_is_valid(text, match):
            counts.append((float(count), match.group(0)))
    for match in re.finditer(r"(?i)\b(?P<count>\d+)\s*(?:seizures|episodes|events)\b", text):
        count = float(match.group("count"))
        if count_value_is_plausible(count) and count_match_is_valid(text, match):
            counts.append((count, match.group(0)))
    return counts


def month_specific_counts(segment_text, default_month):
    results = []

    for match in re.finditer(
        r"(?i)(?P<year>(?:19|20)\d{2})[./\-\s]*(?P<month_name>"
        + "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
        + r")\.?\s*(?P<count>x\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*x|one seizure|once|two seizures|twice)",
        segment_text,
    ):
        month = month_key(match.group("year"), MONTH_NAMES[match.group("month_name").lower()])
        count = explicit_count_patterns(match.group("count"))
        if month and count:
            results.append((month, count[0][0], match.group(0)))

    default_year = int(default_month[:4]) if default_month else None
    if default_year:
        month_names = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
        for match in re.finditer(
            r"(?i)\b(?P<month_name>"
            + month_names
            + r")\.?\s*(?P<count>x\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*x|one seizure|once|two seizures|twice)",
            segment_text,
        ):
            month = month_key(default_year, MONTH_NAMES[match.group("month_name").lower()])
            count = explicit_count_patterns(match.group("count"))
            if month and count:
                results.append((month, count[0][0], match.group(0)))

        for match in re.finditer(
            r"(?<!\d)(?P<month>\d{1,2})/(?P<day>\d{1,2})\s*[:;,]?\s*"
            r"(?P<count>x\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*x|one seizure|once|two seizures|twice)?",
            segment_text,
        ):
            month = month_key(default_year, match.group("month"))
            count_text = match.group("count")
            if month and count_text:
                count = explicit_count_patterns(count_text)
                if count:
                    results.append((month, count[0][0], match.group(0)))

    return results


def infer_single_event_legacy(text):
    if NO_SEIZURE_RE.search(text):
        return False
    if re.search(r"(?i)\b(gtc|cps|sps|convulsive seizure|generalized convulsion|aura)\b", text):
        return True
    if re.search(r"發作|抽蓄|抽搐|臉部扭曲|vocalization", text):
        return True
    return False


def infer_single_event(text):
    if NO_SEIZURE_RE.search(text):
        return False
    if STATIONARY_RE.search(text) and not parse_frequency_count(text):
        return False
    if HISTORICAL_COUNT_CONTEXT_RE.search(text) and not re.search(r"(?i)\b(recur|recurrent|visited|er|today|this month)\b", text):
        return False
    if re.search(r"(?i)\b(gtc|cps|sps|convulsive seizure|generalized convulsion|aura)\b", text):
        return True
    if SEIZURE_KEYWORDS.search(text):
        return True
    return False


def extract_counts_from_segment(segment):
    text = segment["text"]
    month = segment["month"]
    evidence = []
    issues = []

    specific_counts = month_specific_counts(text, month)
    for specific_month, count, snippet in specific_counts:
        evidence.append((specific_month, count, count, count, "high", "month_specific_count", snippet))

    frequency = parse_frequency_count(text)
    if frequency and not specific_counts:
        low, high, estimate, method = frequency
        confidence = "medium" if method != "upper_bound_per_n_months" else "low"
        evidence.append((month, low, high, estimate, confidence, method, text))
        if method in {"upper_bound_per_n_months", "count_per_n_months", "count_range_per_n_months"}:
            issues.append((month, "period_frequency", "review", f"Converted period frequency with method={method}.", text))

    counts = explicit_count_patterns(text)
    if counts and not frequency and not specific_counts:
        total = sum(count for count, _ in counts)
        evidence.append((month, total, total, total, "high", "explicit_count_sum", text))

    if NO_SEIZURE_RE.search(text) and not evidence:
        evidence.append((month, 0.0, 0.0, 0.0, "high", "explicit_no_seizure", text))
    elif NO_SEIZURE_RE.search(text) and evidence:
        issues.append((month, "mixed_no_seizure_and_count", "review", "Segment contains both no-seizure language and a count.", text))

    if not evidence and infer_single_event(text):
        evidence.append((month, 1.0, 1.0, 1.0, "low", "inferred_single_event", text))
        issues.append((month, "inferred_count", "review", "Seizure event was present but no explicit count was found; inferred 1.", text))

    if not evidence and SEIZURE_KEYWORDS.search(text):
        issues.append((month, "uninterpretable_seizure_text", "review", "Seizure-related text had no interpretable count.", text))
    elif not evidence and STATIONARY_RE.search(text):
        issues.append((month, "stationary_without_count", "info", "Stationary note does not prove seizure count.", text))

    return evidence, issues


def add_review(issues, patient_id, cohort, source_file, month, issue_type, severity, reason, evidence):
    issues.append(
        ReviewIssue(
            patient_id=patient_id,
            cohort=cohort,
            source_file=source_file,
            month=month or "",
            issue_type=issue_type,
            severity=severity,
            reason=reason,
            evidence=clean_snippet(evidence, 500),
        )
    )


def extract_patient(path):
    path = Path(path)
    sections = extract_sections(path)
    patient_id = patient_id_from_path(path, sections)
    cohort = cohort_from_path(path)
    source_file = str(path)
    evidence_rows = []
    review_rows = []

    soap = sections.get("asm_soap", "")
    if not soap:
        add_review(
            review_rows,
            patient_id,
            cohort,
            source_file,
            "",
            "missing_soap_section",
            "error",
            "Could not find ASM狀況 -> 原因 SOAP【P/S】 content.",
            "",
        )
        return evidence_rows, review_rows

    segments = split_into_dated_segments(soap)
    if not segments:
        add_review(
            review_rows,
            patient_id,
            cohort,
            source_file,
            "",
            "no_dated_segments",
            "error",
            "SOAP content had no parseable absolute dates.",
            soap,
        )

    for segment in segments:
        segment_evidence, segment_issues = extract_counts_from_segment(segment)
        for month, low, high, estimate, confidence, method, text in segment_evidence:
            evidence_rows.append(
                Evidence(
                    patient_id=patient_id,
                    cohort=cohort,
                    source_file=source_file,
                    month=month,
                    count_min=low,
                    count_max=high,
                    count_estimate=estimate,
                    confidence=confidence,
                    source_section="ASM狀況/原因 SOAP",
                    method=method,
                    evidence=clean_snippet(text),
                )
            )
        for month, issue_type, severity, reason, text in segment_issues:
            add_review(review_rows, patient_id, cohort, source_file, month, issue_type, severity, reason, text)

    if sections.get("semiology_frequency"):
        freq = sections["semiology_frequency"]
        parsed_freq = parse_frequency_count(freq)
        if parsed_freq:
            add_review(
                review_rows,
                patient_id,
                cohort,
                source_file,
                "",
                "undated_semiology_frequency",
                "info",
                "Semiology Frequency has useful undated frequency text; not added to monthly timeline.",
                freq,
            )

    if sections.get("asm_plan") and SEIZURE_KEYWORDS.search(sections["asm_plan"]):
        add_review(
            review_rows,
            patient_id,
            cohort,
            source_file,
            "",
            "supplementary_asm_plan_text",
            "info",
            "診斷計畫 contains seizure-related treatment/frequency text; review if SOAP extraction is incomplete.",
            sections["asm_plan"],
        )

    if evidence_rows:
        by_month = defaultdict(list)
        for row in evidence_rows:
            by_month[row.month].append(row)
        for month, rows in by_month.items():
            has_zero = any(row.count_estimate == 0 for row in rows)
            has_positive = any(row.count_estimate > 0 for row in rows)
            if has_zero and has_positive:
                add_review(
                    review_rows,
                    patient_id,
                    cohort,
                    source_file,
                    month,
                    "conflicting_month_evidence",
                    "review",
                    "Same month contains both zero and positive seizure evidence.",
                    " || ".join(row.evidence for row in rows[:4]),
                )
    else:
        add_review(
            review_rows,
            patient_id,
            cohort,
            source_file,
            "",
            "no_monthly_counts_extracted",
            "review",
            "No monthly seizure counts could be extracted from SOAP.",
            soap,
        )

    return evidence_rows, review_rows


def aggregate_evidence(evidence_rows):
    grouped = defaultdict(list)
    for row in evidence_rows:
        grouped[(row.patient_id, row.cohort, row.source_file, row.month)].append(row)

    output = []
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    frequency_methods = {
        "monthly_range",
        "monthly_count",
        "monthly_lower_bound",
        "count_per_n_months",
        "count_range_per_n_months",
        "upper_bound_per_n_months",
    }
    for (patient_id, cohort, source_file, month), rows in sorted(grouped.items()):
        aggregate_rows = [row for row in rows if row.method in frequency_methods]
        if aggregate_rows:
            # A frequency phrase is already a month-level estimate, not an event to add
            # on top of other snippets in the same month.
            adjusted_rows = [
                (row.count_min, row.count_min, row.count_min)
                if row.method == "monthly_range"
                else (row.count_min, row.count_max, row.count_estimate)
                for row in aggregate_rows
            ]
            total_min = max(row[0] for row in adjusted_rows)
            total_max = max(row[1] for row in adjusted_rows)
            total_estimate = max(row[2] for row in adjusted_rows)
        else:
            aggregate_rows = rows
            total_min = sum(row.count_min for row in aggregate_rows)
            total_max = sum(row.count_max for row in aggregate_rows)
            total_estimate = sum(row.count_estimate for row in aggregate_rows)
        confidence = min(aggregate_rows, key=lambda row: confidence_rank.get(row.confidence, 0)).confidence
        output.append(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "source_file": source_file,
                "month": month,
                "seizure_count_min": format_number(total_min),
                "seizure_count_max": format_number(total_max),
                "seizure_count_estimate": format_number(total_estimate),
                "confidence": confidence,
                "evidence_count": len(rows),
                "methods": "; ".join(sorted({row.method for row in rows})),
                "evidence": " || ".join(row.evidence for row in rows[:5]),
            }
        )
    return output


def month_to_ordinal(month):
    try:
        year, month_number = month.split("-", 1)
        return int(year) * 12 + int(month_number) - 1
    except (AttributeError, TypeError, ValueError):
        return None


def ordinal_to_month(ordinal):
    year, zero_based_month = divmod(ordinal, 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def apply_monthly_range_carry_forward(monthly_rows):
    grouped = defaultdict(list)
    for row in monthly_rows:
        grouped[(row["patient_id"], row["cohort"], row["source_file"])].append(row)

    output = []
    for _group_key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (month_to_ordinal(row["month"]) is None, row["month"]))
        output.extend(rows)

        for index, row in enumerate(rows[:-1]):
            current_ordinal = month_to_ordinal(row["month"])
            next_ordinal = month_to_ordinal(rows[index + 1]["month"])
            if current_ordinal is None or next_ordinal is None:
                continue
            if "monthly_range" not in {method.strip() for method in row["methods"].split(";")}:
                continue
            if next_ordinal <= current_ordinal + 1:
                continue

            lower_bound = row["seizure_count_min"]
            fill_until = min(next_ordinal, current_ordinal + 1 + MAX_MONTHLY_RANGE_CARRY_FORWARD_MONTHS)
            for ordinal in range(current_ordinal + 1, fill_until):
                filled = dict(row)
                filled["month"] = ordinal_to_month(ordinal)
                filled["seizure_count_min"] = lower_bound
                filled["seizure_count_max"] = lower_bound
                filled["seizure_count_estimate"] = lower_bound
                filled["evidence_count"] = 0
                filled["methods"] = "monthly_range_lower_bound_carry_forward"
                filled["evidence"] = (
                    f"Carried lower bound from monthly_range at {row['month']} "
                    f"(capped at {MAX_MONTHLY_RANGE_CARRY_FORWARD_MONTHS} months): {row['evidence']}"
                )
                output.append(filled)

    return sorted(output, key=lambda row: (row["cohort"], row["patient_id"], row["source_file"], row["month"]))


def format_number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = round(float(value), 3)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def collect_docx_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        folder = Path(input_dir)
        if not folder.exists():
            continue
        files.extend(sorted(folder.glob("*.docx")))
    return files


def rows_to_dataframe(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    for col in ("seizure_count_min", "seizure_count_max", "seizure_count_estimate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Date").sort_index()
    return df


def extract_seizure_timeline(filepath):
    evidence_rows, _ = extract_patient(filepath)
    return rows_to_dataframe(apply_monthly_range_carry_forward(aggregate_evidence(evidence_rows)))


def table_selection(filepath):
    return extract_seizure_timeline(filepath)


def process_directory(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    docx_files = sorted(Path(input_dir).glob("*.docx"))
    success, skipped, errors = [], [], []
    for filepath in docx_files:
        try:
            df = extract_seizure_timeline(filepath)
        except ImportError as exc:
            raise ImportError("pandas is required for DataFrame output. Activate the project venv first.") from exc
        except Exception as exc:
            errors.append((filepath.name, str(exc)))
            print(f"  ERROR {filepath.name}: {exc}")
            continue
        if df.empty:
            skipped.append(filepath.name)
            print(f"  SKIP  {filepath.name}: no seizure timeline rows")
            continue
        out_path = os.path.join(output_dir, filepath.name.replace(".docx", "_seizure_timeline.csv"))
        df.to_csv(out_path, index=True, encoding="utf-8-sig")
        success.append(filepath.name)
        print(f"  OK    {filepath.name} -> {out_path}")

    print(f"\nDone: {len(success)} succeeded, {len(skipped)} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for fname, err in errors:
            print(f"  {fname}: {err}")

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(value):
    value = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("_")
    return value or "unknown"


def write_patient_timeline_csvs(monthly_rows, output_dir):
    patient_dir = os.path.join(output_dir, "seizure_timeline")
    grouped = defaultdict(list)
    for row in monthly_rows:
        grouped[row["source_file"]].append(row)

    paths = {}
    fieldnames = [
        "patient_id",
        "cohort",
        "source_file",
        "month",
        "seizure_count_min",
        "seizure_count_max",
        "seizure_count_estimate",
        "confidence",
        "evidence_count",
        "methods",
        "evidence",
    ]
    for source_file, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["month"])
        first = rows[0]
        stem = Path(source_file).stem
        filename = (
            f"{safe_filename(first['patient_id'])}{safe_filename(first['cohort'])}_seizure_timeline.csv"
        )
        out_path = os.path.join(patient_dir, filename)
        write_csv(out_path, rows, fieldnames)
        paths[source_file] = out_path
    return paths


def build_summary_rows(files, monthly_rows, evidence_rows, review_rows, patient_paths):
    by_source = {}
    for path in files:
        source_file = str(path)
        patient_match = re.search(r"\d+", Path(path).stem)
        by_source[source_file] = {
            "patient_id": patient_match.group(0) if patient_match else Path(path).stem,
            "cohort": cohort_from_path(path),
            "source_file": source_file,
            "patient_timeline_csv": patient_paths.get(source_file, ""),
            "monthly_rows": 0,
            "evidence_rows": 0,
            "human_review_rows": 0,
        }

    for row in monthly_rows:
        summary = by_source.setdefault(
            row["source_file"],
            {
                "patient_id": row["patient_id"],
                "cohort": row["cohort"],
                "source_file": row["source_file"],
                "patient_timeline_csv": patient_paths.get(row["source_file"], ""),
                "monthly_rows": 0,
                "evidence_rows": 0,
                "human_review_rows": 0,
            },
        )
        summary["monthly_rows"] += 1
        summary["patient_id"] = row["patient_id"]
        summary["cohort"] = row["cohort"]

    for row in evidence_rows:
        if row["source_file"] in by_source:
            by_source[row["source_file"]]["evidence_rows"] += 1

    for row in review_rows:
        if row["source_file"] in by_source:
            by_source[row["source_file"]]["human_review_rows"] += 1

    summary_rows = []
    for row in sorted(by_source.values(), key=lambda item: (item["cohort"], item["patient_id"], item["source_file"])):
        summary_rows.append(row)
    return summary_rows


def process_directories(input_dirs=DEFAULT_INPUT_DIRS, output_dir=DEFAULT_OUTPUT_DIR):
    files = collect_docx_files(input_dirs)
    all_evidence = []
    all_review = []

    for path in files:
        try:
            evidence, review = extract_patient(path)
        except Exception as exc:
            patient_id = re.search(r"\d+", path.stem)
            all_review.append(
                ReviewIssue(
                    patient_id=patient_id.group(0) if patient_id else path.stem,
                    cohort=cohort_from_path(path),
                    source_file=str(path),
                    month="",
                    issue_type="parser_error",
                    severity="error",
                    reason=str(exc),
                    evidence="",
                )
            )
            continue
        all_evidence.extend(evidence)
        all_review.extend(review)

    monthly_rows = apply_monthly_range_carry_forward(aggregate_evidence(all_evidence))
    evidence_rows = [
        {
            "patient_id": row.patient_id,
            "cohort": row.cohort,
            "source_file": row.source_file,
            "month": row.month,
            "count_min": format_number(row.count_min),
            "count_max": format_number(row.count_max),
            "count_estimate": format_number(row.count_estimate),
            "confidence": row.confidence,
            "source_section": row.source_section,
            "method": row.method,
            "evidence": row.evidence,
        }
        for row in all_evidence
    ]
    review_rows = [
        {
            "patient_id": row.patient_id,
            "cohort": row.cohort,
            "source_file": row.source_file,
            "month": row.month,
            "issue_type": row.issue_type,
            "severity": row.severity,
            "reason": row.reason,
            "evidence": row.evidence,
        }
        for row in all_review
    ]

    patient_paths = write_patient_timeline_csvs(monthly_rows, output_dir)
    patient_dir = os.path.join(output_dir, "seizure_timeline")
    evidence_path = os.path.join(output_dir, "seizure_extraction_evidence.csv")
    review_path = os.path.join(output_dir, "seizure_human_review.csv")
    summary_path = os.path.join(output_dir, "seizure_execution_summary.csv")

    write_csv(
        evidence_path,
        evidence_rows,
        [
            "patient_id",
            "cohort",
            "source_file",
            "month",
            "count_min",
            "count_max",
            "count_estimate",
            "confidence",
            "source_section",
            "method",
            "evidence",
        ],
    )
    write_csv(
        review_path,
        review_rows,
        ["patient_id", "cohort", "source_file", "month", "issue_type", "severity", "reason", "evidence"],
    )
    summary_rows = build_summary_rows(
        files,
        monthly_rows,
        evidence_rows,
        review_rows,
        patient_paths,
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "patient_id",
            "cohort",
            "source_file",
            "patient_timeline_csv",
            "monthly_rows",
            "evidence_rows",
            "human_review_rows",
        ],
    )

    return {
        "files": len(files),
        "evidence_rows": len(evidence_rows),
        "monthly_rows": len(monthly_rows),
        "review_rows": len(review_rows),
        "patient_timeline_files": len(patient_paths),
        "patient_timeline_dir": patient_dir,
        "evidence_path": evidence_path,
        "review_path": review_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract monthly seizure attack counts from DOCX 原因 SOAP sections."
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        default=list(DEFAULT_INPUT_DIRS),
        help="Directories containing patient .docx files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-patient seizure timelines and review reports.",
    )
    args = parser.parse_args()

    result = process_directories(args.input_dirs, args.output_dir)
    print(
        "Processed {files} DOCX files; wrote {patient_timeline_files} patient timeline CSVs, "
        "{monthly_rows} monthly rows, {evidence_rows} evidence rows, and "
        "{review_rows} human-review rows.".format(**result)
    )
    print(f"Patient timelines: {result['patient_timeline_dir']}")
    print(f"Execution summary: {result['summary_path']}")
    print(f"Evidence: {result['evidence_path']}")
    print(f"Human review: {result['review_path']}")


if __name__ == "__main__":
    main()
