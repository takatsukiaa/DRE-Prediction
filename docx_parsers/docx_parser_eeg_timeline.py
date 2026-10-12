import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from docx import Document


DEFAULT_INPUT_DIRS = ("DRE", "NDRE", "DRE_batch2", "NDRE_batch2")
DEFAULT_OUTPUT_DIR = "new_csvs"
EEG_TABLE_TITLE = "EEG"

ED_FEATURES = [
    "Spike ED",
    "Sharp ED",
    "Sharpish configuration",
    "Spike-and-wave ED",
    "Periodic Discharge",
]
ED_LOCATION_ROWS = [
    "TLED",
    "TRED",
    "FLED",
    "FRED",
    "OLED",
    "ORED",
    "PLED",
    "PRED",
    "CLED",
    "CRED",
]
SLOW_FEATURES = ["Delta waves", "Theta waves"]
SLOW_LOCATION_ROWS = ["TL", "TR", "FL", "FR", "OL", "OR", "PL", "PR", "CL", "CR"]
EEG_FEATURES = ED_FEATURES + ED_LOCATION_ROWS + SLOW_FEATURES + SLOW_LOCATION_ROWS
TIMELINE_FIELDNAMES = [
    "patient_id",
    "cohort",
    "source_file",
    "report_date",
    "month",
] + EEG_FEATURES + [
    "confidence",
    "evidence_count",
    "methods",
    "evidence",
]

REGION_CODES = {
    "temporal": "T",
    "frontal": "F",
    "occipital": "O",
    "parietal": "P",
    "central": "C",
}
CODE_TO_REGION = {value: key for key, value in REGION_CODES.items()}
SIDE_CODES = {"left": "L", "right": "R"}
LOCATION_WORDS = set(REGION_CODES) | {
    "frontotemporal",
    "fronto-temporal",
    "frontal-temporal",
    "temporofrontal",
    "temporo-frontal",
    "parieto-occipital",
    "parietal-occipital",
    "occipito-parietal",
    "posterior temporal-parietal",
    "frontocentral",
    "fronto-central",
    "centrotemporal",
    "centro-temporal",
    "mesial temporal",
    "anterior temporal",
    "mid-temporal",
}

ABS_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})[./-](?P<month>\d{1,2})[./-](?P<day>\d{1,2})(?!\d)"
)
COMPACT_DATE_RE = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<month>\d{2})(?P<day>\d{2})(?!\d)")

ELECTRODE_RE = re.compile(
    r"(?i)(?:fpz|fcz|cz|pz|oz|fp1|fp2|sp1|sp2|s1|s2|f[3478z]|t[3456]|c[34z]|p[34z]|o[12z])"
)
MIDLINE_ELECTRODE_RE = re.compile(r"(?i)\b(?:fz|cz|pz|oz|fpz|fcz)\b")

NORMAL_SLEEP_RE = re.compile(
    r"(?i)\b(vertex sharp waves?|sleep spindles?|k[- ]?complex(?:es)?|alpha dropout)\b"
)
NEGATION_RE = re.compile(
    r"(?i)\b("
    r"no\b(?:(?!\bbut\b|[.;]).){0,80}\b(?:spikes?|sharp waves?|epileptiform discharges?|focal slow waves?|slow waves?)|"
    r"without (?:definite )?(?:spikes?|sharp waves?|epileptiform discharges?|focal slow waves?)|"
    r"we do not observe (?:sharp waves?|spikes?)"
    r")\b"
)
ED_NEGATION_RE = re.compile(
    r"(?i)\b("
    r"neither (?:spikes?|sharp waves?) nor (?:spikes?|sharp waves?)|"
    r"no\b(?!\s+(?:activation|build[- ]?up|buildup)\b)(?:(?!\bbut\b|[.;]).){0,80}\b(?:spikes?|sharp waves?|epileptiform discharges?)|"
    r"without (?:definite )?(?:spikes?|sharp waves?|epileptiform discharges?)|"
    r"we do not observe (?:sharp waves?|spikes?)"
    r")\b"
)
SLOW_NEGATION_RE = re.compile(r"(?i)\b(no\b[^.;]{0,80}\b(?:focal slow waves?|slow waves?))\b")
SECTION_HEADING_RE = re.compile(
    r"(?i)\b(?:EEG diagnosis|Description|Abnormal findings?|Epileptiform discharge|Interpretation|Conclusion|"
    r"Awake stage|Sleep stage|Photic stimulation|Hyperventilation|Background)\s*:?"
)
FINAL_VERDICT_HEADING_RE = re.compile(
    r"(?i)\b(?:EEG diagnosis|Description|Abnormal findings?|Interpretation|Conclusion|"
    r"Awake stage|Sleep stage|Photic stimulation|Hyperventilation|Background)\s*:?"
)
FINAL_NEGATIVE_ED_RE = re.compile(
    r"(?i)\b(?:"
    r"negative study|"
    r"no evidence of (?:(?:definite|obvious|prominent)\s+)?(?:epileptiform|epileptic) (?:discharges?|activity)|"
    r"no\b(?:(?!\b(?:but|except|however)\b|[.;]).){0,80}\b"
    r"(?:(?:definite|obvious|prominent|sustained)\s+)?(?:spikes?|sharp waves?|epileptiform discharges?)|"
    r"without (?:(?:definite|obvious|prominent)\s+)?(?:spikes?|sharp waves?|epileptiform discharges?)|"
    r"we do not observe (?:sharp waves?|spikes?)(?:\s+or\s+(?:sharp waves?|spikes?))?"
    r")\b"
)
FINAL_NEGATIVE_SLOW_RE = re.compile(
    r"(?i)\b(?:"
    r"negative study|"
    r"no evidence of (?:(?:definite|obvious|prominent)\s+)?(?:slow[- ]?waves?|slowings?|cortical dysfunction)|"
    r"no\b(?:(?!\b(?:but|except|however)\b|[.;]).){0,80}\b"
    r"(?:(?:focal|regional|diffuse|definite|obvious|prominent)\s+)?(?:slow[- ]?waves?|slowings?|cortical dysfunction)|"
    r"without (?:(?:focal|regional|diffuse|definite|obvious|prominent)\s+)?"
    r"(?:slow[- ]?waves?|slowings?|cortical dysfunction)"
    r")\b"
)

SPIKE_WAVE_RE = re.compile(
    r"(?i)\b(?:"
    r"spikes?\s*(?:-|and\s+)?(?:slow|waves?|wave)\s*(?:complex(?:es)?)?|"
    r"spike[- ]waves?\s*(?:complex(?:es)?)?|"
    r"spike[- ]and[- ]wave(?:s)?|"
    r"spike[- ]and[- ]slow(?: waves?)?|"
    r"polyspike[- ]wave(?:s)?|"
    r"sharp\s+and\s+waves?\b|"
    r"sharp[- ]and[- ]waves?\b"
    r")"
)
STANDALONE_SPIKE_RE = re.compile(r"(?i)\b(?:polyspikes?|spikes?)\b")
SHARP_RE = re.compile(
    r"(?i)\b(?:sharp waves?|sharp[- ]and[- ]slow(?: waves?| complexes?)?|"
    r"sharp[- ]slow(?: waves?| complexes?)?|sharp[- ]wave(?:s)?\s*(?:complex(?:es)?)?)\b"
)
SHARPISH_RE = re.compile(r"(?i)\b(?:sharpish|sharply[- ]contoured|sharp[- ]contoured)\b")
PERIODIC_RE = re.compile(r"(?i)\bperiodic discharge\b")
THETA_RE = re.compile(r"(?i)\btheta\b")
DELTA_RE = re.compile(r"(?i)\bdelta\b")
SLOW_WAVE_RE = re.compile(r"(?i)\b(?:slow waves?|background slow(?:ing)?|diffuse background slow(?:ing)?)\b")
HYPEREXCITABILITY_RE = re.compile(r"(?i)\b(?:hyperexcitability|hyperexcitablity|increased excitability|epileptiform)\b")
DIFFUSE_BACKGROUND_RE = re.compile(r"(?i)\b(?:diffuse background slow|background slow(?:ing)?|diffuse theta to delta)\b")
NON_EEG_NOTICE_TEXTS = (
    "\u4f4f\u9662\u671f\u9593\u6709\u6aa2\u9a57\u5831\u544a",
    "\u6025\u8a3a\u671f\u9593\u6709\u6aa2\u9a57\u5831\u544a",
)
NORMAL_RECORD_SECTION_RE = re.compile(
    r"(?i)\b(?:eeg\s*diagnosis|\[eeg\s*diagnosis\]|diagnosis)\s*:?\s*(?:essentially\s+|probably\s+)?normal\b"
)
PROBABLY_NORMAL_RE = re.compile(r"(?i)\bprobably\s+normal\b")

DOMINANCE_RE = re.compile(
    r"(?i)\b(?:main focus|maximum|originat(?:ed|ing)|initiated from|highest amplitude)\b"
)
LEFT_DOMINANCE_RE = re.compile(
    r"(?i)\b(?:emphasis|especially|worse|prominent|more active|more prominent|main focus|maximum|"
    r"originat(?:ed|ing)|highest amplitude)[^.;]{0,80}\bleft\b|"
    r"\bleft\b[^.;]{0,80}\b(?:worse|emphasis|prominent|more active|more prominent|main focus|maximum|originat(?:ed|ing)|highest amplitude)\b"
)
RIGHT_DOMINANCE_RE = re.compile(
    r"(?i)\b(?:emphasis|especially|worse|prominent|more active|more prominent|main focus|maximum|"
    r"originat(?:ed|ing)|highest amplitude)[^.;]{0,80}\bright\b|"
    r"\bright\b[^.;]{0,80}\b(?:worse|emphasis|prominent|more active|more prominent|main focus|maximum|originat(?:ed|ing)|highest amplitude)\b"
)


@dataclass
class EegReport:
    patient_id: str
    cohort: str
    source_file: str
    report_date: str
    month: str
    row_index: int
    text: str


@dataclass
class EegEvidence:
    patient_id: str
    cohort: str
    source_file: str
    report_date: str
    month: str
    feature: str
    value: float
    location_rows: str
    confidence: str
    method: str
    descriptor: str
    evidence: str


@dataclass
class EegReviewIssue:
    patient_id: str
    cohort: str
    source_file: str
    report_date: str
    month: str
    issue_type: str
    severity: str
    reason: str
    evidence: str


def normalize_text(text):
    text = str(text or "").replace("\xa0", " ")
    text = text.replace("θ", "theta").replace("Θ", "theta")
    text = text.replace("δ", "delta").replace("Δ", "delta")
    text = text.replace("α", "alpha").replace("Α", "alpha")
    text = text.replace("ã€œ", "~").replace("â†’", "->")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parser_text(text):
    text = normalize_text(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("spike and slow", "spike-and-slow")
    text = text.replace("spike and wave", "spike-and-wave")
    text = text.replace("sharp and slow", "sharp-and-slow")
    text = text.replace("sharp and wave", "sharp-and-wave")
    text = re.sub(r"\btheta\s*(?:-|to|or)\s*delta\b", "theta-to-delta", text)
    text = re.sub(r"\bdelta\s*(?:-|to|or)\s*theta\b", "delta-to-theta", text)
    return re.sub(r"\s+", " ", text).strip()


def has_plain_spike(text):
    """Return true for spike/polyspike mentions outside spike-wave phrases."""
    without_spike_wave = SPIKE_WAVE_RE.sub(" ", text)
    return bool(STANDALONE_SPIKE_RE.search(without_spike_wave))


def get_cell_text(cell):
    clean_text = normalize_text(cell.text.replace("\n", " "))
    all_text_nodes = cell._tc.xpath(".//w:t")
    xpath_text = normalize_text("".join(node.text for node in all_text_nodes if node.text))
    if len(xpath_text) > len(clean_text):
        return xpath_text
    return clean_text


def get_unique_row_values(row):
    row_data = []
    visited_cells = set()
    for cell in row.cells:
        if cell._tc in visited_cells:
            continue
        visited_cells.add(cell._tc)
        row_data.append(get_cell_text(cell))
    return row_data


def find_eeg_table(doc):
    for table in doc.tables:
        if not table.rows:
            continue
        first_row = get_unique_row_values(table.rows[0])
        if first_row and first_row[0] == EEG_TABLE_TITLE:
            return table
    return None


def parse_date(value):
    value = normalize_text(value)
    match = ABS_DATE_RE.search(value)
    if not match:
        match = COMPACT_DATE_RE.search(value)
    if not match:
        return None, None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None, None
    return f"{year:04d}-{month:02d}-{day:02d}", f"{year:04d}-{month:02d}"


def collect_docx_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        path = Path(input_dir)
        if path.is_file() and path.suffix.lower() == ".docx" and not path.name.startswith("~$"):
            files.append(path)
        elif path.exists():
            files.extend(sorted(p for p in path.glob("*.docx") if not p.name.startswith("~$")))
    return sorted(files, key=lambda p: str(p).lower())


def patient_id_from_path(path):
    match = re.search(r"\d+", Path(path).stem)
    return match.group(0) if match else Path(path).stem


def cohort_from_path(path):
    parts = [part.upper() for part in Path(path).parts]
    if any("NDRE" in part or part.startswith("NS") for part in parts):
        return "NDRE"
    if any("DRE" in part for part in parts):
        return "DRE"
    return Path(path).parent.name or "unknown"


def safe_filename(value):
    value = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("_")
    return value or "unknown"


def is_non_eeg_notice(text):
    compact = re.sub(r"\s+", "", normalize_text(text))
    return any(notice in compact for notice in NON_EEG_NOTICE_TEXTS)


def is_normal_eeg_record(text):
    normalized = parser_text(text)
    heading = re.split(
        r"(?i)\b(?:description|abnormal findings?|interpretation|conclusion|awake stage|sleep stage|background)\b",
        normalized,
        maxsplit=1,
    )[0]
    return bool(NORMAL_RECORD_SECTION_RE.search(heading))


def should_split_embedded_date(text, match):
    following = text[match.start() : match.start() + 120].lower()
    preceding = text[max(0, match.start() - 40) : match.start()].lower()
    return (
        "eeg" in following
        or "awake stage" in following
        or "clinical dx" in following
        or "description" in following
        or "interpretation" in following
        or "eeg" in preceding
    )


def split_embedded_reports(report_date, month, text):
    matches = [match for match in ABS_DATE_RE.finditer(text) if should_split_embedded_date(text, match)]
    if not matches:
        return [(report_date, month, text)], 0

    reports = []
    first_start = matches[0].start()
    first_text = text[:first_start].strip(" |;")
    if first_text:
        reports.append((report_date, month, first_text))

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        embedded_date, embedded_month = parse_date(match.group(0))
        embedded_text = text[match.end() : next_start].strip(" |;")
        if embedded_date and embedded_text:
            reports.append((embedded_date, embedded_month, embedded_text))

    return reports, max(0, len(reports) - 1)


def extract_reports(path):
    patient_id = patient_id_from_path(path)
    cohort = cohort_from_path(path)
    source_file = str(path)
    reviews = []
    reports = []

    doc = Document(str(path))
    table = find_eeg_table(doc)
    if table is None:
        reviews.append(
            EegReviewIssue(
                patient_id,
                cohort,
                source_file,
                "",
                "",
                "missing_eeg_table",
                "review",
                "No EEG table with first cell 'EEG' was found.",
                "",
            )
        )
        return reports, reviews

    for row_index, row in enumerate(table.rows[1:], start=1):
        values = get_unique_row_values(row)
        if not values or not any(value.strip() for value in values):
            continue
        raw_row_text = normalize_text(" | ".join(values))
        if is_non_eeg_notice(raw_row_text):
            continue
        date_text = values[0].strip()
        if date_text.casefold() == "date":
            continue
        report_date, month = parse_date(date_text)
        text = normalize_text(" ".join(value for value in values[1:] if value))
        if not report_date:
            if text or date_text:
                reviews.append(
                    EegReviewIssue(
                        patient_id,
                        cohort,
                        source_file,
                        "",
                        "",
                        "unparseable_date",
                        "review",
                        "EEG table row did not start with a parseable report date.",
                        normalize_text(" | ".join(values))[:500],
                    )
                )
            continue
        if not text:
            continue
        if is_normal_eeg_record(text):
            continue

        split_reports, embedded_count = split_embedded_reports(report_date, month, text)
        if embedded_count:
            reviews.append(
                EegReviewIssue(
                    patient_id,
                    cohort,
                    source_file,
                    report_date,
                    month,
                    "embedded_report_split",
                    "review",
                    f"Split one EEG table row into {len(split_reports)} report segments.",
                    text[:500],
                )
            )
        for split_date, split_month, split_text in split_reports:
            if is_non_eeg_notice(split_text) or is_normal_eeg_record(split_text):
                continue
            reports.append(EegReport(patient_id, cohort, source_file, split_date, split_month, row_index, split_text))

    return reports, reviews


def format_number(value):
    if value == "":
        return ""
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).rstrip("0").rstrip(".")


def max_value(left, right):
    if left in ("", None):
        return right
    if right in ("", None):
        return left
    return max(float(left), float(right))


def truncate_snippet(text, limit=320):
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def split_contexts(text):
    clean = normalize_text(text)
    clean = SECTION_HEADING_RE.sub(lambda match: f". {match.group(0).rstrip(':')}: ", clean)
    coarse = [part.strip(" -") for part in re.split(r"(?<=[.!?])\s+|;|\n|\r", clean) if part.strip(" -")]
    contexts = []
    for part in coarse:
        subparts = re.split(
            r",\s+(?=(?:and\s+)?(?:less|more|rare|occasional|occasionally|few|a few|some|intermittent|"
            r"frequent|frequently|abundant|continuous|many|several|sometimes|one|two|there|with|sharp|spike|theta|delta)\b)",
            part,
            flags=re.I,
        )
        for subpart in subparts:
            fragment = subpart.strip(" -")
            if fragment:
                contexts.append((part, fragment))
    return contexts


def final_verdict_block(text):
    clean = normalize_text(text)
    matches = list(FINAL_VERDICT_HEADING_RE.finditer(clean))
    verdict_starts = [
        match
        for match in matches
        if re.fullmatch(r"(?i)(?:Interpretation|Conclusion)\s*:?", match.group(0).strip())
    ]
    if not verdict_starts:
        return ""

    start_match = verdict_starts[-1]
    end = len(clean)
    for match in matches:
        if match.start() > start_match.start():
            end = match.start()
            break
    return clean[start_match.end() : end].strip()


def has_negative_ed_final_verdict(text):
    block = parser_text(final_verdict_block(text))
    if not block:
        return False
    block = NORMAL_SLEEP_RE.sub(" ", block)
    contrast = re.search(r"(?i)\b(?:but|except|however)\b", block)
    if contrast:
        tail = block[contrast.end() :]
        if has_plain_spike(tail) or SHARP_RE.search(tail) or SPIKE_WAVE_RE.search(tail) or SHARPISH_RE.search(tail):
            return False
    return bool(FINAL_NEGATIVE_ED_RE.search(block))


def has_slow_positive_marker(text):
    return bool(
        frequency_slow_features(text)
        or SLOW_WAVE_RE.search(text)
        or DIFFUSE_BACKGROUND_RE.search(text)
        or re.search(r"(?i)\b(?:slowings?|cortical dysfunction)\b", text)
    )


def has_negative_slow_final_verdict(text):
    block = parser_text(final_verdict_block(text))
    if not block:
        return False
    contrast = re.search(r"(?i)\b(?:but|except|however)\b", block)
    if contrast and has_slow_positive_marker(block[contrast.end() :]):
        return False
    return bool(FINAL_NEGATIVE_SLOW_RE.search(block))


def descriptor_score(text, feature_type):
    lowered = parser_text(text)
    descriptor = "default"
    if re.search(r"\b(?:near continuous|nearly continuous|continuous)\b", lowered):
        return 4.0, "continuous"
    if feature_type == "slow" and DIFFUSE_BACKGROUND_RE.search(lowered):
        descriptor = "diffuse/background slowing"
        if re.search(r"\b(?:severe|moderate to severe)\b", lowered):
            return 4.0, descriptor
        return 3.0, descriptor
    if "abundant" in lowered:
        return (3.5 if feature_type == "slow" else 3.0), "abundant"
    if re.search(r"\b(?:frequent|frequently|many|repetitive|active)\b", lowered):
        return 3.0, "frequent"
    if re.search(r"\bsometimes\b", lowered):
        return 2.5, "sometimes"
    if re.search(r"\b(?:intermittent|some|grouped|groups?|short[- ]?runs?|moderate amount|several|trains?)\b", lowered):
        return 2.0, "intermittent/some"
    if re.search(r"\b(?:rare|occasional|occasionally|few|a few|isolated|once|one|two|single|small amount)\b", lowered):
        return 1.0, "rare/occasional"
    return 0.5, "unspecified"


def frequency_slow_features(text):
    lowered = parser_text(text)
    slow_text = re.sub(
        r"\b(?:spike|sharp)[- ]and[- ]slow(?: waves?| wave| complexes?| complex)?\b",
        " ",
        lowered,
    )
    features = set()
    if "theta-to-delta" in slow_text or "delta-to-theta" in slow_text:
        features.update(["Theta waves", "Delta waves"])
    if THETA_RE.search(slow_text):
        features.add("Theta waves")
    if DELTA_RE.search(slow_text):
        features.add("Delta waves")
    if not features and SLOW_WAVE_RE.search(slow_text):
        for match in re.finditer(r"(?P<low>\d(?:\.\d+)?)\s*(?:-|~|to)\s*(?P<high>\d(?:\.\d+)?)\s*hz", slow_text):
            low = float(match.group("low"))
            high = float(match.group("high"))
            if low <= 4:
                features.add("Delta waves")
            if high >= 4:
                features.add("Theta waves")
        if not features:
            features.update(["Theta waves", "Delta waves"])
    return features


def is_negated(fragment, feature_type):
    if feature_type == "ed":
        return bool(ED_NEGATION_RE.search(NORMAL_SLEEP_RE.sub(" ", fragment)))
    if feature_type == "slow":
        return bool(SLOW_NEGATION_RE.search(fragment))
    return bool(NEGATION_RE.search(fragment))


def electrode_region(token):
    token = token.lower()
    if token.startswith("fp") or token.startswith("f"):
        return "F"
    if token.startswith("t") or token.startswith("s") or token.startswith("sp"):
        return "T"
    if token.startswith("c"):
        return "C"
    if token.startswith("p"):
        return "P"
    if token.startswith("o"):
        return "O"
    return None


def electrode_side(token):
    token = token.lower().replace("sp", "s")
    if token.endswith("z"):
        return None
    match = re.search(r"(\d)$", token)
    if not match:
        return None
    return "L" if int(match.group(1)) % 2 else "R"


def regions_from_words(text):
    lowered = parser_text(text)
    regions = set()
    phrase_regions = {
        "frontotemporal": {"F", "T"},
        "fronto-temporal": {"F", "T"},
        "frontal-temporal": {"F", "T"},
        "temporofrontal": {"T", "F"},
        "temporo-frontal": {"T", "F"},
        "frontocentral": {"F", "C"},
        "fronto-central": {"F", "C"},
        "centrotemporal": {"C", "T"},
        "centro-temporal": {"C", "T"},
        "parieto-occipital": {"P", "O"},
        "parietal-occipital": {"P", "O"},
        "occipito-parietal": {"O", "P"},
        "posterior temporal-parietal": {"T", "P"},
        "mesial temporal": {"T"},
        "anterior temporal": {"T"},
        "mid-temporal": {"T"},
    }
    for phrase, codes in phrase_regions.items():
        if phrase in lowered:
            regions.update(codes)
    for word, code in REGION_CODES.items():
        if re.search(rf"\b{word}\b", lowered):
            regions.add(code)
    return regions


def word_side_locations(text):
    lowered = parser_text(text)
    regions = regions_from_words(lowered)
    sides = set()
    if re.search(r"\b(?:bilateral|bil|both)\b", lowered):
        sides.update(["L", "R"])
    if re.search(r"\bleft\b", lowered):
        sides.add("L")
    if re.search(r"\bright\b", lowered):
        sides.add("R")
    return {region + side for region in regions for side in sides}


def electrode_locations(text):
    locs = set()
    midline_only = False
    tokens = [match.group(0) for match in ELECTRODE_RE.finditer(text)]
    for token in tokens:
        region = electrode_region(token)
        side = electrode_side(token)
        if region and side:
            locs.add(region + side)
    if tokens and not locs and MIDLINE_ELECTRODE_RE.search(text):
        midline_only = True
    return locs, midline_only


def extract_locations(text):
    electrode_locs, midline_only = electrode_locations(text)
    word_locs = word_side_locations(text)
    return electrode_locs | word_locs, midline_only


def locations_to_rows(locations, row_type):
    rows = set()
    suffix = "ED" if row_type == "ed" else ""
    for loc in locations:
        region = loc[0]
        side = loc[1]
        if row_type == "ed":
            rows.add(region + side + suffix)
        else:
            rows.add(region + side)
    return rows


def less_locations(text):
    less_locs = set()
    for match in re.finditer(r"(?i)\bless(?:\s+(?:at|over|on|in))?\s+([^.;,]+)", text):
        phrase = match.group(1)
        locs, _ = extract_locations(phrase)
        less_locs.update(locs)
    return less_locs


def dominant_locations(text, base_locations):
    text_lower = parser_text(text)
    dominant = set()
    less_locs = less_locations(text_lower)
    if less_locs and base_locations:
        for loc in base_locations:
            same_region_less = any(loc[0] == less_loc[0] and loc[1] != less_loc[1] for less_loc in less_locs)
            if same_region_less and loc not in less_locs:
                dominant.add(loc)

    if LEFT_DOMINANCE_RE.search(text_lower):
        for loc in base_locations:
            if loc.endswith("L") and loc[0] + "R" not in base_locations:
                dominant.add(loc)
    if RIGHT_DOMINANCE_RE.search(text_lower):
        for loc in base_locations:
            if loc.endswith("R") and loc[0] + "L" not in base_locations:
                dominant.add(loc)

    if DOMINANCE_RE.search(text_lower):
        for match in ELECTRODE_RE.finditer(text_lower):
            token = match.group(0)
            context = text_lower[max(0, match.start() - 80) : match.end() + 80]
            if DOMINANCE_RE.search(context):
                region = electrode_region(token)
                side = electrode_side(token)
                if region and side:
                    dominant.add(region + side)
    return dominant


def add_review(reviews, report, issue_type, severity, reason, evidence):
    reviews.append(
        EegReviewIssue(
            report.patient_id,
            report.cohort,
            report.source_file,
            report.report_date,
            report.month,
            issue_type,
            severity,
            reason,
            truncate_snippet(evidence, 500),
        )
    )


def add_evidence(evidence_rows, report, feature, value, location_rows, confidence, method, descriptor, snippet):
    evidence_rows.append(
        EegEvidence(
            report.patient_id,
            report.cohort,
            report.source_file,
            report.report_date,
            report.month,
            feature,
            value,
            ";".join(sorted(location_rows)),
            confidence,
            method,
            descriptor,
            truncate_snippet(snippet),
        )
    )


def set_feature(features, feature, value):
    features[feature] = max_value(features.get(feature, ""), value)


def apply_feature_locations(
    features,
    evidence_rows,
    report,
    locations,
    row_type,
    context,
    confidence,
    descriptor,
    dominance_context=None,
):
    rows = locations_to_rows(locations, row_type)
    dominant = dominant_locations(dominance_context or context, locations)
    for row in rows:
        location = row[:2] if row_type == "ed" else row
        value = 2.0 if location in dominant else 1.0
        set_feature(features, row, value)
        add_evidence(evidence_rows, report, row, value, [], confidence, f"{row_type}_location", descriptor, context)
    return rows


def parse_report(report):
    features = {}
    evidence_rows = []
    reviews = []
    normalized_full = parser_text(report.text)
    contexts = split_contexts(report.text)
    negative_ed_final_verdict = has_negative_ed_final_verdict(report.text)
    negative_slow_final_verdict = has_negative_slow_final_verdict(report.text)

    ed_negation_text = NORMAL_SLEEP_RE.sub(" ", normalized_full)
    positive_ed_text = ED_NEGATION_RE.sub(" ", ed_negation_text)
    if not negative_ed_final_verdict and ED_NEGATION_RE.search(ed_negation_text) and (
        has_plain_spike(positive_ed_text)
        or SHARP_RE.search(positive_ed_text)
        or SPIKE_WAVE_RE.search(positive_ed_text)
        or SHARPISH_RE.search(positive_ed_text)
    ):
        add_review(
            reviews,
            report,
            "conflicting_negation",
            "review",
            "Report contains both ED-positive wording and ED negation.",
            report.text,
        )

    for location_context, fragment in contexts:
        fragment_normalized = parser_text(fragment)
        context_normalized = parser_text(location_context)
        if not fragment_normalized:
            continue

        if NORMAL_SLEEP_RE.search(fragment_normalized):
            continue

        locations, midline_only = extract_locations(context_normalized)
        ambiguous_sharpish = SHARPISH_RE.search(fragment_normalized) and not (
            SHARP_RE.search(fragment_normalized)
            or has_plain_spike(fragment_normalized)
            or SPIKE_WAVE_RE.search(fragment_normalized)
        )
        ed_negated = (
            negative_ed_final_verdict
            or is_negated(fragment_normalized, "ed")
            or is_negated(context_normalized, "ed")
        )
        slow_negated = (
            negative_slow_final_verdict
            or is_negated(fragment_normalized, "slow")
            or is_negated(context_normalized, "slow")
        )

        if PERIODIC_RE.search(fragment_normalized) and not ed_negated:
            score, descriptor = descriptor_score(fragment_normalized, "ed")
            if descriptor == "unspecified":
                descriptor = "unspecified periodic discharge"
            else:
                descriptor = f"{descriptor} periodic discharge"
            set_feature(features, "Periodic Discharge", score)
            ed_rows = apply_feature_locations(
                features,
                evidence_rows,
                report,
                locations,
                "ed",
                location_context,
                "high",
                descriptor,
                report.text,
            )
            add_evidence(
                evidence_rows,
                report,
                "Periodic Discharge",
                score,
                ed_rows,
                "high",
                "periodic_discharge",
                descriptor,
                fragment,
            )

        ed_candidates = []
        if not ed_negated:
            if SPIKE_WAVE_RE.search(fragment_normalized):
                ed_candidates.append(("Spike-and-wave ED", "spike_wave"))
            if has_plain_spike(fragment_normalized) and not re.search(r"(?i)\bvertex sharp waves?\b", fragment_normalized):
                ed_candidates.append(("Spike ED", "spike"))
            explicit_sharp = SHARP_RE.search(fragment_normalized)
            if explicit_sharp:
                ed_candidates.append(("Sharp ED", "sharp"))
            if SHARPISH_RE.search(fragment_normalized):
                ed_candidates.append(("Sharpish configuration", "sharpish_configuration"))

        if ed_candidates:
            score, descriptor = descriptor_score(fragment_normalized, "ed")
            if descriptor == "default":
                context_score, context_descriptor = descriptor_score(context_normalized, "ed")
                if context_descriptor != "default":
                    score, descriptor = context_score, context_descriptor
            confidence = "high"
            if midline_only and not locations:
                add_review(
                    reviews,
                    report,
                    "midline_only_electrodes",
                    "review",
                    "ED feature had only midline electrodes; no left/right location row was assigned.",
                    location_context,
                )
            if not locations:
                add_review(
                    reviews,
                    report,
                    "unclear_location",
                    "review",
                    "ED feature was detected without a parseable left/right regional location.",
                    location_context,
                )
            ed_rows = apply_feature_locations(
                features,
                evidence_rows,
                report,
                locations,
                "ed",
                location_context,
                confidence,
                descriptor,
                report.text,
            )
            for feature, method in ed_candidates:
                set_feature(features, feature, score)
                add_evidence(evidence_rows, report, feature, score, ed_rows, confidence, method, descriptor, fragment)

        slow_candidates = set()
        if not slow_negated:
            slow_candidates = frequency_slow_features(fragment_normalized)
            if ambiguous_sharpish and ("theta" in fragment_normalized or "delta" in fragment_normalized):
                slow_candidates.update(frequency_slow_features(fragment_normalized))

        if slow_candidates:
            score, descriptor = descriptor_score(fragment_normalized, "slow")
            if descriptor == "default":
                context_score, context_descriptor = descriptor_score(context_normalized, "slow")
                if context_descriptor != "default":
                    score, descriptor = context_score, context_descriptor
            if midline_only and not locations:
                add_review(
                    reviews,
                    report,
                    "midline_only_electrodes",
                    "review",
                    "Slow-wave feature had only midline electrodes; no left/right location row was assigned.",
                    location_context,
                )
            slow_rows = apply_feature_locations(
                features,
                evidence_rows,
                report,
                locations,
                "slow",
                location_context,
                "high",
                descriptor,
                report.text,
            )
            for feature in sorted(slow_candidates):
                set_feature(features, feature, score)
                add_evidence(evidence_rows, report, feature, score, slow_rows, "high", "slow_wave", descriptor, fragment)

    return features, evidence_rows, reviews


def extract_patient(path):
    reports, reviews = extract_reports(path)
    all_evidence = []
    all_reviews = list(reviews)
    for report in reports:
        _, evidence_rows, review_rows = parse_report(report)
        all_evidence.extend(evidence_rows)
        all_reviews.extend(review_rows)
    return all_evidence, all_reviews


def aggregate_evidence(evidence_rows):
    grouped = {}
    for evidence in evidence_rows:
        key = (evidence.source_file, evidence.month)
        row = grouped.setdefault(
            key,
            {
                "patient_id": evidence.patient_id,
                "cohort": evidence.cohort,
                "source_file": evidence.source_file,
                "report_dates": set(),
                "month": evidence.month,
                "features": {feature: "" for feature in EEG_FEATURES},
                "confidence_values": [],
                "methods": set(),
                "evidence": [],
                "evidence_count": 0,
            },
        )
        row["report_dates"].add(evidence.report_date)
        row["features"][evidence.feature] = max_value(row["features"].get(evidence.feature, ""), evidence.value)
        row["confidence_values"].append(evidence.confidence)
        row["methods"].add(evidence.method)
        row["evidence"].append(f"{evidence.feature}={format_number(evidence.value)}: {evidence.evidence}")
        row["evidence_count"] += 1

    monthly_rows = []
    for row in grouped.values():
        confidence = "high"
        if "low" in row["confidence_values"]:
            confidence = "low"
        elif "medium" in row["confidence_values"]:
            confidence = "medium"
        output_row = {
            "patient_id": row["patient_id"],
            "cohort": row["cohort"],
            "source_file": row["source_file"],
            "report_date": "||".join(sorted(row["report_dates"])),
            "month": row["month"],
            "confidence": confidence,
            "evidence_count": row["evidence_count"],
            "methods": ";".join(sorted(row["methods"])),
            "evidence": " || ".join(row["evidence"]),
        }
        for feature in EEG_FEATURES:
            output_row[feature] = format_number(row["features"].get(feature, ""))
        monthly_rows.append(output_row)
    return sorted(monthly_rows, key=lambda item: (item["cohort"], item["patient_id"], item["month"]))


def rows_to_dataframe(rows):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for DataFrame output. Activate the project venv first.") from exc
    return pd.DataFrame(rows, columns=TIMELINE_FIELDNAMES)


def table_selection(filepath):
    evidence_rows, _ = extract_patient(filepath)
    return rows_to_dataframe(aggregate_evidence(evidence_rows))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_patient_timeline_csvs(monthly_rows, output_dir):
    patient_dir = os.path.join(output_dir, "eeg_timeline")
    grouped = defaultdict(list)
    for row in monthly_rows:
        grouped[row["source_file"]].append(row)

    paths = {}
    for source_file, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["month"])
        first = rows[0]
        filename = f"{safe_filename(first['patient_id'])}{safe_filename(first['cohort'])}_eeg_timeline.csv"
        out_path = os.path.join(patient_dir, filename)
        write_csv(out_path, rows, TIMELINE_FIELDNAMES)
        paths[source_file] = out_path
    return paths


def build_summary_rows(files, reports_by_source, monthly_rows, evidence_rows, review_rows, patient_paths):
    by_source = {}
    for path in files:
        source_file = str(path)
        patient_id = patient_id_from_path(path)
        cohort = cohort_from_path(path)
        by_source[source_file] = {
            "patient_id": patient_id,
            "cohort": cohort,
            "source_file": source_file,
            "patient_timeline_csv": patient_paths.get(source_file, ""),
            "eeg_report_rows": reports_by_source.get(source_file, 0),
            "monthly_rows": 0,
            "evidence_rows": 0,
            "human_review_rows": 0,
        }

    for row in monthly_rows:
        source_file = row["source_file"]
        by_source.setdefault(
            source_file,
            {
                "patient_id": row["patient_id"],
                "cohort": row["cohort"],
                "source_file": source_file,
                "patient_timeline_csv": patient_paths.get(source_file, ""),
                "eeg_report_rows": reports_by_source.get(source_file, 0),
                "monthly_rows": 0,
                "evidence_rows": 0,
                "human_review_rows": 0,
            },
        )
        by_source[source_file]["monthly_rows"] += 1
    for evidence in evidence_rows:
        if evidence.source_file in by_source:
            by_source[evidence.source_file]["evidence_rows"] += 1
    for review in review_rows:
        if review.source_file in by_source:
            by_source[review.source_file]["human_review_rows"] += 1

    return sorted(by_source.values(), key=lambda item: (item["cohort"], item["patient_id"], item["source_file"]))


def process_directories(input_dirs=DEFAULT_INPUT_DIRS, output_dir=DEFAULT_OUTPUT_DIR):
    files = collect_docx_files(input_dirs)
    all_evidence = []
    all_review = []
    reports_by_source = {}

    for path in files:
        patient_id = patient_id_from_path(path)
        cohort = cohort_from_path(path)
        source_file = str(path)
        try:
            reports, report_reviews = extract_reports(path)
            reports_by_source[source_file] = len(reports)
            all_review.extend(report_reviews)
            for report in reports:
                _, evidence_rows, review_rows = parse_report(report)
                all_evidence.extend(evidence_rows)
                all_review.extend(review_rows)
        except Exception as exc:
            reports_by_source[source_file] = 0
            all_review.append(
                EegReviewIssue(
                    patient_id,
                    cohort,
                    source_file,
                    "",
                    "",
                    "parser_error",
                    "error",
                    str(exc),
                    "",
                )
            )

    monthly_rows = aggregate_evidence(all_evidence)
    evidence_rows = [
        {
            "patient_id": row.patient_id,
            "cohort": row.cohort,
            "source_file": row.source_file,
            "report_date": row.report_date,
            "month": row.month,
            "feature": row.feature,
            "value": format_number(row.value),
            "location_rows": row.location_rows,
            "confidence": row.confidence,
            "method": row.method,
            "descriptor": row.descriptor,
            "evidence": row.evidence,
        }
        for row in all_evidence
    ]
    review_rows = [
        {
            "patient_id": row.patient_id,
            "cohort": row.cohort,
            "source_file": row.source_file,
            "report_date": row.report_date,
            "month": row.month,
            "issue_type": row.issue_type,
            "severity": row.severity,
            "reason": row.reason,
            "evidence": row.evidence,
        }
        for row in all_review
    ]

    patient_paths = write_patient_timeline_csvs(monthly_rows, output_dir)
    patient_dir = os.path.join(output_dir, "eeg_timeline")
    evidence_path = os.path.join(output_dir, "eeg_extraction_evidence.csv")
    review_path = os.path.join(output_dir, "eeg_human_review.csv")
    summary_path = os.path.join(output_dir, "eeg_execution_summary.csv")

    write_csv(
        evidence_path,
        evidence_rows,
        [
            "patient_id",
            "cohort",
            "source_file",
            "report_date",
            "month",
            "feature",
            "value",
            "location_rows",
            "confidence",
            "method",
            "descriptor",
            "evidence",
        ],
    )
    write_csv(
        review_path,
        review_rows,
        [
            "patient_id",
            "cohort",
            "source_file",
            "report_date",
            "month",
            "issue_type",
            "severity",
            "reason",
            "evidence",
        ],
    )
    summary_rows = build_summary_rows(files, reports_by_source, monthly_rows, all_evidence, all_review, patient_paths)
    write_csv(
        summary_path,
        summary_rows,
        [
            "patient_id",
            "cohort",
            "source_file",
            "patient_timeline_csv",
            "eeg_report_rows",
            "monthly_rows",
            "evidence_rows",
            "human_review_rows",
        ],
    )

    return {
        "files": len(files),
        "eeg_report_rows": sum(reports_by_source.values()),
        "monthly_rows": len(monthly_rows),
        "evidence_rows": len(evidence_rows),
        "review_rows": len(review_rows),
        "patient_timeline_files": len(patient_paths),
        "patient_timeline_dir": patient_dir,
        "evidence_path": evidence_path,
        "review_path": review_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract monthly EEG timeline features from DOCX EEG tables.")
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
        help="Directory for per-patient EEG timelines and review reports.",
    )
    args = parser.parse_args()

    result = process_directories(args.input_dirs, args.output_dir)
    print(
        "Processed {files} DOCX files; wrote {patient_timeline_files} patient timeline CSVs, "
        "{monthly_rows} monthly rows, {evidence_rows} evidence rows, and "
        "{review_rows} human-review rows.".format(**result)
    )
    print(f"EEG report rows: {result['eeg_report_rows']}")
    print(f"Patient timelines: {result['patient_timeline_dir']}")
    print(f"Execution summary: {result['summary_path']}")
    print(f"Evidence: {result['evidence_path']}")
    print(f"Human review: {result['review_path']}")


if __name__ == "__main__":
    main()
