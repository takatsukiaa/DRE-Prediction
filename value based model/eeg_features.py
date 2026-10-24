"""Adapt the existing EEG text encoder to dated database records.

No DOCX input, new vocabulary fitting, or outcome labels are needed. Missing
parser output is not negative evidence. Ordinal scores are not probabilities.
"""
from collections import Counter, defaultdict
import datetime as dt
from functools import lru_cache
from pathlib import Path
import re
import sys

PARSERS = Path(__file__).resolve().parents[1] / 'docx_parsers'
if str(PARSERS) not in sys.path:
    sys.path.insert(0, str(PARSERS))
import docx_parser_eeg_timeline as encoder


@lru_cache(maxsize=64)
def feature_slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.casefold()).strip('_')


def _date(value):
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _add_months(date, offset):
    import calendar
    year, month = divmod(date.year * 12 + date.month - 1 + offset, 12)
    return dt.date(year, month + 1, min(date.day, calendar.monthrange(year, month + 1)[1]))


DEFINITE_ED = [name for name in encoder.ED_FEATURES if name != 'Sharpish configuration']


def encode_report(text, patient_id, source_id, report_date):
    """Return scores, explicit negatives, and review evidence from one segment."""
    report = encoder.EegReport(str(patient_id), '', f'sql:EegRecords:{source_id}',
        report_date.isoformat(), report_date.strftime('%Y-%m'), 0, text)
    features, evidence, reviews = encoder.parse_report(report)
    scores = {name: None for name in encoder.EEG_FEATURES}
    issues = {item.issue_type for item in reviews}
    probable = bool(encoder.PROBABLY_NORMAL_RE.search(text))
    normal_heading = encoder.is_normal_eeg_record(text)
    normal = normal_heading and not probable
    unusable = not text.strip() or encoder.is_non_eeg_notice(text)
    if unusable:
        issues.add('empty_or_non_eeg_text')
        features = {}
    elif normal and features:
        issues.add('normal_heading_with_positive_findings')
        normal = False
        features = {}
    elif normal:
        scores = dict.fromkeys(scores, 0.0)
    elif not probable:
        scores.update({name: float(value) for name, value in features.items()})
        if encoder.has_negative_ed_final_verdict(text):
            for name in encoder.ED_FEATURES + encoder.ED_LOCATION_ROWS:
                scores[name] = 0.0
        if encoder.has_negative_slow_final_verdict(text):
            for name in encoder.SLOW_FEATURES + encoder.SLOW_LOCATION_ROWS:
                scores[name] = 0.0
    else:
        issues.add('probably_normal_not_definite_negative')
    # Contradictory ED wording is retained for review, not trusted as an ED
    # absence or an exact severity change. Slow findings can remain assessable.
    if 'conflicting_negation' in issues:
        for name in encoder.ED_FEATURES + encoder.ED_LOCATION_ROWS:
            scores[name] = None
    if not any(value is not None for value in scores.values()):
        issues.add('no_assessable_encoded_findings')
    ed = 1.0 if any(scores[n] is not None and scores[n] > 0 for n in DEFINITE_ED) else 0.0 if all(scores[n] == 0 for n in DEFINITE_ED) else None
    slow = 1.0 if any(scores[n] is not None and scores[n] > 0 for n in encoder.SLOW_FEATURES) else 0.0 if all(scores[n] == 0 for n in encoder.SLOW_FEATURES) else None
    return dict(scores=scores, ed_present=ed, slow_present=slow, explicit_normal=float(normal),
                probably_normal=float(probable), review_required=float(bool(issues)), issues=sorted(issues),
                evidence=[dict(feature=e.feature, value=e.value, method=e.method,
                    confidence=e.confidence, descriptor=e.descriptor, evidence=e.evidence) for e in evidence])


def build_eeg_index(records, patient_ids, as_of, progress=None):
    indexed, audits, counts = defaultdict(list), [], Counter()
    for position, record in enumerate(records):
        pid, source_id = str(record.get('PatientBaselineViewModelPatientId')), str(record.get('Id'))
        if pid not in patient_ids:
            raise ValueError('EEG record has no patient')
        source_date = _date(record.get('Date'))
        if source_date is None or source_date > as_of:
            counts['invalid_or_future_source_date'] += 1
            continue
        text = encoder.normalize_text(record.get('Note'))
        segments, _ = encoder.split_embedded_reports(source_date.isoformat(), source_date.strftime('%Y-%m'), text)
        if not segments:
            segments = [(source_date.isoformat(), source_date.strftime('%Y-%m'), text)]
        for segment_index, (date_text, _, segment) in enumerate(segments):
            report_date = _date(date_text)
            if report_date is None or report_date > as_of:
                counts['invalid_or_future_embedded_date'] += 1
                continue
            encoded = encode_report(segment, pid, source_id, report_date)
            if len(segments) > 1:
                encoded['issues'] = sorted(set(encoded['issues']) | {'embedded_report_split'})
                encoded['review_required'] = 1.0
            available = max(source_date, report_date)
            encoded.update(date=report_date, available=available, source_id=source_id)
            indexed[pid].append(encoded)
            audits.append(dict(patient_id=pid, source_record_id=source_id, segment_index=segment_index,
                source_date=source_date.isoformat(), report_date=report_date.isoformat(), available_on=available.isoformat(),
                scores=encoded['scores'], issues=encoded['issues'], evidence=encoded['evidence']))
            counts['encoded_segments'] += 1
            counts['definite_normal_segments'] += int(encoded['explicit_normal'])
            counts['segments_with_assessable_findings'] += int(any(v is not None for v in encoded['scores'].values()))
            counts['segments_needing_review'] += int(encoded['review_required'])
            counts.update('review:' + issue for issue in encoded['issues'])
        if progress and (position + 1) % 1000 == 0:
            progress(f'Encoded {position + 1}/{len(records)} database EEG records')
    for entries in indexed.values():
        entries.sort(key=lambda row: (row['date'], row['available'], row['source_id']))
    return indexed, dict(counts), audits


@lru_cache(maxsize=3)
def feature_dictionary(history_months=12):
    result = {}
    definitions = {
        'latest_score': 'Existing ordinal encoder value at the latest available EEG date; unmentioned is missing',
        'latest_present': '1 for a positive encoded score, 0 for explicit negative evidence, missing otherwise',
        'delta_score': 'Latest minus previous distinct EEG date score within 12 months, only when both are assessable',
        'max_score_6m': 'Maximum observed ordinal score in the prior 6 months; missing if never assessable',
        'max_score_12m': 'Maximum observed ordinal score in the prior 12 months; missing if never assessable',
    }
    definitions = {key: value.replace('12 months', f'{history_months} months')
                   for key, value in definitions.items() if not key.startswith('max_score_')}
    for months in sorted({min(6, history_months), history_months}):
        definitions[f'max_score_{months}m'] = f'Maximum observed ordinal score in the prior {months} months; missing if never assessable'
    for feature in encoder.EEG_FEATURES:
        for suffix, definition in definitions.items():
            result[f'eeg__{feature_slug(feature)}__{suffix}'] = dict(source='dbo.EegRecords.Note',
                measurement=feature, definition=definition, encoder='docx_parser_eeg_timeline.parse_report')
    flags = {
        'eeg_encoded_latest_report_age_days': 'Clinical report age, not age of the enclosing database row',
        'eeg_encoded_previous_report_gap_days': 'Gap between the two most recent distinct available report dates within 12 months',
        'eeg_latest_explicit_normal': 'Definite normal report heading without conflicting positive findings',
        'eeg_latest_probably_normal': 'Probably-normal wording; not treated as definite negative evidence',
        'eeg_latest_review_required': 'Latest EEG date has parser or same-date conflict issues',
        'eeg_latest_encoded_ed_present': 'ED morphology detected by the encoder; sharpish alone does not establish this indicator',
        'eeg_latest_encoded_slow_present': 'Theta/delta finding detected by the encoder, or explicit negative evidence',
        'eeg_ed_newly_detected': 'Previous ED indicator 0 and latest 1; both within history and assessable',
        'eeg_ed_no_longer_detected': 'Previous ED indicator 1 and latest 0; both within history and assessable',
    }
    flags = {key: value.replace('12 months', f'{history_months} months') for key, value in flags.items()}
    for months in sorted({min(6, history_months), history_months}):
        flags.update({f'eeg_encoded_dates_{months}m': 'Distinct available EEG report dates within this window',
            f'eeg_assessable_dates_{months}m': 'Report dates with at least one assessable encoded score',
            f'eeg_ed_positive_fraction_{months}m': 'Fraction among dates with known ED indicator; missing if none assessable',
            f'eeg_slow_positive_fraction_{months}m': 'Fraction among dates with known slow-wave indicator; missing if none assessable'})
    for name, definition in flags.items():
        result[name] = dict(source='dbo.EegRecords.Note + Date', definition=definition)
    return result


def _same_date(rows):
    """Combine available same-day reports; explicit positive/negative conflicts stay missing."""
    scores, conflict = {}, False
    for name in encoder.EEG_FEATURES:
        known = [r['scores'][name] for r in rows if r['scores'][name] is not None]
        contradictory = 0.0 in known and any(v > 0 for v in known)
        conflict |= contradictory
        scores[name] = None if contradictory or not known else max(known)
    states = {}
    for name in ('ed_present', 'slow_present'):
        known = {r[name] for r in rows if r[name] is not None}
        states[name] = next(iter(known)) if len(known) == 1 else None
        conflict |= len(known) > 1
    return dict(date=rows[0]['date'], scores=scores, **states,
                explicit_normal=float(all(r['explicit_normal'] for r in rows)),
                probably_normal=float(any(r['probably_normal'] for r in rows)),
                review_required=float(conflict or any(r['review_required'] for r in rows)))


def history_features(reports, prediction, history_months=12, *, lower=None, end=None):
    lower = _add_months(prediction, -history_months) if lower is None else lower
    end = prediction if end is None else end
    by_date = defaultdict(list)
    for report in reports:
        if lower <= report['date'] < end and report['available'] < end:
            by_date[report['date']].append(report)
    history = [_same_date(group) for _, group in sorted(by_date.items())]
    result = dict.fromkeys(feature_dictionary(history_months))
    for months in sorted({min(6, history_months), history_months}):
        start = lower if months == history_months else _add_months(end, -months)
        group = [row for row in history if row['date'] >= start]
        result[f'eeg_encoded_dates_{months}m'] = len(group)
        result[f'eeg_assessable_dates_{months}m'] = sum(any(v is not None for v in r['scores'].values()) for r in group)
        for field, output in [('ed_present', 'ed'), ('slow_present', 'slow')]:
            values = [r[field] for r in group if r[field] is not None]
            result[f'eeg_{output}_positive_fraction_{months}m'] = sum(values) / len(values) if values else None
        for name in encoder.EEG_FEATURES:
            values = [r['scores'][name] for r in group if r['scores'][name] is not None]
            result[f'eeg__{feature_slug(name)}__max_score_{months}m'] = max(values) if values else None
    if not history:
        return result
    latest, previous = history[-1], history[-2] if len(history) > 1 else None
    result.update(eeg_encoded_latest_report_age_days=(prediction - latest['date']).days,
        eeg_encoded_previous_report_gap_days=(latest['date'] - previous['date']).days if previous else None,
        eeg_latest_explicit_normal=latest['explicit_normal'], eeg_latest_probably_normal=latest['probably_normal'],
        eeg_latest_review_required=latest['review_required'], eeg_latest_encoded_ed_present=latest['ed_present'],
        eeg_latest_encoded_slow_present=latest['slow_present'])
    for name in encoder.EEG_FEATURES:
        key = 'eeg__' + feature_slug(name)
        value, old = latest['scores'][name], previous['scores'][name] if previous else None
        result[key + '__latest_score'] = value
        result[key + '__latest_present'] = float(value > 0) if value is not None else None
        result[key + '__delta_score'] = value - old if value is not None and old is not None else None
    if previous and previous['ed_present'] is not None and latest['ed_present'] is not None:
        result['eeg_ed_newly_detected'] = float(previous['ed_present'] == 0 and latest['ed_present'] == 1)
        result['eeg_ed_no_longer_detected'] = float(previous['ed_present'] == 1 and latest['ed_present'] == 0)
    return result
