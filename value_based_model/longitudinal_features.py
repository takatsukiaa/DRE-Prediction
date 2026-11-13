"""Past-only feature calculations, using the existing audited medication rules.

History starts 12 calendar months before each medication visit and runs for
3, 6, or 9 months, leaving a prospective prediction gap. Missing is never
silently converted to zero.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from eeg_features import build_eeg_index, feature_dictionary as eeg_dictionary, history_features as eeg_history_features
from feature_schema import CLINICAL_VISIT_POLICY, medication_comparison_feature

RULES = Path(__file__).resolve().parents[1] / 'Database/dre_onset_analysis'
if str(RULES) not in sys.path:
    sys.path.insert(0, str(RULES))
from analyze_onset import ASM_NAMES, normalized, scheduled
from monthly_dre import add_months, covered_days, medication_state, month_end

FK = 'PatientBaselineViewModelPatientId'
HISTORY_WINDOWS = (3, 6, 9)
LABS = {'biochem': ('BioChemRecords', 'BioChemItemSettings'),
        'hematology': ('HematologyRecords', 'HematologyItemSettings'),
        'immuno': ('ImmunoRecords', 'ImmunoItemSettings'),
        'druglevel': ('DrugLevelRecords', 'DrugLevelItemSettings')}
EXTRA_NAMES = {'biochem': ['K (mmol/L)', 'Na (mmol/L)', 'Ca (mg/dL)', 'Ca (mmol/L)',
    'Mg (mg/dL)', 'Mg (mmol/L)', 'Cl (mmol/L)', 'CO2 (mmol/L)', 'ALB (g/dL)',
    'ALP (U/L)', 'GGT (U/L)', 'CK (U/L)', 'CRP (mg/L)', 'URIC (mg/dL)', 'VITB12 (pg/mL)', 'FOL (ng/mL)']}
COGNITION = {'mmse': ('MmseRecords', ['TotalScore'] + [f'Q{i}' for i in range(1, 12)]),
             'casi': ('CasiRecords', ['TotalScore'] + [f'Q{i}' for i in range(1, 10)]),
             'wais': ('WaisRecords', ['TotalScore'])}
EXAMS = {'eeg': ('EegRecords', 'Date'), 'eeg_focality': ('EegFocalityRecords', 'Date'),
         'brain_image': ('BrainImages', 'ExaminationDate')}
NUMERIC_STATS = {'latest': 'Latest unambiguous date-level value in 12 months',
    'delta': 'Latest minus previous distinct-date value in 12 months',
    'age_days': 'Days since latest measurement date',
    'change_gap_days': 'Days between the last two measurement dates',
    'observations_12m': 'Distinct dates with valid numeric observations in 12 months'}


def rows(tables, name):
    table = tables.get('dbo.' + name, pd.DataFrame())
    return table.astype(object).where(table.notna(), None).to_dict('records')


def number(value):
    if value is None or value is pd.NA:
        return None
    text = str(value).strip().rstrip('*').strip()
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
        return None
    result = float(text)
    return result if math.isfinite(result) else None


def day(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[T ].*)?', text):
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def lab_key(value):
    return re.sub(r'\s+', '', str(value).replace('μ', 'u').replace('µ', 'u')).casefold()


def measurement_catalog(tables):
    specs, lookups = {}, {}
    for domain, (table, setting) in LABS.items():
        lookup = {}
        for name in [r.get('Name') for r in rows(tables, setting)] + EXTRA_NAMES.get(domain, []):
            if not name or len(str(name)) > 80 or not re.search('[A-Za-z]', re.sub(r'\([^)]*\)', '', str(name))):
                continue
            key = lab_key(name)
            if key in lookup:
                continue
            slug = re.sub('[^a-z0-9]+', '_', str(name).casefold()).strip('_')[:40]
            feature = f'{domain}__{slug}_{hashlib.sha256(key.encode()).hexdigest()[:8]}'
            lookup[key] = feature
            specs[feature] = dict(source='dbo.' + table, measurement=str(name))
        lookups[domain] = lookup
    for domain, (table, fields) in COGNITION.items():
        for field in fields:
            specs[f'{domain}__{field.casefold()}'] = dict(source='dbo.' + table, measurement=field)
    return specs, lookups


def measurement_index(tables, patients, as_of, lookups, issues):
    observations = defaultdict(lambda: defaultdict(list))
    for domain, (table, _) in LABS.items():
        for record in rows(tables, table):
            pid, date = str(record.get(FK)), day(record.get('Date'))
            if pid not in patients:
                raise ValueError('Orphan patient in ' + table)
            if date is None or date > as_of:
                issues[table + ': invalid/future date'] += 1
                continue
            try:
                values = json.loads(record.get('DynamicDataJson') or '{}')
            except (ValueError, TypeError):
                values = None
            if not isinstance(values, dict):
                issues[table + ': malformed JSON'] += 1
                continue
            for key, raw in values.items():
                feature = lookups[domain].get(lab_key(key))
                if feature is None:
                    issues[table + ': unrecognized measurement key'] += 1
                    continue
                value = number(raw)
                if value is not None:
                    observations[(pid, feature)][date].append(value)
                elif raw not in (None, ''):
                    issues[table + ': nonnumeric/censored result'] += 1
    for domain, (table, fields) in COGNITION.items():
        for record in rows(tables, table):
            pid, date = str(record.get(FK)), day(record.get('Date'))
            if pid not in patients:
                raise ValueError('Orphan patient in ' + table)
            if date is None or date > as_of:
                issues[table + ': invalid/future date'] += 1
                continue
            for field in fields:
                value = number(record.get(field))
                if value is not None:
                    observations[(pid, f'{domain}__{field.casefold()}')][date].append(value)
                elif record.get(field) not in (None, ''):
                    issues[table + ': nonnumeric score'] += 1
    indexed = defaultdict(dict)
    for (pid, feature), values in observations.items():
        dates, numbers = [], []
        for date, items in sorted(values.items()):
            dates.append(date)
            numbers.append(items[0] if len(set(items)) == 1 else None)
            issues['measurements: conflicting same-date values'] += int(len(set(items)) > 1)
        indexed[pid][feature] = dates, numbers
    return indexed


def history_bounds(prediction, history_months):
    """Inclusive start and exclusive end, anchored to prediction minus 12 months."""
    if history_months not in HISTORY_WINDOWS:
        raise ValueError('history_months must be 3, 6, or 9; the 12-month model is retired')
    return add_months(prediction, -12), add_months(prediction, history_months - 12)


def numeric_history(dates, values, prediction, history_months=9, *, lower=None, end=None):
    lower = add_months(prediction, -history_months) if lower is None else lower
    end = prediction if end is None else end
    left, right = bisect_left(dates, lower), bisect_left(dates, end)
    result = {f'observations_{history_months}m': sum(v is not None for v in values[left:right])}
    if right > left:
        result.update(latest=values[right - 1], age_days=(prediction - dates[right - 1]).days)
        if right - left >= 2:
            gap = (dates[right - 1] - dates[right - 2]).days
            result['change_gap_days'] = gap
            if values[right - 1] is not None and values[right - 2] is not None:
                result['delta'] = values[right - 1] - values[right - 2]
    return result


def medication_snapshot(drug_rows):
    clean = [{**r, 'DailyDose': number(r.get('DailyDose')), 'Quantity': number(r.get('Quantity'))} for r in drug_rows]
    state = medication_state(clean)
    known, doses, ddds = set(state['substances']), {}, {}
    complete = state['count'] is not None and not state['uncertain']
    ambiguous = 0
    for substance in known:
        active = [r for r in clean if normalized(r.get('DrugName')) == substance and scheduled(r.get('Frequency'))
                  and r['DailyDose'] is not None and r['DailyDose'] > 0]
        # Multiple details can be duplicate/total-dose entries; do not guess how
        # to add strengths. Leave their dose missing pending reconciliation.
        if len(active) == 1:
            doses[substance] = active[0]['DailyDose']
            divisor = number(active[0].get('WhoDddValue'))
            if divisor is not None and divisor > 0:
                ddds[substance] = doses[substance] / divisor
        elif len(active) > 1:
            ambiguous += 1
    if ambiguous:
        state['flags'].append('Multiple active details: dose needs reconciliation')
    return dict(known=known, complete=complete, count=state['count'], doses=doses,
                total_ddd=sum(ddds.values()) if complete and len(ddds) == len(known) else None, flags=state['flags'])


def medication_difference(old, new):
    if not old['complete'] or not new['complete']:
        return {}
    a, b = old['known'], new['known']
    result = dict(asm_drug_count_delta=len(b) - len(a), asm_drugs_added_count=len(b - a),
                  asm_drugs_stopped_count=len(a - b), asm_regimen_changed=int(a != b))
    continuing = a & b
    comparable = [n for n in continuing if n in old['doses'] and n in new['doses']]
    result['asm_dose_comparable_drugs'] = len(comparable)
    if len(comparable) == len(continuing):
        result['asm_dose_increased_count'] = sum(new['doses'][n] > old['doses'][n] for n in continuing)
        result['asm_dose_decreased_count'] = sum(new['doses'][n] < old['doses'][n] for n in continuing)
    if old['total_ddd'] is not None and new['total_ddd'] is not None:
        result['asm_total_ddd_delta'] = new['total_ddd'] - old['total_ddd']
    return result


def actual_clinical_visit(value):
    """The source legend encodes both physician categories as actual visits."""
    if value is None or value is pd.NA:
        return False
    return str(value).strip() in ('0', '1')


def validate_seizure_intervals(events):
    """The source contract is non-overlapping closed intervals per patient."""
    previous = None
    for event in sorted(events, key=lambda e: (e['start'], e['end'], str(e['Id']))):
        if previous is not None and event['start'] <= previous['end']:
            raise ValueError(f'Overlapping seizure intervals: {previous["Id"]} and {event["Id"]}; reconcile source records')
        previous = event


def recorded_seizure_count(events):
    """Sum recorded counts without allocating counts across interval boundaries."""
    validate_seizure_intervals(events)
    return sum(e['SeizureCount'] for e in events)


def seizure_window(events, prediction, lower, upper):
    selected = [e for e in events if e['available'] < prediction and e['start'] <= upper and e['end'] >= lower]
    known = [e for e in selected if e['SeizureCount'] is not None]
    contained = [e for e in known if e['start'] >= lower and e['end'] <= upper]
    days = (upper - lower).days + 1
    coverage = covered_days([(e['start'], e['end']) for e in known], lower, upper)
    # A completed zero-count interval remains evidence of observed zero even
    # when it crosses the lookback boundary. Positive partial counts cannot be
    # allocated to a window without assuming event timing.
    count = recorded_seizure_count(contained) if contained or any(e['SeizureCount'] == 0 for e in known) else None
    return dict(coverage_fraction=coverage / days,
        zero_days=covered_days([(e['start'], e['end']) for e in known if e['SeizureCount'] == 0], lower, upper),
        count=count,
        unknown_intervals=sum(e['SeizureCount'] is None for e in selected))


def build_features(tables, as_of, progress=None, male_code='1', include_doses=True,
                   history_months=9, include_status_history=True):
    if history_months not in HISTORY_WINDOWS:
        raise ValueError('history_months must be 3, 6, or 9; the 12-month model is retired')
    windows = sorted({min(6, history_months), history_months})
    issues, dictionary = Counter(), {}
    patients = {str(r['PatientId']): r for r in rows(tables, 'Patients')}
    if len(patients) != len(rows(tables, 'Patients')):
        raise ValueError('Duplicate patient IDs')
    visits, record_index, drugs = defaultdict(list), {}, defaultdict(list)
    seen_dates = set()
    for record in rows(tables, 'AsmRecords'):
        pid, rid, date = str(record.get(FK)), record['Id'], day(record.get('Date'))
        if pid not in patients or rid in record_index:
            raise ValueError('Orphan patient or duplicate medication record ID')
        record_index[rid] = dict(pid=pid, date=date)
        if date is None or date > as_of:
            issues['AsmRecords: invalid/future date'] += 1
            continue
        if (pid, date) in seen_dates:
            raise ValueError('Same-day medication visits need reconciliation')
        seen_dates.add((pid, date))
        visits[pid].append(dict(date=date, id=rid, actual_visit=actual_clinical_visit(record.get('IsWuVisit'))))
    drug_details = rows(tables, 'AsmDrugDetail')
    for record in drug_details:
        if record.get('AsmRecordViewModelId') not in record_index:
            raise ValueError('Drug detail has no medication visit')
        drugs[record['AsmRecordViewModelId']].append(record)
    for group in visits.values():
        group.sort(key=lambda r: r['date'])
        for visit in group:
            visit['med'] = medication_snapshot(drugs[visit['id']])
            issues.update('medication: ' + s for s in visit['med']['flags'])
    labels, seen_months = defaultdict(list), set()
    for record in rows(tables, 'PatientMonthlyDreStatus'):
        pid, start, date = str(record.get('PatientId')), day(record.get('MonthStart')), day(record.get('AssessmentDate'))
        if pid not in patients or start is None or start.day != 1 or date != month_end(start):
            raise ValueError('Invalid monthly label key or assessment date')
        status = int(record['IsDRE']) if isinstance(record.get('IsDRE'), bool) else number(record.get('IsDRE'))
        if record.get('IsDRE') is not None and status not in (0, 1):
            raise ValueError('Monthly label must be 0, 1, or unknown')
        if (pid, date) in seen_months:
            raise ValueError('Duplicate patient-month label')
        seen_months.add((pid, date))
        if date <= as_of:
            labels[pid].append(dict(date=date, status=status))
        else:
            issues['monthly labels: future assessment excluded'] += 1
    for group in labels.values():
        group.sort(key=lambda r: r['date'])
    events = defaultdict(list)
    for record in rows(tables, 'SeizureEventViewModel'):
        pid = str(record.get(FK))
        if pid not in patients:
            raise ValueError('Orphan seizure patient')
        start, end, count = day(record.get('DateString')), day(record.get('EndDateString')), number(record.get('SeizureCount'))
        if start is None or end is None or start > end or end > as_of or (count is not None and count < 0):
            issues['seizures: invalid/future interval or negative count'] += 1
            continue
        source = record_index.get(record.get('AsmRecordViewModelId'))
        if record.get('AsmRecordViewModelId') is not None and (source is None or source['pid'] != pid or source['date'] is None):
            issues['seizures: invalid linked visit'] += 1
            continue
        available = max(end, source['date']) if source else end
        events[pid].append(dict(Id=record['Id'], start=start, end=end, SeizureCount=count, available=available))
    for pid, group in events.items():
        try:
            validate_seizure_intervals(group)
        except ValueError as error:
            raise ValueError(f'Patient {pid}: {error}') from error
    specs, lookups = measurement_catalog(tables)
    measures = measurement_index(tables, patients, as_of, lookups, issues)
    exams = defaultdict(dict)
    for domain, (table, field) in EXAMS.items():
        grouped = defaultdict(set)
        for record in rows(tables, table):
            pid, date = str(record.get(FK)), day(record.get(field))
            if pid not in patients:
                raise ValueError('Orphan patient in ' + table)
            if date is None or date > as_of:
                issues[table + ': invalid/future date'] += 1
            else:
                grouped[pid].add(date)
        for pid, dates in grouped.items():
            exams[pid][domain] = sorted(dates)
    for feature, spec in specs.items():
        for suffix, definition in NUMERIC_STATS.items():
            suffix = suffix.replace('12m', f'{history_months}m')
            dictionary[feature + '__' + suffix] = dict(**spec, definition=definition.replace('12 months', f'{history_months} months'))

    eeg_index, eeg_summary, eeg_review = build_eeg_index(rows(tables, 'EegRecords'), patients, as_of, progress)
    dictionary.update(eeg_dictionary(history_months))

    chunks, metadata, comparison_evidence = [], [], []
    coverage_diagnostics = {m: Counter() for m in windows}
    for patient_index, pid in enumerate(sorted(patients)):
        group, patient_rows = visits[pid], []
        visit_dates, label_dates = [v['date'] for v in group], [r['date'] for r in labels[pid]]
        clinical = [visit for visit in group if visit['actual_visit']]
        clinical_dates = [visit['date'] for visit in clinical]
        historical_dates = visit_dates + [e['available'] for e in events[pid]]
        historical_dates += [d for dates, _ in measures[pid].values() for d in dates]
        first_date = min(historical_dates) if historical_dates else None
        for i, visit in enumerate(group):
            prediction, row = visit['date'], {}
            lower, end = history_bounds(prediction, history_months)
            window_starts = {m: (lower if m == history_months else add_months(end, -m)) for m in windows}

            def put(name, value, source, definition):
                if name not in dictionary:
                    dictionary[name] = dict(source=source, definition=definition)
                if value is not None:
                    if not math.isfinite(float(value)):
                        raise ValueError('Nonfinite generated feature: ' + name)
                    row[name] = float(value)

            def asm(name, value):
                if history_months == 3 and medication_comparison_feature(name):
                    return
                if medication_comparison_feature(name) or name.startswith('asm_visits_'):
                    definition = (f'Actual clinical visits in the {history_months}-month input history; '
                        + CLINICAL_VISIT_POLICY + ' Unassessable changes remain missing; unchanged valid pairs equal zero.')
                    if name == 'asm_dose_comparable_drugs':
                        definition += ' Count continuing ASMs with unambiguous daily doses at both of the last two actual visits.'
                    put(name, value, 'dbo.AsmRecords.IsWuVisit + Date + dbo.AsmDrugDetail', definition)
                else:
                    put(name, value, 'dbo.AsmRecords + dbo.AsmDrugDetail', f'Past-only latest medication record in {history_months}-month history; missing regimens and ambiguous doses remain missing')

            birthday = day(patients[pid].get('Birthday'))
            put('age_years', (prediction - birthday).days / 365.2425 if birthday and birthday <= prediction else None, 'dbo.Patients.Birthday', 'Age at prediction, not Age2026')
            raw_gender = patients[pid].get('Gender')
            gender = str(raw_gender if raw_gender is not None else '').strip().casefold()
            sex_male = {'male': 1, 'm': 1, '男': 1, 'female': 0, 'f': 0, '女': 0}.get(gender)
            if male_code is not None and gender in ('0', '1'):
                sex_male = int(gender == str(male_code))
            put('sex_male', sex_male, 'dbo.Patients.Gender', 'Recognized immutable sex category; numeric codes require a confirmed legend')
            if include_status_history:
                put('history_days_available', max(0, (prediction - first_date).days) if first_date else None, 'dated source records', 'Days since earliest recorded observation, not complete coverage')
            left, stop = bisect_left(visit_dates, lower, 0, i), bisect_left(visit_dates, end, 0, i)
            history = group[left:stop]
            clinical_left, clinical_stop = bisect_left(clinical_dates, lower), bisect_left(clinical_dates, end)
            clinical_history = clinical[clinical_left:clinical_stop]
            comparison_latest = clinical_history[-1]['med'] if clinical_history else None
            previous = clinical_history[-2]['med'] if len(clinical_history) >= 2 else None
            for months in windows:
                asm(f'asm_visits_{months}m', clinical_stop - bisect_left(clinical_dates, window_starts[months], 0, clinical_stop))
            asm('asm_pair_available', int(previous is not None and comparison_latest['complete'] and previous['complete']))
            latest = history[-1]['med'] if history else None
            asm('asm_latest_age_days', (prediction - history[-1]['date']).days if history else None)
            asm('asm_previous_visit_gap_days', (clinical_history[-1]['date'] - clinical_history[-2]['date']).days if previous else None)
            asm('asm_regimen_complete', int(latest['complete']) if latest else None)
            asm('asm_drug_count_latest', len(latest['known']) if latest and latest['complete'] else None)
            asm('asm_total_ddd_latest', latest['total_ddd'] if latest else None)
            difference = medication_difference(previous, comparison_latest) if previous else {}
            for key in ('asm_drug_count_delta', 'asm_drugs_added_count', 'asm_drugs_stopped_count', 'asm_regimen_changed',
                        'asm_dose_comparable_drugs', 'asm_dose_increased_count', 'asm_dose_decreased_count', 'asm_total_ddd_delta'):
                asm(key, difference.get(key))
            for substance in ASM_NAMES:
                base = 'asm__' + substance.casefold()
                presence = 1 if latest and substance in latest['known'] else 0 if latest and latest['complete'] else None
                dose = latest['doses'].get(substance) if latest and substance in latest['known'] else 0 if latest and latest['complete'] else None
                compared_dose = comparison_latest['doses'].get(substance) if comparison_latest and substance in comparison_latest['known'] else 0 if comparison_latest and comparison_latest['complete'] else None
                old = previous['doses'].get(substance) if previous and substance in previous['known'] else 0 if previous and previous['complete'] else None
                asm(base + '_present', presence)
                asm(base + '_daily_mg', dose)
                asm(base + '_daily_mg_delta', compared_dose - old if compared_dose is not None and old is not None else None)
            window_changes = []
            for months in windows:
                part = clinical[bisect_left(clinical_dates, window_starts[months], 0, clinical_stop):clinical_stop]
                pairs = [(a, b) for a, b in zip(part, part[1:]) if a['med']['complete'] and b['med']['complete']]
                changes = [b['date'] for a, b in pairs if a['med']['known'] != b['med']['known']]
                asm(f'asm_assessable_pairs_{months}m', len(pairs))
                asm(f'asm_composition_changes_{months}m', len(changes) if pairs else None)
                if months == history_months:
                    window_changes = changes
            asm('asm_days_since_last_observed_composition_change', (prediction - window_changes[-1]).days if window_changes else None)
            if history_months != 3:
                pair = clinical_history[-2:] if len(clinical_history) >= 2 else []
                comparison_evidence.append(dict(patient_id=pid, prediction_date=prediction.isoformat(),
                    input_start=lower.isoformat(), input_end_exclusive=end.isoformat(), actual_visits=len(clinical_history),
                    previous_record_id=str(pair[0]['id']) if pair else None,
                    latest_record_id=str(pair[1]['id']) if pair else None,
                    previous_date=pair[0]['date'].isoformat() if pair else None,
                    latest_date=pair[1]['date'].isoformat() if pair else None,
                    pair_assessable=bool(pair and pair[0]['med']['complete'] and pair[1]['med']['complete'])))
            upper = end - dt.timedelta(days=1)
            for months in windows:
                start = window_starts[months]
                for key, value in seizure_window(events[pid], end, start, upper).items():
                    if key == 'coverage_fraction':
                        summary = coverage_diagnostics[months]
                        summary['windows'] += 1
                        summary['fraction_sum'] += value
                        summary['complete' if value == 1 else 'zero' if value == 0 else 'partial'] += 1
                    put(f'seizure_{key}_{months}m', value, 'dbo.SeizureEventViewModel',
                        'Completed and available intervals only; counts sum fully contained records; coverage is the clipped interval union; zero does not imply unobserved days are seizure-free')
            for feature, (dates, values) in measures[pid].items():
                for key, value in numeric_history(dates, values, prediction, history_months, lower=lower, end=end).items():
                    if value is not None:
                        row[feature + '__' + key] = value
            for name, value in eeg_history_features(eeg_index[pid], prediction, history_months, lower=lower, end=end).items():
                if value is not None:
                    row[name] = value
            for domain in EXAMS:
                dates = exams[pid].get(domain, [])
                right, first = bisect_left(dates, end), bisect_left(dates, lower)
                timings = [(f'dates_{m}m', right - bisect_left(dates, window_starts[m])) for m in windows]
                timings.append(('latest_age_days', (prediction - dates[right - 1]).days if right > first else None))
                for suffix, value in timings:
                    put(domain + '_' + suffix, value, 'dbo.' + EXAMS[domain][0], 'Database examination timing; encoded EEG content is represented separately')
            right, first = bisect_left(label_dates, end), bisect_left(label_dates, lower)
            part = labels[pid][first:right]
            current = labels[pid][right - 1] if right else None
            known = [r['status'] for r in part if r['status'] is not None]
            adjacent = [(a, b) for a, b in zip(part, part[1:]) if a['status'] is not None and b['status'] is not None and month_end(add_months(a['date'], 1)) == b['date']]
            values = dict(current_dre=current['status'] if current else None, current_dre_unknown=int(not current or current['status'] is None),
                current_state_age_days=(prediction - current['date']).days if current else None,
                monthly_known_fraction_12m=len(known) / len(part) if part else None,
                monthly_positive_fraction_12m=sum(known) / len(known) if known else None,
                monthly_observed_switches_12m=sum(a['status'] != b['status'] for a, b in adjacent) if adjacent else None)
            for name, value in values.items():
                if include_status_history:
                    put(name.replace('12m', f'{history_months}m'), value, 'dbo.PatientMonthlyDreStatus', 'Completed month-end history strictly before prediction; unknowns are not filled')
            meta = dict(patient_id=pid, prediction_date=prediction, history_start=lower, history_end_exclusive=end, source_asm_record_id=str(visit['id']),
                        current_state_date=current['date'] if current else None, current_dre=values['current_dre'])
            metadata.append(meta)
            patient_rows.append(row)
        if patient_rows:
            chunks.append(pd.DataFrame(patient_rows, dtype=np.float32))
        if progress and (patient_index + 1) % 25 == 0:
            progress(f'Processed {patient_index + 1}/{len(patients)} patients')
    if not metadata:
        raise ValueError('No valid prediction visits')
    features = pd.concat(chunks, ignore_index=True).reindex(columns=list(dictionary)).astype(np.float32)
    if not include_doses:
        omitted = [name for name in features if '_daily_mg' in name or name.startswith(('asm_total_ddd', 'asm_dose_'))]
        features = features.drop(columns=omitted)
        for name in omitted:
            dictionary.pop(name)
    for name in features.columns:
        if name.endswith(f'__observations_{history_months}m'):
            features[name] = features[name].fillna(0)
    metadata = pd.DataFrame(metadata)
    metadata['current_dre'] = metadata['current_dre'].astype('Int8')
    for name, spec in dictionary.items():
        if name not in ('age_years', 'sex_male'):
            spec['history_start_months_before_prediction'] = 12
            spec['history_end_months_before_prediction'] = 12 - history_months
            spec['history_end_exclusive'] = True
    if np.isinf(features.to_numpy()).any():
        raise ValueError('Feature overflow; review source numeric values')
    audit = dict(patient_count=len(patients), patients_with_visits=int(metadata.patient_id.nunique()), prediction_visits=len(metadata),
        feature_count=len(features.columns), all_missing_features=int(features.isna().all().sum()),
        as_of=as_of.isoformat(), history_months=history_months, history_anchor_months=12, status_history_included=include_status_history,
        prediction_timing='strictly before medication visit',
        daily_dose_unit='mg/day', dose_features_included=include_doses, numeric_male_code=male_code,
        clinical_visit_policy=CLINICAL_VISIT_POLICY, medication_comparisons_enabled=history_months != 3,
        medication_comparison_evidence=comparison_evidence,
        coverage_fractions_in_predictors=True,
        seizure_coverage_fraction_diagnostics={f'{m}m': dict(windows=s['windows'],
            complete=int(s['complete']), zero=int(s['zero']), partial=int(s['partial']),
            mean_coverage_fraction=s['fraction_sum'] / s['windows'] if s['windows'] else None)
            for m, s in coverage_diagnostics.items()},
        dose_value_policy='Preserve stored doses without clipping; largest source doses listed for review',
        source_issues=dict(sorted(issues.items())), eeg_encoding=eeg_summary, eeg_report_audit=eeg_review,
        limitations=['Recording timestamps unavailable; observation dates are used.',
            'EEG findings use the existing rule encoder and require representative clinical review; imaging remains timing-only.',
            'Lab recognition is conservative; unrecognized keys and censored values are excluded.',
            'Multiple active medication details leave dose missing pending reconciliation.',
            'No learned imputation, predictive feature selection, or model fitting has run.'])
    # Ranked source review only, not an inferred clinical dose threshold and not
    # a predictor. Do not silently clip or delete these source measurements.
    largest_doses = sorted((r for r in drug_details if number(r.get('DailyDose')) is not None),
                           key=lambda r: number(r['DailyDose']), reverse=True)[:10]
    audit['largest_stored_daily_doses_for_review'] = [dict(drug_detail_id=str(r['Id']),
        asm_record_id=str(r['AsmRecordViewModelId']), substance=normalized(r.get('DrugName')),
        daily_mg=number(r['DailyDose'])) for r in largest_doses]
    return features, metadata, dictionary, audit
