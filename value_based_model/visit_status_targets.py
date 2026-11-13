"""Visit-date status under the project's existing three-valued DRE rules.

The target is assessed using visit-day evidence. Predictors must stop before
the visit. The target's 12-month seizure window is independent of input length.
Unknown visit status uses the latest completed monthly assessment when known.
"""
from bisect import bisect_left
from collections import defaultdict
import datetime as dt

import pandas as pd

from longitudinal_features import FK, HISTORY_WINDOWS, add_months, day, month_end, number, rows
from monthly_dre import classify, medication_state, seizure_state

TARGET_DEFINITION = ('DRE status documented at a medication visit: at least two concurrent '
                     'scheduled ASMs and positive seizure evidence in '
                     '(visit minus 12 calendar months, visit]; existing three-valued rules. '
                     'If visit status is unknown, use the latest monthly assessment strictly '
                     'before the visit only when that assessment is known; do not skip unknown months.')


def build_visit_status_targets(visits, tables, as_of):
    records, drugs, events = {}, defaultdict(list), defaultdict(list)
    monthly, seen_months = defaultdict(list), set()
    for row in rows(tables, 'PatientMonthlyDreStatus'):
        pid = str(row['PatientId'])
        start, assessment = day(row.get('MonthStart')), day(row.get('AssessmentDate'))
        raw_status = row.get('IsDRE')
        status = int(raw_status) if isinstance(raw_status, bool) else number(raw_status)
        if (start is None or start.day != 1 or assessment != month_end(start)
                or (row.get('IsDRE') is not None and status not in (0, 1))):
            raise ValueError('Invalid monthly status assessment')
        if (pid, assessment) in seen_months:
            raise ValueError('Duplicate patient-month assessment')
        seen_months.add((pid, assessment))
        if assessment <= as_of:
            monthly[pid].append((assessment, status, row.get('Reason')))
    for history in monthly.values():
        history.sort(key=lambda row: row[0])
    monthly_dates = {pid: [row[0] for row in history] for pid, history in monthly.items()}
    for record in rows(tables, 'AsmRecords'):
        key = str(record['Id'])
        if key in records:
            raise ValueError('Duplicate medication record ID')
        records[key] = dict(patient_id=str(record[FK]), date=day(record.get('Date')))
    for drug in rows(tables, 'AsmDrugDetail'):
        key = str(drug['AsmRecordViewModelId'])
        if key not in records:
            raise ValueError('Drug detail has no medication visit')
        drugs[key].append({**drug, 'DailyDose': number(drug.get('DailyDose')),
                          'Quantity': number(drug.get('Quantity'))})
    seen = set()
    for event in rows(tables, 'SeizureEventViewModel'):
        if event['Id'] in seen:
            raise ValueError('Duplicate seizure interval ID')
        seen.add(event['Id'])
        pid = str(event[FK])
        start, end = day(event.get('DateString')), day(event.get('EndDateString'))
        count = number(event.get('SeizureCount'))
        if start is None or end is None or start > end or end > as_of or (count is not None and count < 0):
            continue
        linked = event.get('AsmRecordViewModelId')
        source = records.get(str(linked)) if linked is not None else None
        if linked is not None and (source is None or source['patient_id'] != pid or source['date'] is None):
            continue
        available = max(end, source['date']) if source else end
        events[pid].append(dict(Id=event['Id'], start=start, end=end,
                                SeizureCount=count, available=available))
    result = []
    for visit in visits.to_dict('records'):
        pid, date, rid = str(visit['patient_id']), day(visit['prediction_date']), str(visit['source_asm_record_id'])
        record = records.get(rid)
        if date is None or date > as_of or record != dict(patient_id=pid, date=date):
            raise ValueError('Visit does not match its source medication record')
        lower = add_months(date, -12) + dt.timedelta(days=1)
        med = medication_state(drugs[rid])
        sz = seizure_state([e for e in events[pid] if e['available'] <= date], lower, date)
        status, reason = classify(med['state'], sz['state'])
        if status is None:
            history = monthly.get(pid, [])
            previous = bisect_left(monthly_dates.get(pid, []), date) - 1
            if previous >= 0 and history[previous][1] is not None:
                _, status, monthly_reason = history[previous]
                reason = monthly_reason or ('DRE status.' if status else 'NDRE status.')
        result.append(dict(patient_id=pid, prediction_date=date, source_asm_record_id=rid,
            status_at_visit=None if status is None else int(status), target_reason=reason,
            target_window_start=lower, target_window_end=date,
            target_asm_count=med['count'], target_asm_substances=med['substances'],
            target_asm_uncertain=med['uncertain'], target_asm_flags=med['flags'],
            # Legacy monthly-rule key is confined to this compatibility boundary.
            target_seizure_count=sz['lower_bound'],
            target_seizure_known_days=sz['known_days'], target_seizure_zero_days=sz['zero_days'],
            target_window_days=sz['window_days'], target_positive_interval_ids=sz['positive_ids'],
            target_partial_positive_interval_ids=sz['partial_positive_ids'],
            target_zero_interval_ids=sz['zero_ids'], target_unknown_interval_ids=sz['unknown_ids']))
    frame = pd.DataFrame(result, index=visits.index)
    frame['status_at_visit'] = frame.status_at_visit.astype('Int8')
    return frame
