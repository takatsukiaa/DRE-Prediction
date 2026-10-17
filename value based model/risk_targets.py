"""Any opposite monthly state during follow-up, including subsequent reversals.

Targets concern observations on the stored month-end grid. Switches between
assessments that reverse before the next assessment cannot be identified.
"""
import datetime as dt

import numpy as np
import pandas as pd

from longitudinal_features import add_months, day, month_end


def future_assessments(prediction, months):
    """Month ends strictly after prediction and on/before its exact anniversary."""
    cutoff = add_months(prediction, months)
    current = month_end(prediction)
    if current <= prediction:
        current = month_end(add_months(current, 1))
    dates = []
    while current <= cutoff:
        dates.append(current)
        current = month_end(add_months(current, 1))
    return dates, cutoff


def build_switch_targets(visits, monthly, as_of):
    lookup, by_patient = {}, {}
    for row in monthly.astype(object).where(monthly.notna(), None).to_dict('records'):
        pid, date = str(row['PatientId']), day(row['AssessmentDate'])
        start = day(row['MonthStart'])
        if date is None or start is None or start.day != 1 or month_end(start) != date:
            raise ValueError('Invalid monthly assessment date')
        status = row['IsDRE']
        if status is not None and status not in (0, 1):
            raise ValueError('Monthly status must be 0, 1, or unknown')
        if (pid, date) in lookup:
            raise ValueError('Duplicate patient-month assessment')
        lookup[(pid, date)] = int(status) if status is not None else None
        if date <= as_of:
            by_patient.setdefault(pid, []).append((date, lookup[(pid, date)]))
    for values in by_patient.values():
        values.sort()
    result = []
    from bisect import bisect_left
    for row in visits.to_dict('records'):
        pid, prediction = str(row['patient_id']), day(row['prediction_date'])
        if prediction is None or prediction > as_of:
            raise ValueError('Invalid prediction date')
        history = by_patient.get(pid, [])
        i = bisect_left([date for date, _ in history], prediction)
        origin_date, origin = history[i - 1] if i else (None, None)
        supplied = row.get('current_dre')
        if (origin is None) != pd.isna(supplied) or (origin is not None and int(supplied) != origin):
            raise ValueError('Feature bundle current state disagrees with its source monthly table')
        item = dict(patient_id=pid, prediction_date=prediction, current_dre=origin,
                    current_state_date=origin_date,
                    transition_direction='NDRE_to_DRE' if origin == 0 else 'DRE_to_NDRE' if origin == 1 else None)
        for months in (6, 12):
            expected, cutoff = future_assessments(prediction, months)
            known = [(date, lookup.get((pid, date))) for date in expected if date <= as_of]
            opposite = [date for date, status in known if status is not None and origin is not None and status != origin]
            observed = sum(status is not None for _, status in known)
            complete = cutoff <= as_of and observed == len(expected)
            target = None if origin is None else 1 if opposite else 0 if complete else None
            reason = 'unknown_starting_state' if origin is None else 'opposite_observed' if opposite else 'complete_no_opposite' if complete else 'incomplete_followup'
            item.update({f'switch_within_{months}m': target, f'followup_end_{months}m': cutoff,
                f'first_opposite_assessment_{months}m': min(opposite) if opposite else None,
                f'expected_months_{months}m': len(expected), f'known_months_{months}m': observed,
                f'target_reason_{months}m': reason})
        result.append(item)
    targets = pd.DataFrame(result, index=visits.index)
    for name in ('current_dre', 'switch_within_6m', 'switch_within_12m'):
        targets[name] = targets[name].astype('Int8')
    # Horizon consistency of the underlying event definition, not a correction
    # of probabilities later estimated by independent forests.
    assert not (targets.switch_within_6m.eq(1) & targets.switch_within_12m.eq(0)).any()
    return targets


def risk_percentages(probability, current_dre):
    probability = np.asarray(probability, dtype=float)
    current = pd.to_numeric(pd.Series(current_dre), errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    if len(probability) != len(current) or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid risk probabilities or row alignment')
    applicable = np.isin(current, [0, 1])
    switch = np.where(applicable, probability * 100, np.nan)
    return dict(risk_any_switch_pct=switch, risk_no_switch_pct=100 - switch,
                risk_ndre_to_dre_pct=np.where(current == 0, switch, np.nan),
                risk_dre_to_ndre_pct=np.where(current == 1, switch, np.nan))
