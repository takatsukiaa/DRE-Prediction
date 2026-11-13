"""Active feature policy and explicit compatibility with historical column names."""

FEATURE_SCHEMA = 'recorded_counts_actual_visits_v2'
CLINICAL_VISIT_POLICY = ('AsmRecords.IsWuVisit: 0 = visit with another physician; '
    '1 = visit with Dr. Wu; - = no visit record. Missing/unrecognized markers are excluded. '
    'Compare consecutive distinct actual visit dates strictly inside the input window.')
COMPARISON_FIELDS = frozenset(('asm_pair_available', 'asm_previous_visit_gap_days',
    'asm_drug_count_delta', 'asm_drugs_added_count', 'asm_drugs_stopped_count',
    'asm_regimen_changed', 'asm_dose_comparable_drugs', 'asm_dose_increased_count',
    'asm_dose_decreased_count', 'asm_total_ddd_delta',
    'asm_days_since_last_observed_composition_change'))


def medication_comparison_feature(name):
    return (name in COMPARISON_FIELDS or name.startswith(('asm_assessable_pairs_', 'asm_composition_changes_'))
        or (name.startswith('asm__') and name.endswith('_daily_mg_delta')))


def migrate_feature(name):
    """Legacy identifiers are confined to migration and schema validation."""
    if name == 'asm_drug_count_lower_bound':
        return None, 'Consolidated into asm_drug_count_latest'
    if name.startswith('seizure_count_lower_bound_'):
        return name.replace('seizure_count_lower_bound_', 'seizure_count_', 1), 'Recorded count terminology'
    if name.startswith('seizure_rate_'):
        return None, 'Retain recorded count instead of calendar-scaled rate'
    if name.startswith('seizure_partial_positive_intervals_'):
        return None, 'Removed by explicit user instruction'
    if name.startswith('seizure_known_days_'):
        return None, 'Retain coverage fraction instead of coverage days'
    if name.startswith('eeg__') and name.endswith('__latest_present'):
        return None, 'Retain encoded score instead of derived binary presence'
    if name.endswith('__change_per30d'):
        return None, 'Retain delta and elapsed days instead of derived change rate'
    return name, 'Retained'


def validate_active_features(columns, history_months=None):
    invalid = [name for name in columns if migrate_feature(name)[0] != name or 'lower_bound' in name
        or (history_months == 3 and medication_comparison_feature(name))]
    if invalid:
        raise ValueError('Historical feature definitions require rebuilding: ' + ', '.join(invalid[:8]))


def validate_legacy_migration(old_frames, new_frames, old_visits, new_visits, old_targets, new_targets):
    """Reproduce the recorded-count v1 audit without importing optional audit runners."""
    import pandas as pd
    pd.testing.assert_frame_equal(old_visits, new_visits)
    pd.testing.assert_frame_equal(old_targets.rename(columns={
        'target_seizure_count_lower_bound': 'target_seizure_count'}), new_targets)
    inventory = []
    for months, old in old_frames.items():
        expected = []
        for name in old:
            replacement, reason = migrate_feature(name)
            inventory.append(dict(history_months=months, old_feature=name, new_feature=replacement,
                action='removed' if replacement is None else 'renamed' if replacement != name else 'retained', reason=reason))
            if replacement is not None:
                expected.append(replacement)
                pd.testing.assert_series_equal(old[name], new_frames[months][replacement], check_names=False)
        if set(expected) != set(new_frames[months]) or len(expected) != len(new_frames[months].columns):
            raise AssertionError('Unexpected feature-schema change')
    return pd.DataFrame(inventory)
