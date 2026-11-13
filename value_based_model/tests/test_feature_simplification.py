"""The active schema removes derived copies while preserving evidence and timing."""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from feature_schema import migrate_feature, validate_active_features, validate_legacy_migration as validate_migration
from longitudinal_features import (build_features, history_bounds, numeric_history,
    recorded_seizure_count, seizure_window, validate_seizure_intervals)
from tests.helpers import fixture
from visit_status_targets import build_visit_status_targets


class FeatureSimplificationTests(unittest.TestCase):
    def test_active_catalog_removes_every_requested_representation(self):
        for months in (3, 6, 9):
            X, _, dictionary, audit = build_features(fixture(), dt.date(2021, 2, 1),
                history_months=months, include_status_history=False)
            validate_active_features(X.columns)
            self.assertEqual(list(X), list(dictionary))
            self.assertIn('asm_drug_count_latest', X)
            self.assertIn('asm_total_ddd_latest', X)
            self.assertIn('mmse__totalscore__latest', X)
            for width in sorted({min(6, months), months}):
                self.assertIn(f'seizure_count_{width}m', X)
                self.assertIn(f'seizure_coverage_fraction_{width}m', X)
                self.assertTrue(X[f'seizure_coverage_fraction_{width}m'].between(0, 1).all())
                self.assertIn(f'seizure_zero_days_{width}m', X)
            for name in X:
                self.assertNotIn('lower_bound', name)
                self.assertFalse(name.startswith(('seizure_rate_', 'seizure_known_days_',
                    'seizure_partial_positive_intervals_')))
                self.assertFalse(name.endswith(('__latest_present', '__change_per30d')))
            self.assertTrue(audit['coverage_fractions_in_predictors'])
            validate_active_features(X.columns, months)

    def test_direct_count_rejects_closed_interval_overlap_but_accepts_adjacency(self):
        base = dt.date(2020, 1, 1)
        events = [dict(Id=1, start=base, end=base, SeizureCount=2),
            dict(Id=2, start=base+dt.timedelta(days=1), end=base+dt.timedelta(days=2), SeizureCount=3)]
        self.assertEqual(recorded_seizure_count(events), 5)
        events[1]['start'] = base
        with self.assertRaisesRegex(ValueError, '1 and 2'):
            recorded_seizure_count(events)

    def test_all_source_intervals_are_checked_even_when_counts_are_zero_or_unknown(self):
        base = dt.date(2020, 1, 1)
        events = [dict(Id=1, start=base, end=base, SeizureCount=0),
            dict(Id=2, start=base, end=base, SeizureCount=None)]
        with self.assertRaisesRegex(ValueError, 'Overlapping seizure intervals'):
            validate_seizure_intervals(events)

    def test_boundary_positive_is_unallocated_but_coverage_is_preserved(self):
        start, end = history_bounds(dt.date(2020, 1, 15), 3)
        event = dict(Id=1, start=start-dt.timedelta(days=1), end=start+dt.timedelta(days=2),
            available=start+dt.timedelta(days=2), SeizureCount=4)
        result = seizure_window([event], end, start, end-dt.timedelta(days=1))
        self.assertIsNone(result['count'])
        self.assertAlmostEqual(result['coverage_fraction'], 3 / (end-start).days)
        self.assertEqual(set(result), {'count', 'coverage_fraction', 'zero_days', 'unknown_intervals'})
        event['SeizureCount'] = 0
        self.assertEqual(seizure_window([event], end, start, end-dt.timedelta(days=1))['count'], 0)

    def test_measurement_delta_and_gap_survive_without_change_rate(self):
        result = numeric_history([dt.date(2019, 11, 1), dt.date(2019, 12, 1)], [4, 7], dt.date(2020, 1, 15))
        self.assertEqual(result['delta'], 3)
        self.assertEqual(result['change_gap_days'], 30)
        self.assertNotIn('change_per30d', result)

    def test_new_target_audit_renames_only_the_count_heading(self):
        tables = fixture()
        _, visits, _, _ = build_features(tables, dt.date(2021, 2, 1), include_status_history=False)
        targets = build_visit_status_targets(visits, tables, dt.date(2021, 2, 1))
        self.assertIn('target_seizure_count', targets)
        self.assertFalse(any('lower_bound' in name for name in targets))
        # This is label-adjudication evidence, not a model predictor.
        self.assertIn('target_partial_positive_interval_ids', targets)

    def test_migration_fails_on_missingness_or_outcome_changes(self):
        old = pd.DataFrame({'seizure_count_lower_bound_3m': [0., np.nan],
            'asm_drug_count_latest': [2., np.nan], 'asm_drug_count_lower_bound': [2., np.nan]})
        new = old.drop(columns='asm_drug_count_lower_bound').rename(columns={'seizure_count_lower_bound_3m': 'seizure_count_3m'})
        visits = pd.DataFrame({'patient_id': ['a', 'b']})
        old_targets = pd.DataFrame({'status_at_visit': [0., np.nan], 'target_seizure_count_lower_bound': [0., np.nan]})
        new_targets = old_targets.rename(columns={'target_seizure_count_lower_bound': 'target_seizure_count'})
        old_frames, new_frames = {m: old for m in (3,6,9)}, {m: new for m in (3,6,9)}
        inventory = validate_migration(old_frames, new_frames, visits, visits, old_targets, new_targets)
        self.assertEqual(inventory.action.eq('removed').sum(), 3)
        changed = new.copy()
        changed.loc[1, 'seizure_count_3m'] = 0
        with self.assertRaises(AssertionError):
            validate_migration(old_frames, {**new_frames, 3: changed}, visits, visits, old_targets, new_targets)
        changed_targets = new_targets.copy()
        changed_targets.loc[0, 'status_at_visit'] = 1
        with self.assertRaises(AssertionError):
            validate_migration(old_frames, new_frames, visits, visits, old_targets, changed_targets)

    def test_historical_schema_requires_rebuild_for_new_training(self):
        for name in ('asm_drug_count_lower_bound', 'seizure_count_lower_bound_9m',
                     'seizure_partial_positive_intervals_6m', 'eeg__spike_ed__latest_present'):
            with self.assertRaisesRegex(ValueError, 'require rebuilding'):
                validate_active_features([name])
        validate_active_features(['seizure_coverage_fraction_6m'])
        self.assertEqual(migrate_feature('seizure_count_lower_bound_9m')[0], 'seizure_count_9m')


if __name__ == '__main__':
    unittest.main()
