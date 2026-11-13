"""Audit invariants: patient isolation, missingness, interval logic and resampling."""
import datetime as dt
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from feature_audit import choose_candidate, fit_pair
    from feature_audit_diagnostics import (diagnostic_frame, exact_groups, experiment_definitions,
        domain, observation_feature, pair_evidence, representation_evidence)
    from feature_audit_metrics import cluster_bootstrap, summarize
except ModuleNotFoundError as error:
    if error.name not in ('feature_audit', 'feature_audit_diagnostics', 'feature_audit_metrics'):
        raise
    raise unittest.SkipTest('Historical audit sources are absent; active clinical-visit and bootstrap tests run separately') from error
from forest_workflow import FOREST_DEFAULTS, patient_weights
from longitudinal_features import history_bounds, medication_snapshot, seizure_window
from monthly_dre import seizure_lower_bound
from tests.helpers import drug, local_temp_directory


class FeatureAuditTests(unittest.TestCase):
    def test_raw_examination_age_is_separate_from_encoded_eeg_content(self):
        self.assertEqual(domain('eeg_latest_age_days'), 'examination_timing')
        self.assertEqual(domain('eeg_focality_latest_age_days'), 'examination_timing')
        self.assertEqual(domain('eeg__spike_ed__latest_score'), 'eeg_content')
        self.assertEqual(domain('eeg_encoded_dates_3m'), 'eeg_content')

    def test_exact_duplicates_preserve_distinct_missingness_and_prefer_complete_count(self):
        X = pd.DataFrame({'asm_drug_count_lower_bound': [1., 2., np.nan],
            'asm_drug_count_latest': [1., 2., np.nan], 'other': [1., np.nan, np.nan]})
        groups = exact_groups(X)
        self.assertEqual(groups, [['asm_drug_count_latest', 'asm_drug_count_lower_bound']])
        definitions = experiment_definitions(X.columns, groups)
        self.assertNotIn('asm_drug_count_lower_bound', definitions['deduplicate_exact'])
        self.assertIn('other', definitions['deduplicate_exact'])
        evidence = pair_evidence(X, 'asm_drug_count_latest', 'other', pd.Series(['a', 'b', 'c']))
        self.assertEqual(evidence['different_missingness_visits'], 1)

    def test_diagnostic_model_excludes_clinical_values_and_includes_availability(self):
        X = pd.DataFrame({'asm_visits_3m': [1, 2], 'asm_drug_count_latest': [2, np.nan], 'age_years': [40, 50]})
        Z = diagnostic_frame(X)
        self.assertNotIn('asm_drug_count_latest', Z)
        self.assertNotIn('age_years', Z)
        self.assertEqual(Z['availability__asm_drug_count_latest'].tolist(), [1, 0])
        self.assertIn('asm_visits_3m', Z)
        self.assertFalse(observation_feature('seizure_zero_days_3m'))
        self.assertFalse(observation_feature('seizure_partial_positive_intervals_3m'))
        self.assertFalse(observation_feature('asm_dose_comparable_drugs'))

    def test_cognitive_sum_checks_distinguish_same_date_and_unaligned_measurements(self):
        X = pd.DataFrame({f'casi__q{i}__latest': [1., 1., 1.] for i in range(1, 10)})
        X['casi__totalscore__latest'] = [9., 10., 12.]
        for i in range(1, 10):
            X[f'casi__q{i}__age_days'] = [1., 1., 2.]
        X['casi__totalscore__age_days'] = [1., 1., 3.]
        data = representation_evidence(X, pd.Series(['a', 'b', 'c']))['cognition_total_components'][0]
        self.assertEqual(data['complete_visits'], 3)
        self.assertEqual(data['mismatches'], 2)
        self.assertEqual(data['same_date_complete_visits'], 2)
        self.assertEqual(data['same_date_mismatches'], 1)

    def test_bootstrap_matches_explicit_cluster_weights_for_all_metrics(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([.1, .5, .5, .9, .4, .8])
        ids = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
        boot = cluster_bootstrap(y, p, ids, replicates=25, seed=42)
        multiplicity = np.random.default_rng(42).multinomial(3, [1/3] * 3, size=25)
        for i, draw in enumerate(multiplicity):
            weights = np.repeat(draw, 2) / 2
            if weights[y == 0].sum() and weights[y == 1].sum():
                self.assertAlmostEqual(boot['roc_auc'][i], roc_auc_score(y, p, sample_weight=weights))
                self.assertAlmostEqual(boot['average_precision'][i], average_precision_score(y, p, sample_weight=weights))
            negative = p < .5
            expected = weights[(y == 0) & negative].sum() / weights[negative].sum() if weights[negative].sum() else np.nan
            np.testing.assert_allclose(boot['npv_at_0_5'][i], expected, equal_nan=True)
            self.assertAlmostEqual(boot['brier'][i], np.average((p-y)**2, weights=weights))

    def test_paired_self_comparison_is_zero_and_npv_can_be_undefined(self):
        y, p, ids = [0, 1, 0, 1], [.7, .8, .6, .9], ['a', 'a', 'b', 'b']
        metrics = summarize(y, p, ids, p, replicates=20)
        self.assertIsNone(metrics['npv_at_0_5'])
        self.assertIsNone(metrics['bootstrap_95']['npv_at_0_5'])
        self.assertEqual(metrics['paired_difference_from_baseline']['brier']['estimate'], 0)
        self.assertEqual(metrics['paired_difference_from_baseline']['brier']['bootstrap_95'], [0., 0.])

    def test_partial_positive_crosses_left_boundary_and_cannot_establish_count(self):
        lower, end = history_bounds(dt.date(2020, 1, 15), 3)
        events = [dict(Id=1, start=lower-dt.timedelta(days=2), end=lower+dt.timedelta(days=2),
            available=lower+dt.timedelta(days=2), SeizureCount=4),
            dict(Id=2, start=end-dt.timedelta(days=2), end=end+dt.timedelta(days=2),
            available=end+dt.timedelta(days=2), SeizureCount=99)]
        result = seizure_window(events, end, lower, end-dt.timedelta(days=1))
        self.assertNotIn('partial_positive_intervals', result)
        self.assertIsNone(result['count'])
        self.assertAlmostEqual(result['coverage_fraction'], 3 / (end-lower).days)

    def test_lower_bound_is_valid_but_loose_for_chain_overlap(self):
        base = dt.date(2020, 1, 1)
        events = [dict(Id=i, start=base+dt.timedelta(days=a), end=base+dt.timedelta(days=b), SeizureCount=c)
                  for i, (a,b,c) in enumerate([(0, 2, 2), (2, 4, 1), (4, 6, 3)])]
        self.assertEqual(seizure_lower_bound(events), 3)
        # First and last intervals are disjoint and guarantee at least five.
        self.assertEqual(events[0]['SeizureCount'] + events[2]['SeizureCount'], 5)

    def test_recorded_zero_with_partial_coverage_is_not_seizure_freedom(self):
        lower = dt.date(2020, 1, 1)
        event = dict(Id=1, start=lower, end=lower, available=lower, SeizureCount=0)
        result = seizure_window([event], dt.date(2020, 2, 1), lower, dt.date(2020, 1, 31))
        self.assertEqual(result['count'], 0)
        self.assertLess(result['coverage_fraction'], 1)

    def test_incomplete_regimen_is_not_promoted_to_a_complete_count(self):
        records = [drug('Lamotrigine'), dict(drug('Valproate'), DailyDose=None, Quantity=None)]
        state = medication_snapshot(records)
        self.assertEqual(state['count'], 1)
        self.assertFalse(state['complete'])

    def test_calendar_lengths_vary_without_independent_rate_measurements(self):
        lengths = []
        for date in [dt.date(2020, 1, 31), dt.date(2020, 5, 31)]:
            start, end = history_bounds(date, 3)
            lengths.append((end-start).days)
        self.assertNotEqual(*lengths)
        for days in lengths:
            count, rate = 4, 4*30/days
            self.assertAlmostEqual(rate*days/30, count)

    def test_cache_reuses_predictions_but_invalidates_changed_feature_pool(self):
        X = pd.DataFrame({'a': np.arange(30), 'b': np.arange(30)[::-1]})
        train, valid = np.arange(20), np.arange(20, 30)
        y = np.tile([0, 1], 10)
        ids = pd.Series([str(i//2) for i in train])
        with local_temp_directory() as directory:
            cache = Path(directory) / 'fit'
            settings = dict(FOREST_DEFAULTS, n_estimators=4, n_jobs=1)
            a, meta = fit_pair(X, y, ids, train, valid, list(X), settings, cache, 'synthetic')
            b, reused = fit_pair(X, y, ids, train, valid, list(X), settings, cache, 'synthetic')
            np.testing.assert_array_equal(a['selected_features'], b['selected_features'])
            self.assertEqual(meta['cache_key'], reused['cache_key'])
            _, changed = fit_pair(X, y, ids, train, valid, ['a'], settings, cache, 'synthetic')
            self.assertNotEqual(meta['cache_key'], changed['cache_key'])


if __name__ == '__main__':
    unittest.main()
