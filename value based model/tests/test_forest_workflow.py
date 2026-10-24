from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from forest_workflow import (FOREST_DEFAULTS, TrainingFeatureFilter, change_feature,
    compare_brier, core_feature, fit_forest, patient_weights, performance, source_importance)
from tests.helpers import local_temp_directory


class ForestWorkflowTests(unittest.TestCase):
    def test_filter_uses_training_only_and_preserves_missingness_information(self):
        training = pd.DataFrame(dict(signal=[0., 1., 0., 1.], empty=[np.nan] * 4,
            constant=[7.] * 4, observed_constant_with_missing=[5., np.nan, 5., np.nan]))
        filt = TrainingFeatureFilter(list(training)).fit(training)
        test = pd.DataFrame(dict(signal=[99.], empty=[123.], constant=[666.], observed_constant_with_missing=[np.nan]))
        result = filt.transform(test)
        self.assertEqual(list(result), ['signal', 'observed_constant_with_missing'])
        self.assertEqual(filt.all_missing_, ['empty'])
        self.assertEqual(filt.constant_, ['constant'])

    def test_patient_weights_do_not_reward_frequent_visitors(self):
        ids = pd.Series(['a'] * 4 + ['b'] * 2)
        weights = patient_weights(ids)
        self.assertAlmostEqual(weights[:4].sum(), 1)
        self.assertAlmostEqual(weights[4:].sum(), 1)
        metrics = performance([0, 0, 0, 0, 1, 1], [.5] * 6, ids, bootstrap=20)
        self.assertAlmostEqual(metrics['patient_weighted_event_fraction'], .5)
        self.assertAlmostEqual(metrics['brier'], .25)

    def test_imputation_learns_training_medians_and_models_reload(self):
        X = pd.DataFrame(dict(signal=[0., 0., 10., np.nan, 0., 10.],
                              empty=[np.nan] * 6, constant=[8.] * 6))
        model = fit_forest(X, [0, 0, 1, 1, 0, 1], ['a', 'a', 'b', 'b', 'c', 'c'],
            list(X), 'with_changes', dict(FOREST_DEFAULTS, n_estimators=8, min_samples_leaf=1, n_jobs=1))
        self.assertEqual(model.pipeline.named_steps['imputer'].statistics_[0], 0)
        test = pd.DataFrame(dict(signal=[np.nan, 1e6], empty=[123., 456.], constant=[99., 100.]))
        before = model.predict_probability(test)
        self.assertTrue(((before >= 0) & (before <= 1)).all())
        with local_temp_directory() as directory:
            file = Path(directory) / 'forest.joblib'
            joblib.dump(model, file)
            loaded = joblib.load(file)
            np.testing.assert_array_equal(before, loaded.predict_probability(test))

    def test_delta_waves_are_not_mistaken_for_change_features(self):
        self.assertFalse(change_feature('eeg__delta_waves__latest_score'))
        self.assertTrue(change_feature('eeg__delta_waves__delta_score'))
        self.assertTrue(change_feature('asm__lamotrigine_daily_mg_delta'))
        self.assertTrue(change_feature('mmse__totalscore__change_per30d'))
        self.assertTrue(change_feature('asm_regimen_changed'))
        self.assertFalse(change_feature('asm_latest_age_days'))

    def test_core_panel_is_defined_by_clinical_names_not_values(self):
        self.assertTrue(core_feature('eeg__delta_waves__latest_score', {}))
        self.assertTrue(core_feature('biochem__na__latest', {'measurement': 'Na (mmol/L)'}))
        self.assertFalse(core_feature('biochem__afp__latest', {'measurement': 'AFP (ng/mL)'}))
        self.assertFalse(core_feature('mmse__q1__latest', {'measurement': 'Q1'}))
        self.assertTrue(core_feature('mmse__totalscore__latest', {'measurement': 'TotalScore'}))

    def test_single_class_test_metrics_do_not_fabricate_auc(self):
        result = performance([0, 0], [.1, .2], ['a', 'b'], bootstrap=10)
        self.assertIsNone(result['roc_auc'])
        self.assertIsNone(result['average_precision'])

    def test_change_comparison_is_paired_by_patient(self):
        result = compare_brier(np.array([0, 0, 1, 1]), [.1, .1, .9, .9], [.5] * 4, ['a', 'a', 'b', 'b'], bootstrap=20)
        self.assertAlmostEqual(result['brier_with_changes_minus_without_changes'], -.24)
        np.testing.assert_allclose(result['bootstrap_95'], [-.24, -.24])

    def test_unusable_training_set_fails_instead_of_using_test_values(self):
        with self.assertRaisesRegex(ValueError, 'No usable training features'):
            TrainingFeatureFilter(['x']).fit(pd.DataFrame({'x': [np.nan, np.nan]}))

    def test_source_ranking_groups_missing_indicator_and_audits_all_candidates(self):
        X = pd.DataFrame(dict(signal=[0., np.nan, 1., np.nan] * 10,
            empty=[np.nan] * 40, constant=[8.] * 40, other=[0., 1., 0., 1.] * 10))
        model = fit_forest(X, [0, 1, 0, 1] * 10, ['a'] * 20 + ['b'] * 20,
            list(X), 'all_features', dict(FOREST_DEFAULTS, n_estimators=20, min_samples_leaf=1, n_jobs=1))
        ranking, selected = source_importance(model, list(X), 1)
        self.assertEqual(len(ranking), 4)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(sum(row['impurity_importance'] for row in ranking), 1.)
        by_name = {r['feature']: r for r in ranking}
        self.assertEqual(by_name['empty']['status'], 'all_missing_in_training')
        self.assertIsNone(by_name['constant']['rank'])
        self.assertEqual(by_name['constant']['impurity_importance'], 0.)
        self.assertGreater(by_name['signal']['missing_indicator_importance'], 0.)
        self.assertEqual(selected, [ranking[0]['feature']])


if __name__ == '__main__':
    unittest.main()
