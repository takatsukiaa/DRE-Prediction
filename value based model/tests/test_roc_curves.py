"""ROC comparisons preserve the held-out cohort and patient-weighted metrics."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from plot_roc_curves import main, roc_comparison_figure
from tests.helpers import local_temp_directory
from tests.test_reports import report_fixtures


def prediction_fixture():
    return pd.DataFrame(dict(patient_id=['0001', '0001', '0001', '0002', '0003'],
        split=['test'] * 5, prediction_scope=['heldout_patient'] * 5,
        observed_status_at_visit=[0, 0, 1, 1, None],
        risk_dre_pct_history_3m=[10, 60, 50, 90, np.nan],
        risk_dre_pct_history_6m=[90, 40, 50, 10, np.nan],
        risk_dre_pct_history_9m=[50, 50, 50, 50, np.nan],
        risk_dre_pct_history_12m=[10, 20, 80, 90, np.nan]))


class RocCurveTests(unittest.TestCase):
    def test_four_curves_use_common_known_labels_and_equal_patient_weights(self):
        figure, data = roc_comparison_figure(prediction_fixture())
        try:
            self.assertEqual((data['labeled_visits'], data['patients']), (4, 2))
            self.assertEqual(list(data['curves']), ['3m', '6m', '9m', '12m'])
            for history, expected in zip(data['curves'], [.875, .125, .5, 1.]):
                curve = data['curves'][history]
                self.assertAlmostEqual(curve['roc_auc'], expected)
                self.assertAlmostEqual(np.trapezoid(curve['true_positive_rate'], curve['false_positive_rate']), expected)
            # Unweighted 3-month AUROC would be .75, so this detects lost weights.
            np.testing.assert_allclose(data['curves']['3m']['false_positive_rate'], [0, 0, .5, .5, 1])
            np.testing.assert_allclose(data['curves']['3m']['true_positive_rate'], [0, .75, .75, 1, 1])
            axes = figure.axes[0]
            self.assertEqual(len(axes.lines), 5)
            self.assertEqual(axes.lines[0].get_label(), '3-month history (AUROC = 0.875)')
            self.assertEqual(axes.lines[-1].get_linestyle(), '--')
            self.assertIn('False positive rate', axes.get_xlabel())
            self.assertIn('True positive rate', axes.get_ylabel())
        finally:
            figure.clear()

    def test_training_rows_are_rejected(self):
        predictions = prediction_fixture()
        for column, value in [('split', 'train'), ('prediction_scope', 'in_sample_training_patient')]:
            with self.subTest(column=column):
                changed = predictions.copy()
                changed.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, 'held-out test patients only'):
                    roc_comparison_figure(changed)

    def test_missing_or_invalid_scores_cannot_change_the_comparison_cohort(self):
        predictions = prediction_fixture()
        for value in [np.nan, np.inf, -1, 101]:
            with self.subTest(value=value):
                changed = predictions.copy()
                changed.loc[0, 'risk_dre_pct_history_9m'] = value
                with self.assertRaisesRegex(ValueError, 'common cohort'):
                    roc_comparison_figure(changed)
        with self.assertRaisesRegex(ValueError, 'all four'):
            roc_comparison_figure(predictions.drop(columns='risk_dre_pct_history_9m'))

    def test_single_observed_class_marks_auc_undefined(self):
        predictions = prediction_fixture()
        predictions['observed_status_at_visit'] = 0
        predictions = predictions.iloc[:4]
        figure, data = roc_comparison_figure(predictions)
        try:
            self.assertTrue(all(curve['roc_auc'] is None for curve in data['curves'].values()))
            self.assertTrue(all(curve['false_positive_rate'] == [] for curve in data['curves'].values()))
            self.assertIn('AUROC undefined', figure.axes[0].lines[0].get_label())
        finally:
            figure.clear()

    def test_unknown_or_invalid_labels_are_rejected(self):
        for labels in ([None] * 5, [0, 0, 1, 2, None]):
            predictions = prediction_fixture()
            predictions['observed_status_at_visit'] = labels
            with self.assertRaisesRegex(ValueError, 'binary visit labels'):
                roc_comparison_figure(predictions)

    def test_existing_results_generate_plot_data_and_report_without_retraining(self):
        with local_temp_directory() as directory:
            root = Path(directory)
            report = dict(report_fixtures(), task='visit_status', history_windows_months=[3, 6, 9, 12])
            original = json.loads(json.dumps(report))
            (root / 'metrics.json').write_text(json.dumps(report), encoding='utf-8')
            prediction_fixture().to_json(root / 'heldout_status_predictions.jsonl', orient='records', lines=True)
            with patch('sys.argv', ['plot_roc_curves.py', '--results-dir', str(root)]), \
                 patch('builtins.print'), patch('visit_status_workflow.run_status_training', side_effect=AssertionError('Retrained')):
                main()
            updated = json.loads((root / 'metrics.json').read_text())
            roc = updated.pop('roc_comparison')
            self.assertEqual(updated, original)
            with Image.open(root / roc['plot']) as png:
                self.assertEqual(png.format, 'PNG')
                self.assertGreater(png.width, 1000)
                png.verify()
            data = json.loads((root / roc['data']).read_text())
            self.assertEqual(data['curves']['3m']['roc_auc'], .875)
            self.assertIn(roc['plot'], (root / 'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
