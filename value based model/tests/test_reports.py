"""Report regression checks for saved uncertainty and threshold metrics."""
import copy
from pathlib import Path
import unittest

from tests.helpers import local_temp_directory
from visit_status_workflow import write_status_report


def metric_fixture():
    return dict(patients=2, visits=4, patient_weighted_event_fraction=.25,
        brier=.123456, roc_auc=.8123, average_precision=.6543,
        accuracy_at_0_5=.75, precision_at_0_5=.4, sensitivity_at_0_5=.5,
        bootstrap_95={'brier': [.10001, .15009]})


def report_fixtures():
    metrics = metric_fixture()
    baseline = dict(metrics, precision_at_0_5=None, sensitivity_at_0_5=0., bootstrap_95={})
    status = dict(target_definition='Synthetic status target', top_features=20,
        forest_settings={'n_estimators': 500}, split_patient_counts={'train': 2, 'test': 2},
        labeled_train_visits=4, labeled_test_visits=4, excluded_unknown_status_visits=0,
        limitations=[], tasks={'visit_status_history_3m': dict(models={
            'selected_features': metrics, 'training_prevalence_baseline': baseline})})
    return status


def metric_rows(content):
    """Read data tables by their headers, rejecting misaligned Markdown rows."""
    header = None
    for line in content.splitlines():
        if not line.startswith('|'):
            header = None
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if header is None:
            header = cells
        elif all(set(cell) <= set('-:') for cell in cells):
            continue
        else:
            if len(header) != len(cells):
                raise AssertionError('Report table header/data column mismatch')
            yield dict(zip(header, cells))


class ReportMetricsTests(unittest.TestCase):
    def test_saved_intervals_and_precision_are_reported(self):
        status = report_fixtures()
        for writer, report in [(write_status_report, status)]:
            with self.subTest(task=writer.__name__), local_temp_directory() as directory:
                before = copy.deepcopy(report)
                writer(report, Path(directory))
                content = (Path(directory) / 'report.md').read_text(encoding='utf-8')
                self.assertIn('9m uses [visit - 12m, visit - 3m)', content)
                self.assertIn('All four windows', content)
                rows = list(metric_rows(content))
                selected = next(row for row in rows if row.get('Model') == 'selected_features')
                self.assertEqual(selected['Brier 95% CI'], '[0.1000, 0.1501]')
                self.assertEqual(selected['Accuracy at 0.5'], '0.750')
                self.assertEqual(selected['Precision at 0.5'], '0.400')
                self.assertEqual(selected['Sensitivity at 0.5'], '0.500')
                self.assertEqual(report, before)

    def test_missing_intervals_and_undefined_precision_are_not_fabricated(self):
        status = report_fixtures()
        for writer, report in [(write_status_report, status)]:
            with self.subTest(task=writer.__name__), local_temp_directory() as directory:
                writer(report, Path(directory))
                rows = list(metric_rows((Path(directory) / 'report.md').read_text(encoding='utf-8')))
                baseline = next(row for row in rows if row.get('Model', '').endswith('baseline'))
                self.assertEqual(baseline['Precision at 0.5'], 'NA')
                self.assertEqual(baseline['Brier 95% CI'], 'NA')
                self.assertEqual(baseline['Sensitivity at 0.5'], '0.000')


if __name__ == '__main__':
    unittest.main()
