"""Plot orientation, class percentages, and reuse of saved evaluation results."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from forest_workflow import performance
from plot_confusion_matrices import confusion_matrix_figure, main
from tests.helpers import local_temp_directory


class ConfusionMatrixTests(unittest.TestCase):
    def test_chart_uses_observed_rows_and_threshold_inclusive_dre_columns(self):
        metrics = performance([0, 0, 0, 1, 1, 1, 1], [.1, .5, .2, .1, .2, .3, .5],
                              ['a'] * 3 + ['b'] * 4, bootstrap=0)
        figure = confusion_matrix_figure(metrics, '3m history | Selected features')
        try:
            axes = figure.axes[0]
            self.assertEqual(metrics['confusion_matrix_raw'], [[2, 1], [3, 1]])
            self.assertEqual([text.get_text() for text in axes.get_xticklabels()], ['DRE', 'NDRE'])
            self.assertEqual([text.get_text() for text in axes.get_yticklabels()], ['DRE', 'NDRE'])
            self.assertEqual(axes.xaxis.get_label_position(), 'top')
            self.assertEqual(axes.get_xlabel(), 'Predicted')
            self.assertEqual(axes.get_ylabel(), 'Actual')
            self.assertEqual(axes.images[0].get_cmap().name, 'Blues')
            # Cells retain only the original count and percentage text.
            annotations = [text.get_text() for text in axes.texts]
            self.assertEqual(annotations, ['1\n(25.0%)', '3\n(75.0%)', '1\n(33.3%)', '2\n(66.7%)'])
            self.assertEqual([text.get_text() for text in figure.texts], ['3m history | Selected features'])
        finally:
            figure.clear()

    def test_single_observed_class_keeps_both_axes_and_marks_undefined_percentages(self):
        figure = confusion_matrix_figure({'confusion_matrix_raw': [[2, 1], [0, 0]]}, 'Single class')
        try:
            labels = [text.get_text() for text in figure.axes[0].texts]
            self.assertEqual(labels[:2], ['0\n(N/A)', '0\n(N/A)'])
        finally:
            figure.clear()

    def test_existing_metrics_generate_readable_png_without_training(self):
        with local_temp_directory() as directory:
            root = Path(directory)
            report = dict(task='visit_status', tasks={'visit_status_history_6m': dict(models={
                'selected_features': {'confusion_matrix_raw': [[2, 1], [3, 1]]}})})
            metrics_path = root / 'metrics.json'
            original = json.dumps(report)
            metrics_path.write_text(original, encoding='utf-8')
            with patch('sys.argv', ['plot_confusion_matrices.py', '--results-dir', str(root)]), \
                 patch('builtins.print'), patch('visit_status_workflow.run_status_training', side_effect=AssertionError('Retrained')):
                main()
            with Image.open(root / 'confusion_matrices/visit_status_history_6m_selected_features.png') as png:
                self.assertEqual(png.format, 'PNG')
                self.assertGreater(png.width, 800)
                png.verify()
            self.assertEqual(metrics_path.read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
