"""Latest-visit cohort selection must not depend on label availability."""
import unittest
import json
from pathlib import Path

import pandas as pd

from evaluate_latest_visits import apply_folder_labels, latest_visit_indices
from tests.helpers import local_temp_directory


class LatestVisitTests(unittest.TestCase):
    def test_user_confirmation_fills_missing_folder_label_and_has_priority(self):
        with local_temp_directory() as directory:
            root = Path(directory)
            for group in ('DRE', 'NDRE'):
                (root / group).mkdir()
            (root / 'NDRE' / '0001 NS2.docx').write_bytes(b'annotation')
            overrides = root / 'overrides.json'
            overrides.write_text(json.dumps(dict(scope='latest_recorded_visit', labels=[
                dict(patient_id='0258', status='DRE', source='User confirmation'),
                dict(patient_id='0001', status='DRE', source='User correction')])) )
            targets = pd.DataFrame(dict(patient_id=['0001', '0258'], prediction_date=['2026-06-04'] * 2,
                source_asm_record_id=['1', '2'], status_at_visit=pd.Series([0, None], dtype='Int8'),
                target_reason=['computed'] * 2))
            updated, audit, summary = apply_folder_labels(targets, root, overrides)
            self.assertEqual(updated.status_at_visit.tolist(), [1, 1])
            self.assertEqual(updated.reference_label_source.tolist(), ['user_confirmed_label'] * 2)
            self.assertEqual(updated.automated_status_at_visit.iloc[0], 0)
            self.assertTrue(pd.isna(updated.automated_status_at_visit.iloc[1]))
            self.assertEqual(summary['patients_remaining_unknown'], [])
            self.assertEqual(summary['patients_without_folder_labels'], ['0258'])
            self.assertEqual(summary['patients_with_folder_labels'], 1)
            self.assertEqual(audit.folder_status_at_visit.iloc[0], 0)
            self.assertEqual(audit.reference_label_note.iloc[1], 'User confirmation')

    def test_folder_labels_fill_unknown_and_override_known_latest_labels(self):
        with local_temp_directory() as directory:
            root = Path(directory)
            (root / 'DRE').mkdir()
            (root / 'NDRE').mkdir()
            (root / 'NDRE' / '0001 NS2.docx').write_bytes(b'NDRE annotation')
            (root / 'DRE' / '0002 DRE2.docx').write_bytes(b'DRE annotation')
            targets = pd.DataFrame(dict(patient_id=['0001', '0002', '0003'],
                prediction_date=['2026-01-01'] * 3, source_asm_record_id=['1', '2', '3'],
                status_at_visit=pd.Series([None, 0, None], dtype='Int8'), target_reason=['computed'] * 3))
            updated, audit, summary = apply_folder_labels(targets, root)
            self.assertEqual(updated.status_at_visit.iloc[:2].tolist(), [0, 1])
            self.assertTrue(pd.isna(updated.status_at_visit.iloc[2]))
            self.assertTrue(pd.isna(updated.automated_status_at_visit.iloc[0]))
            self.assertEqual(updated.automated_status_at_visit.iloc[1], 0)
            self.assertEqual(summary['unknown_labels_resolved'], 1)
            self.assertEqual(summary['known_labels_changed'], 1)
            self.assertEqual(summary['patients_remaining_unknown'], ['0003'])
            self.assertEqual(audit.reference_label_source.iloc[:2].tolist(), ['provided_folder_label'] * 2)
            self.assertTrue(pd.isna(targets.status_at_visit.iloc[0]))

    def test_conflicting_folder_labels_fail(self):
        with local_temp_directory() as directory:
            root = Path(directory)
            for group in ('DRE', 'NDRE'):
                (root / group).mkdir()
                (root / group / '0001 label.docx').write_bytes(b'annotation')
            with self.assertRaisesRegex(ValueError, 'Duplicate folder label'):
                apply_folder_labels(pd.DataFrame(), root)

    def test_unsorted_dates_select_one_latest_visit_and_keep_unknown_outcome(self):
        visits = pd.DataFrame({
            'patient_id': ['0001', '0002', '0001', '0002'],
            'prediction_date': ['2020-12-01', '2019-01-01', '2019-01-01', '2021-01-01'],
            'observed_status_at_visit': [None, 1, 0, 0],
        }, index=[8, 4, 5, 2])
        selected = visits.loc[latest_visit_indices(visits)]
        self.assertEqual(selected.index.tolist(), [8, 2])
        self.assertEqual(selected.patient_id.tolist(), ['0001', '0002'])
        self.assertTrue(pd.isna(selected.iloc[0].observed_status_at_visit))
        self.assertEqual(selected.observed_status_at_visit.notna().sum(), 1)

    def test_missing_dates_are_rejected(self):
        visits = pd.DataFrame({'patient_id': ['0001'], 'prediction_date': [None]})
        with self.assertRaisesRegex(ValueError, 'valid dates'):
            latest_visit_indices(visits)

    def test_duplicate_same_day_visits_are_rejected(self):
        visits = pd.DataFrame({'patient_id': ['0001', '0001'],
                               'prediction_date': ['2020-01-01', '2020-01-01']})
        with self.assertRaisesRegex(ValueError, 'reconciliation'):
            latest_visit_indices(visits)


if __name__ == '__main__':
    unittest.main()
