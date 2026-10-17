"""Database EEG adapter and availability tests; no clinical files required."""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from eeg_features import build_eeg_index, encode_report, feature_dictionary, history_features
from longitudinal_features import build_features
from tests.test_longitudinal_features import fixture


NORMAL = 'EEG diagnosis: Normal. Description: Vertex sharp waves and sleep spindles during sleep.'
POSITIVE = 'EEG diagnosis: Abnormal. Description: Frequent spikes over the left temporal region.'


def record(identifier, date, text):
    return dict(Id=identifier, Date=date, Note=text, PatientBaselineViewModelPatientId='0001')


def index(records):
    return build_eeg_index(records, {'0001'}, dt.date(2021, 12, 31))[0]['0001']


class EegFeatureTests(unittest.TestCase):
    def test_existing_scores_and_locations_are_reused(self):
        parsed = encode_report(POSITIVE, '0001', '1', dt.date(2020, 1, 1))
        self.assertEqual(parsed['scores']['Spike ED'], 3)
        self.assertEqual(parsed['scores']['TLED'], 1)
        self.assertEqual(parsed['ed_present'], 1)
        self.assertIsNone(parsed['scores']['Sharp ED'])
        self.assertTrue(parsed['evidence'])

    def test_normal_and_missing_reports_are_different(self):
        normal = history_features(index([record(1, '2019-12-01', NORMAL)]), dt.date(2020, 1, 15))
        missing = history_features([], dt.date(2020, 1, 15))
        self.assertEqual(normal['eeg__spike_ed__latest_score'], 0)
        self.assertEqual(normal['eeg_latest_explicit_normal'], 1)
        self.assertIsNone(missing['eeg__spike_ed__latest_score'])
        self.assertEqual(missing['eeg_encoded_dates_12m'], 0)

    def test_unknown_or_probably_normal_does_not_become_negative(self):
        for text in ('Report unavailable; please review.', 'EEG diagnosis: Probably normal.'):
            parsed = encode_report(text, '0001', '1', dt.date(2020, 1, 1))
            self.assertIsNone(parsed['ed_present'])
            self.assertIsNone(parsed['scores']['Spike ED'])
            self.assertEqual(parsed['review_required'], 1)

    def test_normal_sleep_is_not_encoded_as_spike_discharge(self):
        parsed = encode_report(NORMAL, '0001', '1', dt.date(2020, 1, 1))
        self.assertEqual(parsed['scores']['Spike ED'], 0)
        self.assertEqual(parsed['scores']['Sharp ED'], 0)

    def test_final_explicit_negation_is_retained(self):
        parsed = encode_report('Interpretation: No evidence of epileptiform activity. No slow waves.', '0001', '1', dt.date(2020, 1, 1))
        self.assertEqual(parsed['ed_present'], 0)
        self.assertEqual(parsed['slow_present'], 0)

    def test_last_two_reports_produce_changes_within_history_only(self):
        reports = index([record(1, '2019-11-01', NORMAL), record(2, '2019-12-01', POSITIVE)])
        result = history_features(reports, dt.date(2020, 1, 15))
        self.assertEqual(result['eeg__spike_ed__delta_score'], 3)
        self.assertEqual(result['eeg__spike_ed__latest_present'], 1)
        self.assertEqual(result['eeg_ed_newly_detected'], 1)
        self.assertEqual(result['eeg_encoded_previous_report_gap_days'], 30)
        self.assertIsNone(history_features(reports, dt.date(2020, 11, 15))['eeg__spike_ed__delta_score'])

    def test_prediction_day_and_future_reports_are_excluded(self):
        old = index([record(1, '2019-12-01', NORMAL)])
        future = index([record(1, '2019-12-01', NORMAL), record(2, '2020-01-15', POSITIVE), record(3, '2020-02-01', POSITIVE)])
        self.assertEqual(history_features(old, dt.date(2020, 1, 15)), history_features(future, dt.date(2020, 1, 15)))

    def test_embedded_future_report_is_not_backdated(self):
        reports = index([record(1, '2020-01-01', '2020-03-01 EEG diagnosis: Abnormal. Frequent spikes over the left temporal region.')])
        self.assertIsNone(history_features(reports, dt.date(2020, 1, 15))['eeg__spike_ed__latest_score'])
        self.assertEqual(history_features(reports, dt.date(2020, 4, 1))['eeg__spike_ed__latest_score'], 3)

    def test_old_embedded_report_is_not_available_before_database_row(self):
        reports = index([record(1, '2020-03-01', '2019-12-01 EEG diagnosis: Abnormal. Frequent spikes over the left temporal region.')])
        self.assertIsNone(history_features(reports, dt.date(2020, 1, 15))['eeg__spike_ed__latest_score'])

    def test_unparsed_latest_report_does_not_carry_old_positive_forward(self):
        reports = index([record(1, '2019-11-01', POSITIVE), record(2, '2019-12-01', 'No usable report text.')])
        values = history_features(reports, dt.date(2020, 1, 15))
        self.assertIsNone(values['eeg__spike_ed__latest_score'])
        self.assertEqual(values['eeg__spike_ed__max_score_12m'], 3)
        self.assertEqual(values['eeg_latest_review_required'], 1)

    def test_sharpish_is_preserved_separately_from_definite_ed_indicator(self):
        parsed = encode_report('Description: Rare sharpish configuration over the left temporal region.', '0001', '1', dt.date(2020, 1, 1))
        self.assertEqual(parsed['scores']['Sharpish configuration'], 1)
        self.assertIsNone(parsed['ed_present'])

    def test_same_day_contradiction_is_not_arbitrarily_ordered(self):
        reports = index([record(1, '2019-12-01', NORMAL), record(2, '2019-12-01', POSITIVE)])
        values = history_features(reports, dt.date(2020, 1, 15))
        self.assertIsNone(values['eeg__spike_ed__latest_score'])
        self.assertIsNone(values['eeg_latest_encoded_ed_present'])
        self.assertEqual(values['eeg_latest_review_required'], 1)

    def test_full_builder_excludes_report_text_and_keeps_future_labels_separate(self):
        tables = fixture()
        tables['dbo.EegRecords'] = pd.DataFrame([record(1, '2019-12-01', POSITIVE)])
        features, visits, dictionary, audit = build_features(tables, dt.date(2021, 2, 1))
        self.assertEqual(features.iloc[-1]['eeg__spike_ed__latest_score'], 3)
        self.assertEqual(features.iloc[-1]['eeg_dates_12m'], 1)
        self.assertEqual(visits.iloc[-1].target_dre_6m, 0)
        self.assertTrue(all(dtype == np.dtype('float32') for dtype in features.dtypes))
        self.assertTrue(set(feature_dictionary()).issubset(dictionary))
        self.assertEqual(audit['eeg_encoding']['encoded_segments'], 1)
        self.assertEqual(audit['eeg_report_audit'][0]['source_record_id'], '1')


if __name__ == '__main__':
    unittest.main()
