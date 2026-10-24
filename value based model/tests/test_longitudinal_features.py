"""Temporal and clinical invariants for feature engineering; no database access."""
import copy
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from longitudinal_features import (build_features, medication_difference, medication_snapshot,
                                   number, numeric_history, seizure_window)
from tests.helpers import drug, fixture


class FeatureTests(unittest.TestCase):
    def build(self, tables=None):
        return build_features(tables or fixture(), dt.date(2021, 2, 1))

    def test_substitution_and_dose_change_despite_unchanged_count(self):
        features, visits, _, _ = self.build()
        row = features.iloc[-1]
        self.assertEqual(row.asm_drug_count_delta, 0)
        self.assertEqual(row.asm_drugs_added_count, 1)
        self.assertEqual(row.asm_drugs_stopped_count, 1)
        self.assertEqual(row.asm_regimen_changed, 1)
        self.assertEqual(row.asm_dose_increased_count, 1)
        self.assertEqual(row['asm__lamotrigine_daily_mg_delta'], 100)
        self.assertEqual(row.asm_total_ddd_delta, 1)
        self.assertEqual(row.asm_previous_visit_gap_days, 30)
        self.assertEqual(visits.iloc[-1].patient_id, '0001')

    def test_prediction_visit_medication_is_excluded(self):
        tables = fixture()
        before, _, _, _ = self.build(tables)
        tables['dbo.AsmDrugDetail'].loc[lambda r: r.AsmRecordViewModelId.eq(3), 'DailyDose'] = 99999
        after, _, _, _ = self.build(tables)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.iloc[-1]['asm__carbamazepine_present'], 0)

    def test_future_labels_do_not_change_historical_features(self):
        tables = fixture()
        before, visits, _, _ = self.build(tables)
        self.assertEqual(visits.iloc[-1].current_dre, 1)
        self.assertFalse(any(c.startswith(("target_", "outcome_", "sample_weight_")) for c in visits))
        tables["dbo.PatientMonthlyDreStatus"].loc[lambda r: r.MonthStart.eq("2020-01-01"), "IsDRE"] = 1
        after, changed, _, _ = self.build(tables)
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(visits, changed)

    def test_unknown_latest_month_does_not_backfill_previous_positive(self):
        tables = fixture()
        extra = pd.DataFrame([dict(PatientId='0001', MonthStart='2019-11-01', AssessmentDate='2019-11-30', IsDRE=1)])
        tables['dbo.PatientMonthlyDreStatus'] = pd.concat([tables['dbo.PatientMonthlyDreStatus'], extra])
        tables['dbo.PatientMonthlyDreStatus'].loc[lambda r: r.MonthStart.eq('2019-12-01'), 'IsDRE'] = None
        features, visits, _, _ = self.build(tables)
        self.assertTrue(np.isnan(features.iloc[-1].current_dre))
        self.assertEqual(features.iloc[-1].current_dre_unknown, 1)
        self.assertTrue(pd.isna(visits.iloc[-1].current_dre))

    def test_first_visit_and_expired_history_do_not_create_zero_changes(self):
        tables = fixture()
        features, _, _, _ = self.build(tables)
        self.assertTrue(np.isnan(features.iloc[0].asm_drug_count_delta))
        self.assertEqual(features.iloc[0].asm_pair_available, 0)
        tables['dbo.AsmRecords'].loc[lambda r: r.Id.eq(1), 'Date'] = '2018-11-01'
        features, _, _, _ = self.build(tables)
        self.assertTrue(np.isnan(features.iloc[-1].asm_drug_count_delta))
        self.assertEqual(features.iloc[-1].asm_visits_12m, 1)

    def test_unknown_regimen_and_ambiguous_dose_remain_missing(self):
        known = medication_snapshot([drug('Lamotrigine')])
        unknown = medication_snapshot([])
        self.assertFalse(unknown['complete'])
        self.assertIsNone(unknown['lower_count'])
        self.assertEqual(medication_difference(known, unknown), {})
        duplicate = medication_snapshot([drug('Lamotrigine'), drug('Lamotrigine')])
        self.assertNotIn('Lamotrigine', duplicate['doses'])
        self.assertIsNone(duplicate['total_ddd'])

    def test_numeric_changes_use_distinct_dates_within_window(self):
        prediction = dt.date(2020, 1, 15)
        values = numeric_history([dt.date(2018, 1, 1), dt.date(2019, 11, 1), dt.date(2019, 12, 1), prediction], [999, 4, 7, 1000], prediction)
        self.assertEqual(values['latest'], 7)
        self.assertEqual(values['delta'], 3)
        self.assertEqual(values['change_gap_days'], 30)
        self.assertEqual(values['observations_12m'], 2)
        self.assertEqual(number('0'), 0)
        self.assertEqual(number('23*'), 23)
        self.assertIsNone(number('>90'))
        self.assertIsNone(number('24/30'))

    def test_lab_units_are_separate_and_same_day_conflicts_are_missing(self):
        tables = fixture()
        tables['dbo.BioChemRecords'] = pd.DataFrame([
            dict(Id=1, PatientBaselineViewModelPatientId='0001', Date='2019-12-01', DynamicDataJson='{"Ca (mg/dL)": 9, "Ca (mmol/L)": 2.2}'),
            dict(Id=2, PatientBaselineViewModelPatientId='0001', Date='2019-12-01', DynamicDataJson='{"Ca (mg/dL)": 10, "123": 123}'),
            dict(Id=3, PatientBaselineViewModelPatientId='0001', Date='2020-01-15', DynamicDataJson='{"Ca (mmol/L)": 99}')])
        features, _, dictionary, audit = self.build(tables)
        mg = next(k for k, v in dictionary.items() if v.get('measurement') == 'Ca (mg/dL)' and k.endswith('__latest'))
        mmol = next(k for k, v in dictionary.items() if v.get('measurement') == 'Ca (mmol/L)' and k.endswith('__latest'))
        self.assertTrue(np.isnan(features.iloc[-1][mg]))
        self.assertAlmostEqual(features.iloc[-1][mmol], 2.2, places=5)
        self.assertEqual(audit['source_issues']['measurements: conflicting same-date values'], 1)
        self.assertEqual(audit['source_issues']['BioChemRecords: unrecognized measurement key'], 1)

    def test_seizure_linked_to_prediction_visit_is_not_available(self):
        tables = fixture()
        tables['dbo.SeizureEventViewModel'] = pd.DataFrame([dict(Id=1, PatientBaselineViewModelPatientId='0001',
            DateString='2019-01-01', EndDateString='2019-01-31', SeizureCount=3, AsmRecordViewModelId=3)])
        features, _, _, _ = self.build(tables)
        self.assertEqual(features.iloc[-1].seizure_known_days_12m, 0)
        self.assertTrue(np.isnan(features.iloc[-1].seizure_count_lower_bound_12m))
        self.assertEqual(features.iloc[0].history_days_available, 0)

    def test_overlap_is_not_double_counted_and_partial_intervals_are_not_prorated(self):
        events = [dict(Id=i, start=dt.date(2019, 12, 1), end=dt.date(2019, 12, 31), SeizureCount=n, available=dt.date(2019, 12, 31)) for i, n in [(1, 3), (2, 4)]]
        values = seizure_window(events, dt.date(2020, 1, 15), dt.date(2019, 1, 15), dt.date(2020, 1, 14))
        self.assertEqual(values['count_lower_bound'], 4)
        self.assertEqual(values['known_days'], 31)
        partial = seizure_window(events, dt.date(2020, 1, 15), dt.date(2019, 12, 15), dt.date(2020, 1, 14))
        self.assertIsNone(partial['count_lower_bound'])
        self.assertEqual(partial['partial_positive_intervals'], 2)

    def test_duplicate_visit_and_orphan_drug_fail_explicitly(self):
        tables = fixture()
        extra = copy.deepcopy(tables['dbo.AsmRecords'].iloc[[0]])
        extra['Id'] = 4
        tables['dbo.AsmRecords'] = pd.concat([tables['dbo.AsmRecords'], extra])
        with self.assertRaisesRegex(ValueError, 'Same-day'):
            self.build(tables)
        tables = fixture()
        tables['dbo.AsmDrugDetail'].loc[0, 'AsmRecordViewModelId'] = 999
        with self.assertRaisesRegex(ValueError, 'no medication visit'):
            self.build(tables)

    def test_zero_interval_crossing_window_is_observed_zero_not_missing(self):
        events = [dict(Id=1, start=dt.date(2018, 1, 1), end=dt.date(2019, 12, 31),
                       available=dt.date(2019, 12, 31), SeizureCount=0)]
        values = seizure_window(events, dt.date(2020, 1, 15), dt.date(2019, 1, 15), dt.date(2020, 1, 14))
        self.assertEqual(values['count_lower_bound'], 0)
        self.assertGreater(values['known_days'], 0)
        missing = seizure_window([], dt.date(2020, 1, 15), dt.date(2019, 1, 15), dt.date(2020, 1, 14))
        self.assertIsNone(missing['count_lower_bound'])

    def test_confirmed_gender_default_and_explicit_unknown_legend(self):
        tables = fixture()
        tables['dbo.Patients']['Gender'] = ['1', '0']
        tables['dbo.AsmRecords'] = pd.concat([tables['dbo.AsmRecords'], pd.DataFrame([
            dict(Id=4, Date='2020-01-15', PatientBaselineViewModelPatientId='0002')])], ignore_index=True)
        features, _, _, _ = build_features(tables, dt.date(2021, 2, 1), male_code=None)
        self.assertTrue(features.sex_male.isna().all())
        confirmed, visits, dictionary, audit = build_features(tables, dt.date(2021, 2, 1), include_doses=False)
        self.assertTrue(confirmed.loc[visits.patient_id.eq('0001'), 'sex_male'].eq(1).all())
        self.assertTrue(confirmed.loc[visits.patient_id.eq('0002'), 'sex_male'].eq(0).all())
        self.assertFalse(any('_daily_mg' in c or c.startswith('asm_total_ddd') for c in confirmed))
        self.assertEqual(set(confirmed.columns), set(dictionary))
        self.assertFalse(audit['dose_features_included'])
        self.assertEqual(audit['numeric_male_code'], '1')


if __name__ == '__main__':
    unittest.main()
