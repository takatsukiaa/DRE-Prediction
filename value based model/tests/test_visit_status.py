"""Visit-status target timing, bounded histories, and stage integrity."""
import copy
import datetime as dt
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from build_visit_status import build_status_bundle, make_patient_split, validate_split
from longitudinal_features import FK, build_features, history_bounds
from sql_snapshot import DatabaseSnapshot
from tests.helpers import drug, fixture, local_temp_directory
from visit_status_targets import build_visit_status_targets
from visit_status_workflow import load_status_bundle, run_status_training

AS_OF = dt.date(2021, 2, 1)


def status_fixture():
    tables = fixture()
    tables['dbo.PatientMonthlyDreStatus'] = tables['dbo.PatientMonthlyDreStatus'].iloc[:0].copy()
    # Earlier visits have two ASMs, while the January visit has one.
    tables['dbo.SeizureEventViewModel'] = pd.DataFrame([
        dict(Id=1, **{FK: '0001'}, DateString='2019-10-01', EndDateString='2019-10-31',
             SeizureCount=1, AsmRecordViewModelId=1)])
    return tables


class VisitStatusTests(unittest.TestCase):
    def test_same_day_medications_define_target_but_never_predictors(self):
        tables = status_fixture()
        before, visits, _, _ = build_features(tables, AS_OF, include_status_history=False, history_months=3)
        target_before = build_visit_status_targets(visits, tables, AS_OF)
        self.assertEqual(target_before.status_at_visit.tolist(), [1, 1, 0])
        tables['dbo.AsmDrugDetail'] = pd.concat([tables['dbo.AsmDrugDetail'], pd.DataFrame([drug('Lamotrigine', record=3)])])
        after, _, _, _ = build_features(tables, AS_OF, include_status_history=False, history_months=3)
        target_after = build_visit_status_targets(visits, tables, AS_OF)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(target_after.iloc[-1].status_at_visit, 1)

    def test_seizure_evidence_linked_to_later_visit_is_unavailable_at_target(self):
        tables = status_fixture()
        tables['dbo.SeizureEventViewModel']['AsmRecordViewModelId'] = 3
        _, visits, _, _ = build_features(tables, AS_OF, include_status_history=False)
        target = build_visit_status_targets(visits, tables, AS_OF)
        self.assertTrue(pd.isna(target.iloc[0].status_at_visit))
        self.assertTrue(pd.isna(target.iloc[1].status_at_visit))
        self.assertEqual(target.iloc[-1].status_at_visit, 0)

    def test_monthly_and_final_labels_do_not_enter_status_predictors(self):
        tables = status_fixture()
        tables['dbo.PatientMonthlyDreStatus'] = fixture()['dbo.PatientMonthlyDreStatus']
        before, _, _, _ = build_features(tables, AS_OF, include_status_history=False)
        tables['dbo.PatientMonthlyDreStatus']['IsDRE'] = 0
        tables['dbo.Patients']['IsDRE'] = 1
        after, _, dictionary, _ = build_features(tables, AS_OF, include_status_history=False)
        pd.testing.assert_frame_equal(before, after)
        self.assertFalse(any(c.startswith(('current_', 'monthly_', 'target_')) for c in dictionary))
        self.assertNotIn('history_days_available', dictionary)

    def test_latest_completed_month_fills_only_unknown_visit_status_without_markers(self):
        tables = status_fixture()
        tables['dbo.SeizureEventViewModel']['AsmRecordViewModelId'] = 3
        tables['dbo.PatientMonthlyDreStatus'] = pd.DataFrame([
            dict(PatientId='0001', MonthStart='2019-10-01', AssessmentDate='2019-10-31', IsDRE=False),
            dict(PatientId='0001', MonthStart='2019-11-01', AssessmentDate='2019-11-30', IsDRE=True),
            dict(PatientId='0001', MonthStart='2019-12-01', AssessmentDate='2019-12-31', IsDRE=True)])
        _, visits, _, _ = build_features(tables, AS_OF, include_status_history=False)
        filled = build_visit_status_targets(visits, tables, AS_OF)
        self.assertEqual(filled.status_at_visit.tolist(), [0, 1, 0])
        # The known January NDRE label takes precedence over monthly DRE.
        self.assertFalse(any('fallback' in name or name == 'label_source' for name in filled))
        self.assertFalse(filled.target_reason.str.contains('fallback', case=False).any())

    def test_unknown_latest_month_is_not_skipped_for_an_older_known_month(self):
        tables = status_fixture()
        tables['dbo.SeizureEventViewModel']['AsmRecordViewModelId'] = 3
        tables['dbo.PatientMonthlyDreStatus'] = pd.DataFrame([
            dict(PatientId='0001', MonthStart='2019-10-01', AssessmentDate='2019-10-31', IsDRE=1),
            dict(PatientId='0001', MonthStart='2019-11-01', AssessmentDate='2019-11-30', IsDRE=None)])
        _, visits, _, _ = build_features(tables, AS_OF)
        targets = build_visit_status_targets(visits, tables, AS_OF)
        self.assertEqual(targets.iloc[0].status_at_visit, 1)
        self.assertTrue(pd.isna(targets.iloc[1].status_at_visit))

    def test_same_day_and_future_monthly_assessments_cannot_fill_an_unknown_visit(self):
        tables = status_fixture()
        tables['dbo.SeizureEventViewModel']['AsmRecordViewModelId'] = 3
        tables['dbo.AsmRecords'].loc[lambda r: r.Id.eq(2), 'Date'] = '2019-11-30'
        tables['dbo.PatientMonthlyDreStatus'] = pd.DataFrame([
            dict(PatientId='0001', MonthStart='2019-11-01', AssessmentDate='2019-11-30', IsDRE=1),
            dict(PatientId='0001', MonthStart='2019-12-01', AssessmentDate='2019-12-31', IsDRE=0)])
        _, visits, _, _ = build_features(tables, AS_OF)
        targets = build_visit_status_targets(visits, tables, AS_OF)
        self.assertTrue(pd.isna(targets.iloc[0].status_at_visit))
        self.assertTrue(pd.isna(targets.iloc[1].status_at_visit))

    def test_short_windows_share_annual_start_and_exclude_later_evidence(self):
        tables = status_fixture()
        dates = ['2019-01-14', '2019-01-15', '2019-02-15', '2019-04-15',
                 '2019-07-15', '2019-10-15', '2019-12-01', '2020-01-15']
        tables['dbo.AsmRecords'] = pd.DataFrame([
            dict(Id=i, Date=date, **{FK: '0001'}) for i, date in enumerate(dates, 1)])
        tables['dbo.AsmDrugDetail'] = pd.DataFrame([
            drug('Lamotrigine', dose=i * 100, record=i) for i in range(1, len(dates) + 1)])
        tables['dbo.BioChemRecords'] = pd.DataFrame([
            dict(Id=i, **{FK: '0001'}, Date=date, DynamicDataJson=json.dumps({'Na (mmol/L)': i * 10}))
            for i, date in enumerate(dates, 1)])
        tables['dbo.MmseRecords'] = pd.DataFrame([
            dict(Id=i, **{FK: '0001'}, Date=date, TotalScore=i * 10) for i, date in enumerate(dates, 1)])
        tables['dbo.EegRecords'] = pd.DataFrame([
            dict(Id=i, **{FK: '0001'}, Date=date, Note='EEG diagnosis: Normal.') for i, date in enumerate(dates, 1)])
        tables['dbo.BrainImages'] = pd.DataFrame([
            dict(Id=i, **{FK: '0001'}, ExaminationDate=date) for i, date in enumerate(dates, 1)])
        tables['dbo.SeizureEventViewModel'] = pd.DataFrame([
            dict(Id=i, **{FK: '0001'}, DateString=date, EndDateString=date,
                 SeizureCount=1, AsmRecordViewModelId=i) for i, date in enumerate(dates, 1)])
        for months, count in [(3, 2), (6, 3), (9, 4), (12, 6)]:
            with self.subTest(months=months):
                features, visits, dictionary, audit = build_features(tables, AS_OF, history_months=months, include_status_history=False)
                last = features.iloc[-1]
                self.assertEqual(visits.iloc[-1].history_start, dt.date(2019, 1, 15))
                self.assertEqual(visits.iloc[-1].history_end_exclusive, history_bounds(dt.date(2020, 1, 15), months)[1])
                self.assertEqual(audit['history_anchor_months'], 12)
                self.assertEqual(last[f'asm_visits_{months}m'], count)
                self.assertEqual(last[f'eeg_encoded_dates_{months}m'], count)
                self.assertEqual(last[f'brain_image_dates_{months}m'], count)
                self.assertEqual(last[f'seizure_count_lower_bound_{months}m'], count)
                if months == 9:
                    # The recent six months end at October 15, not at prediction.
                    for feature in ('asm_visits_6m', 'eeg_dates_6m', 'brain_image_dates_6m',
                                    'seizure_count_lower_bound_6m'):
                        self.assertEqual(last[feature], 2)
                self.assertEqual(last['mmse__totalscore__latest'], (count + 1) * 10)
                self.assertEqual(last['mmse__totalscore__delta'], 10)
                self.assertEqual(last['asm__lamotrigine_daily_mg_delta'], 100)
                lab = next(k for k, v in dictionary.items() if v.get('measurement') == 'Na (mmol/L)' and k.endswith('__latest'))
                self.assertEqual(last[lab], (count + 1) * 10)
                changed = copy.deepcopy(tables)
                excluded = [1] + list(range(count + 2, len(dates) + 1))
                changed['dbo.AsmDrugDetail'].loc[lambda r: r.AsmRecordViewModelId.isin(excluded), 'DailyDose'] = 99999
                changed['dbo.BioChemRecords'].loc[lambda r: r.Id.isin(excluded), 'DynamicDataJson'] = '{"Na (mmol/L)": 999}'
                changed['dbo.MmseRecords'].loc[lambda r: r.Id.isin(excluded), 'TotalScore'] = 999
                changed['dbo.EegRecords'].loc[lambda r: r.Id.isin(excluded), 'Note'] = 'Abnormal. Frequent spikes over the left temporal region.'
                changed['dbo.SeizureEventViewModel'].loc[lambda r: r.Id.isin(excluded), 'SeizureCount'] = 999
                after, _, _, _ = build_features(changed, AS_OF, history_months=months, include_status_history=False)
                pd.testing.assert_series_equal(last, after.iloc[-1])

    def test_nine_month_recent_summaries_share_clipped_calendar_cutoff(self):
        tables = status_fixture()
        dates = ['2019-08-28', '2019-08-29', '2020-05-31']
        tables['dbo.AsmRecords'] = pd.DataFrame([
            dict(Id=i, Date=date, **{FK: '0001'}) for i, date in enumerate(dates, 1)])
        tables['dbo.EegRecords'] = pd.DataFrame([
            dict(Id=i, Date=date, Note='EEG diagnosis: Normal.', **{FK: '0001'})
            for i, date in enumerate(dates, 1)])
        features, visits, _, _ = build_features(tables, AS_OF, history_months=9, include_status_history=False)
        self.assertEqual(visits.iloc[-1].history_end_exclusive, dt.date(2020, 2, 29))
        # Recent six-month summaries start on August 29 after clipping the end.
        for feature in ('asm_visits_6m', 'eeg_dates_6m', 'eeg_encoded_dates_6m'):
            self.assertEqual(features.iloc[-1][feature], 1)

    def test_late_linked_seizures_and_embedded_eeg_cannot_enter_closed_windows(self):
        for months, end in [(3, '2019-04-15'), (6, '2019-07-15'), (9, '2019-10-15')]:
            tables = status_fixture()
            tables['dbo.AsmRecords'].loc[lambda r: r.Id.eq(1), 'Date'] = end
            tables['dbo.SeizureEventViewModel'] = pd.DataFrame([
                dict(Id=1, **{FK: '0001'}, DateString='2019-02-01', EndDateString='2019-02-28',
                     SeizureCount=10, AsmRecordViewModelId=1)])
            tables['dbo.EegRecords'] = pd.DataFrame([
                dict(Id=1, **{FK: '0001'}, Date=end,
                     Note='2019-02-01 EEG diagnosis: Abnormal. Frequent spikes over the left temporal region.')])
            features, _, _, _ = build_features(tables, AS_OF, history_months=months, include_status_history=False)
            self.assertTrue(np.isnan(features.iloc[-1][f'seizure_count_lower_bound_{months}m']))
            self.assertEqual(features.iloc[-1][f'eeg_encoded_dates_{months}m'], 0)

    def test_calendar_bounds_preserve_prediction_anniversaries(self):
        for prediction in [dt.date(2020, 2, 29), dt.date(2020, 5, 31), dt.date(2021, 1, 31)]:
            bounds = [history_bounds(prediction, m) for m in (3, 6, 9, 12)]
            self.assertEqual(len({start for start, _ in bounds}), 1)
            self.assertEqual(bounds[-1][1], prediction)
            self.assertTrue(all(a[1] < b[1] for a, b in zip(bounds, bounds[1:])))
        self.assertEqual(history_bounds(dt.date(2020, 5, 31), 3), (dt.date(2019, 5, 31), dt.date(2019, 8, 31)))
        self.assertEqual(history_bounds(dt.date(2020, 5, 31), 9), (dt.date(2019, 5, 31), dt.date(2020, 2, 29)))
        with self.assertRaises(ValueError):
            history_bounds(dt.date(2020, 1, 1), 4)

    def test_exact_target_window_ignores_old_partial_and_future_positive_intervals(self):
        tables = status_fixture()
        # Give the January visit two ASMs so seizure evidence decides its target.
        tables['dbo.AsmDrugDetail'] = pd.concat([tables['dbo.AsmDrugDetail'], pd.DataFrame([drug('Lamotrigine', record=3)])])
        tables['dbo.SeizureEventViewModel'] = pd.DataFrame([
            dict(Id=1, **{FK: '0001'}, DateString='2019-01-01', EndDateString='2019-01-15', SeizureCount=1, AsmRecordViewModelId=1),
            dict(Id=2, **{FK: '0001'}, DateString='2020-01-16', EndDateString='2020-01-16', SeizureCount=1, AsmRecordViewModelId=3)])
        _, visits, _, _ = build_features(tables, AS_OF)
        target = build_visit_status_targets(visits, tables, AS_OF)
        self.assertEqual(target.iloc[-1].target_window_start, dt.date(2019, 1, 16))
        self.assertTrue(pd.isna(target.iloc[-1].status_at_visit))
        tables['dbo.SeizureEventViewModel'].loc[0, 'DateString'] = '2019-01-16'
        tables['dbo.SeizureEventViewModel'].loc[0, 'EndDateString'] = '2019-01-16'
        self.assertEqual(build_visit_status_targets(visits, tables, AS_OF).iloc[-1].status_at_visit, 1)

    def prepare(self, directory, seed=42):
        source, output = Path(directory) / 'source', Path(directory) / 'bundle'
        source.mkdir()
        (source / 'manifest.json').write_text('{}')
        tables = status_fixture()
        # Duplicate the synthetic patient's visits for an independent test patient.
        tables['dbo.AsmRecords'] = pd.concat([tables['dbo.AsmRecords'], pd.DataFrame([
            dict(Id=i + 10, Date=date, **{FK: '0002'}) for i, date in enumerate(['2019-11-01', '2019-12-01', '2020-01-15'], 1)])])
        tables['dbo.AsmDrugDetail'] = pd.concat([tables['dbo.AsmDrugDetail'], pd.DataFrame([
            drug('Lamotrigine', record=11), drug('Valproate', record=11), drug('Lamotrigine', record=12), drug('Lamotrigine', record=13)])])
        tables['dbo.SeizureEventViewModel'] = pd.concat([tables['dbo.SeizureEventViewModel'], pd.DataFrame([
            dict(Id=2, **{FK: '0002'}, DateString='2019-10-01', EndDateString='2019-10-31', SeizureCount=1, AsmRecordViewModelId=11)])])
        snapshot = DatabaseSnapshot(tables, dict(extracted_at_utc='2021-02-01T00:00:00+00:00'), [])
        with patch('build_visit_status.read_snapshot', return_value=snapshot):
            build_status_bundle(source, output, progress=None, seed=seed)
        return source, output

    def test_prepared_bundle_trains_without_source_and_models_reload(self):
        with local_temp_directory() as directory:
            source, bundle = self.prepare(directory)
            (source / 'manifest.json').unlink()
            frames, visits, targets, _, _, _, _ = load_status_bundle(bundle)
            self.assertEqual(visits.patient_id.tolist(), ['0001'] * 3 + ['0002'] * 3)
            result = Path(directory) / 'results'
            with patch('build_visit_status.read_snapshot', side_effect=AssertionError('Training opened source')):
                report = run_status_training(bundle, result, trees=8, jobs=1, bootstrap=0, top_features=3, progress=None)
            predictions = pd.read_json(result / 'heldout_status_predictions.jsonl', lines=True)
            self.assertEqual(report['history_windows_months'], [3, 6, 9, 12])
            self.assertEqual(len(report['tasks']), 4)
            self.assertEqual(len(predictions), 3)
            saved_report = (result / 'report.md').read_text(encoding='utf-8')
            roc = report['roc_comparison']
            self.assertGreater((result / roc['plot']).stat().st_size, 1000)
            self.assertIn(roc['plot'], saved_report)
            curves = json.loads((result / roc['data']).read_text())['curves']
            for months in (3, 6, 9, 12):
                self.assertAlmostEqual(curves[f'{months}m']['roc_auc'],
                    report['tasks'][f'visit_status_history_{months}m']['models']['selected_features']['roc_auc'])
            self.assertEqual(len(list((result / 'confusion_matrices').glob('*.png'))), 12)
            for task, row in report['tasks'].items():
                self.assertEqual(set(report['confusion_matrix_plots'][task]), set(row['models']))
                for path in report['confusion_matrix_plots'][task].values():
                    self.assertGreater((result / path).stat().st_size, 1000)
                    self.assertIn(path, saved_report)
            for months in (3, 6, 9, 12):
                np.testing.assert_allclose(predictions[f'risk_dre_pct_history_{months}m'] + predictions[f'risk_ndre_pct_history_{months}m'], 100)
                model = joblib.load(result / f'models/visit_status_history_{months}m_selected_features.joblib')
                np.testing.assert_allclose(model.predict_dre_probability(frames[months].loc[visits['split'].eq('test')]) * 100, predictions[f'risk_dre_pct_history_{months}m'])
                self.assertEqual(model.history_months, months)
                self.assertEqual(model.history_anchor_months, 12)
                for date, start, end in zip(predictions.prediction_date, predictions[f'history_start_{months}m'], predictions[f'history_end_exclusive_{months}m']):
                    bounds = history_bounds(pd.Timestamp(date).date(), months)
                    self.assertEqual((start, end), tuple(d.isoformat() for d in bounds))
            self.assertTrue(predictions.prediction_scope.eq('heldout_patient').all())

    def test_bundle_without_nine_month_history_requires_rebuilding(self):
        with local_temp_directory() as directory:
            _, bundle = self.prepare(directory)
            path = bundle / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['history_windows_months'] = [3, 6, 12]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'run build_visit_status.py first'):
                load_status_bundle(bundle)

    def test_old_trailing_window_bundle_requires_rebuilding(self):
        with local_temp_directory() as directory:
            _, bundle = self.prepare(directory)
            path = bundle / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['format_version'] = 1
            manifest.pop('history_anchor_months')
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'run build_visit_status.py first'):
                load_status_bundle(bundle)

    def test_status_label_tampering_is_rejected(self):
        with local_temp_directory() as directory:
            _, bundle = self.prepare(directory)
            with (bundle / 'status_targets.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch: status_targets'):
                load_status_bundle(bundle)

    def test_invalid_or_overlapping_patient_split_fails(self):
        with self.assertRaises(ValueError):
            validate_split(dict(train=['a'], test=['a']), ['a', 'b'])
        with self.assertRaises(ValueError):
            validate_split(dict(train=['a', 'a'], test=['b']), ['a', 'b'])
        with self.assertRaises(ValueError):
            validate_split(dict(train=['a'], test=['b']), ['a', 'b', 'c'])

    def test_generated_split_is_reproducible_disjoint_and_order_independent(self):
        ids = [f'{i:04}' for i in range(265)]
        split = make_patient_split(ids)
        self.assertEqual(split, make_patient_split(reversed(ids)))
        self.assertEqual((len(split['train']), len(split['test'])), (132, 133))
        self.assertFalse(set(split['train']) & set(split['test']))
        self.assertEqual(set(split['train'] + split['test']), set(ids))
        self.assertNotEqual(split, make_patient_split(ids, seed=43))

    def test_split_rejects_insufficient_or_duplicate_patients(self):
        for ids in ([], ['a'], ['a', 'a'], ['a', 'b', 'a']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                make_patient_split(ids)

    def test_bundle_generates_and_records_its_own_split_without_external_file(self):
        with local_temp_directory() as directory:
            _, bundle = self.prepare(directory, seed=7)
            self.assertFalse((Path(directory) / 'patient_split.json').exists())
            saved = json.loads((bundle / 'patient_split.json').read_text())
            self.assertEqual(saved, make_patient_split(['0001', '0002'], seed=7))
            provenance = json.loads((bundle / 'provenance.json').read_text())
            self.assertEqual(provenance['split_generation']['seed'], 7)
            self.assertNotIn('split_source', provenance)
            _, visits, _, _, loaded, _, _ = load_status_bundle(bundle)
            self.assertEqual(loaded, saved)
            self.assertTrue(visits.groupby('patient_id')['split'].nunique().eq(1).all())
            for part, ids in saved.items():
                self.assertEqual(set(visits.loc[visits['split'].eq(part), 'patient_id']), set(ids))

    def test_existing_bundle_is_not_overwritten(self):
        with local_temp_directory() as directory, patch('build_visit_status.read_snapshot') as read:
            with self.assertRaises(FileExistsError):
                build_status_bundle('unused', directory)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
