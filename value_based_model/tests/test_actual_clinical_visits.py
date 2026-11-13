"""Medication comparisons require distinct explicitly recorded clinic visits."""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from feature_schema import medication_comparison_feature, validate_active_features
from longitudinal_features import FK, actual_clinical_visit, build_features
from tests.helpers import drug, fixture


def clinical_fixture():
    tables = fixture()
    rows = [(1,'2019-02-01','1',100),(2,'2019-03-01','-',999),
        (3,'2019-04-01','0',200),(4,'2019-05-01','-',888),
        (5,'2019-08-01','1',300),(6,'2019-09-01',None,777),
        (7,'2020-01-15','1',400)]
    tables['dbo.AsmRecords'] = pd.DataFrame([dict(Id=i, Date=d, IsWuVisit=flag, **{FK:'0001'}) for i,d,flag,dose in rows])
    tables['dbo.AsmDrugDetail'] = pd.DataFrame([drug('Lamotrigine', dose=dose, record=i) for i,d,flag,dose in rows])
    return tables


class ActualClinicalVisitTests(unittest.TestCase):
    def build(self, tables=None, months=9):
        return build_features(tables or clinical_fixture(), dt.date(2021,2,1),
            history_months=months, include_status_history=False)

    def test_source_legend_includes_both_physicians_and_excludes_unconfirmed_markers(self):
        for flag in ('0','1',0,1,' 1 '):
            self.assertTrue(actual_clinical_visit(flag))
        for flag in ('-',None,pd.NA,'','unknown','2'):
            self.assertFalse(actual_clinical_visit(flag))

    def test_three_month_model_has_no_medication_comparison_predictors(self):
        X,_,dictionary,audit = self.build(months=3)
        self.assertFalse(any(medication_comparison_feature(name) for name in X))
        validate_active_features(X,3)
        self.assertEqual(list(X),list(dictionary))
        self.assertFalse(audit['medication_comparisons_enabled'])
        self.assertEqual(X.iloc[-1].asm_visits_3m,2)
        self.assertIn('asm_drug_count_latest',X)
        self.assertIn('asm_total_ddd_latest',X)

    def test_six_month_pair_ignores_repeated_monthly_records(self):
        X,_,_,audit = self.build(months=6)
        row = X.iloc[-1]
        self.assertEqual(row.asm_visits_6m,2)
        self.assertEqual(row['asm__lamotrigine_daily_mg_delta'],100)
        self.assertEqual(row.asm_previous_visit_gap_days,59)
        self.assertEqual(row.asm_dose_comparable_drugs,1)
        pair=audit['medication_comparison_evidence'][-1]
        self.assertEqual((pair['previous_record_id'],pair['latest_record_id']),('1','3'))

    def test_nine_month_uses_two_actual_visits_even_when_last_record_is_not_a_visit(self):
        X,_,_,audit = self.build()
        row=X.iloc[-1]
        self.assertEqual(row.asm_visits_9m,3)
        self.assertEqual(row.asm_visits_6m,1)
        self.assertEqual(row['asm__lamotrigine_daily_mg'],777)
        self.assertEqual(row['asm__lamotrigine_daily_mg_delta'],100)
        self.assertEqual(row.asm_previous_visit_gap_days,122)
        pair=audit['medication_comparison_evidence'][-1]
        self.assertEqual((pair['previous_record_id'],pair['latest_record_id']),('3','5'))

    def test_nonvisit_doses_do_not_change_comparison_features(self):
        tables=clinical_fixture()
        for months in (6,9):
            before,_,_,_=self.build(tables,months)
            changed={key:value.copy(deep=True) for key,value in tables.items()}
            changed['dbo.AsmDrugDetail'].loc[lambda d:d.AsmRecordViewModelId.isin([2,4,6]),'DailyDose']=99999
            after,_,_,_=self.build(changed,months)
            fields=[c for c in before if medication_comparison_feature(c)]
            pd.testing.assert_series_equal(before.iloc[-1][fields],after.iloc[-1][fields])

    def test_insufficient_actual_visits_remains_missing_without_searching_outside_window(self):
        tables=clinical_fixture()
        tables['dbo.AsmRecords'].loc[lambda r:r.Id.isin([3,5]),'IsWuVisit']='-'
        for months in (6,9):
            X,_,_,_=self.build(tables,months)
            row=X.iloc[-1]
            self.assertEqual(row.asm_pair_available,0)
            self.assertTrue(np.isnan(row.asm_dose_comparable_drugs))
            self.assertTrue(np.isnan(row['asm__lamotrigine_daily_mg_delta']))

    def test_unknown_actual_visit_regimen_prevents_skipping_to_older_known_visit(self):
        tables=clinical_fixture()
        tables['dbo.AsmDrugDetail']=tables['dbo.AsmDrugDetail'].loc[lambda r:r.AsmRecordViewModelId.ne(5)]
        X,_,_,audit=self.build(tables)
        self.assertEqual(X.iloc[-1].asm_pair_available,0)
        self.assertTrue(np.isnan(X.iloc[-1].asm_dose_comparable_drugs))
        self.assertFalse(audit['medication_comparison_evidence'][-1]['pair_assessable'])

    def test_identical_prescriptions_at_distinct_visits_are_still_a_valid_pair(self):
        tables=clinical_fixture()
        tables['dbo.AsmDrugDetail']['DailyDose']=100
        X,_,_,_=self.build(tables)
        self.assertEqual(X.iloc[-1].asm_dose_comparable_drugs,1)
        self.assertEqual(X.iloc[-1].asm_regimen_changed,0)
        self.assertEqual(X.iloc[-1]['asm__lamotrigine_daily_mg_delta'],0)


if __name__=='__main__':
    unittest.main()
