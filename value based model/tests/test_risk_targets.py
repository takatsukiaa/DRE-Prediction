import datetime as dt
import unittest

import numpy as np
import pandas as pd

from risk_targets import build_switch_targets, future_assessments, risk_percentages


def monthly(values):
    return pd.DataFrame([dict(PatientId='0001', MonthStart=date[:7] + '-01', AssessmentDate=date, IsDRE=status) for date, status in values])


def visit(state=0):
    return pd.DataFrame([dict(patient_id='0001', prediction_date='2020-01-15', current_dre=state)])


class SwitchTargetTests(unittest.TestCase):
    def test_reversal_before_endpoint_still_counts_as_switch(self):
        source = monthly([('2019-12-31', 0), ('2020-01-31', 1), ('2020-02-29', 0), ('2020-06-30', 0)])
        result = build_switch_targets(visit(), source, dt.date(2021, 2, 1)).iloc[0]
        self.assertEqual(result.switch_within_6m, 1)
        self.assertEqual(result.switch_within_12m, 1)
        self.assertEqual(result.first_opposite_assessment_6m, dt.date(2020, 1, 31))

    def test_dre_to_ndre_is_an_event(self):
        source = monthly([('2019-12-31', 1), ('2020-01-31', 0), ('2020-02-29', 1)])
        result = build_switch_targets(visit(1), source, dt.date(2021, 2, 1)).iloc[0]
        self.assertEqual(result.transition_direction, 'DRE_to_NDRE')
        self.assertEqual(result.switch_within_6m, 1)

    def test_negative_requires_all_expected_assessments(self):
        dates, _ = future_assessments(dt.date(2020, 1, 15), 6)
        complete = monthly([('2019-12-31', 0)] + [(d.isoformat(), 0) for d in dates])
        result = build_switch_targets(visit(), complete, dt.date(2021, 2, 1))
        self.assertEqual(result.iloc[0].switch_within_6m, 0)
        missing = complete.drop(index=2)
        self.assertTrue(pd.isna(build_switch_targets(visit(), missing, dt.date(2021, 2, 1)).iloc[0].switch_within_6m))

    def test_unknown_month_does_not_prevent_a_positive_event(self):
        source = monthly([('2019-12-31', 0), ('2020-01-31', None), ('2020-02-29', 1)])
        self.assertEqual(build_switch_targets(visit(), source, dt.date(2021, 2, 1)).iloc[0].switch_within_6m, 1)

    def test_unknown_origin_is_not_assigned_a_direction(self):
        source = monthly([('2019-12-31', None), ('2020-01-31', 1)])
        result = build_switch_targets(visit(None), source, dt.date(2021, 2, 1)).iloc[0]
        self.assertTrue(pd.isna(result.switch_within_6m))
        self.assertIsNone(result.transition_direction)

    def test_exact_horizon_excludes_next_month_end(self):
        dates, cutoff = future_assessments(dt.date(2020, 1, 15), 6)
        self.assertEqual(cutoff, dt.date(2020, 7, 15))
        self.assertEqual(dates[-1], dt.date(2020, 6, 30))
        self.assertEqual(len(dates), 6)

    def test_month_end_prediction_does_not_use_same_date_assessment(self):
        dates, cutoff = future_assessments(dt.date(2020, 1, 31), 6)
        self.assertEqual(dates[0], dt.date(2020, 2, 29))
        self.assertEqual(dates[-1], dt.date(2020, 7, 31))

    def test_positive_can_be_known_before_entire_horizon_finishes(self):
        source = monthly([('2019-12-31', 0), ('2020-01-31', 1)])
        result = build_switch_targets(visit(), source, dt.date(2020, 2, 1)).iloc[0]
        self.assertEqual(result.switch_within_12m, 1)

    def test_risk_columns_are_direction_specific_and_complementary(self):
        result = risk_percentages([.25, .8, .5], [0, 1, None])
        np.testing.assert_allclose(result['risk_any_switch_pct'][:2], [25, 80])
        self.assertEqual(result['risk_ndre_to_dre_pct'][0], 25)
        self.assertTrue(np.isnan(result['risk_ndre_to_dre_pct'][1]))
        self.assertEqual(result['risk_dre_to_ndre_pct'][1], 80)
        self.assertTrue(np.isnan(result['risk_any_switch_pct'][2]))
        np.testing.assert_allclose((result['risk_any_switch_pct'] + result['risk_no_switch_pct'])[:2], 100)


if __name__ == '__main__':
    unittest.main()
