"""Patient resampling, tied scores and undefined NPV in the active comparisons."""
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from historical_audit import cluster_bootstrap, paired_summary


class PatientBootstrapTests(unittest.TestCase):
    def test_bootstrap_matches_patient_resampling_for_auc_ap_brier_and_npv(self):
        y=np.array([0,0,1,1,0,1]); p=np.array([.1,.5,.5,.9,.4,.8])
        ids=np.array(['a','a','b','b','c','c'])
        result=cluster_bootstrap(y,p,ids,25)
        draws=np.random.default_rng(42).multinomial(3,[1/3]*3,size=25)
        for i,draw in enumerate(draws):
            w=np.repeat(draw,2)/2
            if w[y==0].sum() and w[y==1].sum():
                self.assertAlmostEqual(result['roc_auc'][i],roc_auc_score(y,p,sample_weight=w))
                self.assertAlmostEqual(result['average_precision'][i],average_precision_score(y,p,sample_weight=w))
            self.assertAlmostEqual(result['brier'][i],np.average((p-y)**2,weights=w))
            negative=p<.5
            expected=w[(y==0)&negative].sum()/w[negative].sum() if w[negative].sum() else np.nan
            np.testing.assert_allclose(result['npv_at_0_5'][i],expected,equal_nan=True)
            tp=w[(y==1)&~negative].sum(); fp=w[(y==0)&~negative].sum(); fn=w[(y==1)&negative].sum()
            expected_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
            np.testing.assert_allclose(result['f1_at_0_5'][i],expected_f1,equal_nan=True)

    def test_no_negative_predictions_produces_undefined_npv(self):
        result=paired_summary([0,1,0,1],[.7,.8,.6,.9],['a','a','b','b'],[.7,.8,.6,.9],20)
        self.assertIsNone(result['npv_at_0_5'])
        self.assertIsNone(result['bootstrap_95']['npv_at_0_5'])
        self.assertEqual(result['paired_difference_from_baseline']['brier']['bootstrap_95'],[0.,0.])


if __name__=='__main__':
    unittest.main()
