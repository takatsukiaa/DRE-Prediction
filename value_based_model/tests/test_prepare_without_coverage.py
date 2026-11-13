"""Historical coverage-free preparation preserves the exact cohort and schema."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from build_visit_status import digest,write_json
from feature_schema import FEATURE_SCHEMA
from historical_audit import HISTORICAL_SCHEMA, prepare_without_coverage
from tests.helpers import local_temp_directory
from tests import test_visit_status
from visit_status_workflow import load_status_bundle


class PrepareWithoutCoverageTests(unittest.TestCase):
    def test_derivation_removes_only_coverage_and_preserves_source_bytes(self):
        with local_temp_directory() as directory:
            _,source=test_visit_status.VisitStatusTests().prepare(directory)
            manifest=json.loads((source/'manifest.json').read_text())
            manifest['feature_schema']='recorded_counts_actual_visits_v2'
            for months in (3,6,9):
                path=source/f'features_{months}m.pkl.gz'
                X=pd.read_pickle(path)
                dictionary=json.loads((source/f'feature_dictionary_{months}m.json').read_text())
                for width in sorted({min(6,months),months}):
                    name=f'seizure_coverage_fraction_{width}m'
                    X[name]=np.linspace(0,1,len(X)).astype(np.float32)
                    dictionary[name]=dict(source='synthetic',definition='Historical coverage fraction')
                X.to_pickle(path,compression='gzip')
                write_json(source/f'feature_dictionary_{months}m.json',dictionary)
                manifest['windows'][f'{months}m']['features']=len(X.columns)
            manifest['files']={name:digest(source/name) for name in manifest['files']}
            write_json(source/'manifest.json',manifest)
            hashes={p.name:digest(p) for p in source.iterdir() if p.is_file()}
            before,visits,targets,_,split,_,_=load_status_bundle(source)
            output=Path(directory)/'without_coverage'
            prepare_without_coverage(source,output)
            after,v,t,dictionaries,new_split,new_manifest,provenance=load_status_bundle(output)
            self.assertEqual(new_manifest['feature_schema'],HISTORICAL_SCHEMA)
            self.assertNotEqual(new_manifest['feature_schema'],FEATURE_SCHEMA)
            self.assertEqual(split,new_split)
            pd.testing.assert_frame_equal(visits,v)
            pd.testing.assert_frame_equal(targets,t)
            for months in (3,6,9):
                keep=[c for c in before[months] if not c.startswith('seizure_coverage_fraction_')]
                pd.testing.assert_frame_equal(before[months][keep],after[months])
                self.assertEqual(list(after[months]),list(dictionaries[months]))
            for name,h in hashes.items():
                self.assertEqual(digest(source/name),h)
            self.assertEqual(provenance['source_feature_manifest_sha256'],hashes['manifest.json'])
            with self.assertRaises(FileExistsError):
                prepare_without_coverage(source,output)


if __name__=='__main__':
    unittest.main()
