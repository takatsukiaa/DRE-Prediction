"""Stage boundaries and prepared-bundle integrity; no live SQL required."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import extract_sql
import build_visit_status
import train
from sql_snapshot import read_snapshot
from tests.helpers import example_snapshot, local_temp_directory


class WorkflowTests(unittest.TestCase):
    def test_preparation_entrypoint_generates_split_with_requested_seed(self):
        args = ['build_visit_status.py', '--snapshot-dir', 'snapshot', '--output-dir', 'bundle', '--seed', '7']
        with patch('sys.argv', args), patch('build_visit_status.build_status_bundle') as build:
            build_visit_status.main()
        self.assertEqual(build.call_args.args, (Path('snapshot'), Path('bundle')))
        self.assertEqual(build.call_args.kwargs['seed'], 7)
        self.assertNotIn('split_file', build.call_args.kwargs)

    def test_training_entrypoint_uses_only_prepared_bundle(self):
        args = ['train.py', '--feature-dir', 'prepared', '--output-dir', 'trained', '--trees', '8']
        with patch('sys.argv', args), patch('visit_status_workflow.run_status_training') as fit, redirect_stdout(StringIO()):
            train.main()
        self.assertEqual(fit.call_args.args, (Path('prepared'), Path('trained')))
        self.assertEqual(fit.call_args.kwargs['trees'], 8)
        self.assertFalse(hasattr(train, 'save_bundle'))

    def test_extraction_writes_snapshot_after_connection_is_closed(self):
        with local_temp_directory() as directory:
            output = Path(directory) / 'snapshot'
            args = ['extract_sql.py', '--output-dir', str(output)]
            with patch('sys.argv', args), patch('extract_sql.connect_epilepsydb') as connect, \
                 patch('extract_sql.extract_snapshot', return_value=example_snapshot()), redirect_stdout(StringIO()):
                extract_sql.main()
                connect.return_value.__exit__.assert_called_once()
            saved = read_snapshot(output)
            self.assertEqual(len(saved.tables), 2)
            self.assertEqual(saved.tables['dbo.Patients'].PatientId.tolist(), ['0001', '0002'])

    def test_extract_command_can_verify_a_saved_snapshot(self):
        from sql_snapshot import write_snapshot

        with local_temp_directory() as directory:
            snapshot_path = Path(directory) / 'snapshot'
            snapshot = example_snapshot()
            write_snapshot(snapshot, snapshot_path)
            args = ['extract_sql.py', '--verify', str(snapshot_path), '--server', 'fixture']
            with patch('sys.argv', args), patch('extract_sql.connect_epilepsydb') as connect, \
                 patch('extract_sql.extract_snapshot', return_value=snapshot), redirect_stdout(StringIO()):
                extract_sql.main()
                connect.assert_called_once_with('fixture')
            report = json.loads((snapshot_path / 'verification.json').read_text(encoding='utf-8'))
            self.assertTrue(report['passed'])
            self.assertEqual(report['tables_verified'], 2)


if __name__ == '__main__':
    unittest.main()
