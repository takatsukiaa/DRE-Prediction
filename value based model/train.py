r"""Build historical features or train any-switch risk forests on a saved split.

Run from the project root:
    .\.venv\Scripts\python.exe "value based model/train.py"

Default mode builds un-imputed features. --train-forest fits directional risk
models using a saved feature bundle. Neither mode connects to the live database.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from longitudinal_features import build_features
from sql_snapshot import read_snapshot

ROOT = Path(__file__).resolve().parent


def make_patient_split(patient_ids, monthly, seed=42):
    """Split unique patients once; never split their individual visits."""
    from collections import Counter, defaultdict
    from sklearn.model_selection import train_test_split
    ids, states = sorted(patient_ids), defaultdict(set)
    for patient, status in zip(monthly['PatientId'], monthly['IsDRE']):
        if not pd.isna(status):
            states[str(patient)].add(int(status))
    strata = ['ever_positive' if 1 in states[p] else 'observed_negative' if 0 in states[p] else 'all_unknown' for p in ids]
    counts = Counter(strata)
    stratify = strata if min(counts.values()) >= 2 and len(ids) // 2 >= len(counts) else None
    training, test = train_test_split(ids, test_size=0.5, random_state=seed, stratify=stratify)
    assert not set(training) & set(test)
    return dict(seed=seed, unit='patient', train=sorted(training), test=sorted(test),
                stratification='stored monthly history' if stratify is not None else 'none: small stratum')


def save_bundle(features, visits, dictionary, audit, split, output, snapshot_dir):
    output.mkdir(parents=True, exist_ok=False)
    features.to_pickle(output / 'features.pkl.gz', compression='gzip')
    visits.to_json(output / 'visits.jsonl', orient='records', lines=True, date_format='iso')
    pd.DataFrame(audit.get('eeg_report_audit', [])).to_json(output / 'eeg_report_audit.jsonl', orient='records', lines=True)
    artifacts = {'feature_dictionary.json': dictionary, 'audit.json': {k: v for k, v in audit.items() if k != 'eeg_report_audit'}, 'patient_split.json': split,
                 'provenance.json': dict(snapshot=str(snapshot_dir.resolve()),
                    snapshot_manifest_sha256=hashlib.sha256((snapshot_dir / 'manifest.json').read_bytes()).hexdigest(),
                    builder_sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), ROOT / 'longitudinal_features.py', ROOT / 'eeg_features.py',
                                  ROOT.parent / 'docx_parsers/docx_parser_eeg_timeline.py')},
                    pandas=pd.__version__, numpy=np.__version__,
                    row_alignment='features.iloc[i] corresponds to visits.jsonl line i+1')}
    for name, content in artifacts.items():
        (output / name).write_text(json.dumps(content, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, default=ROOT / 'data/sql_snapshots/epilepsydb_20261016')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--male-code', choices=['0', '1'], default='1',
                        help='Numeric Gender code for male; confirmed project default: 1=male, 0=female')
    parser.add_argument('--dose-features', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--train-forest', action='store_true', help='Fit any-switch risk models instead of rebuilding features')
    parser.add_argument('--feature-dir', type=Path, default=ROOT / 'data/features/previsit_20261016_eeg_v1')
    parser.add_argument('--panel', choices=['core', 'all'], default='all')
    parser.add_argument('--top-features', type=int, default=30, help='Source features ranked on training patients only')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=1000, help='Patient-level metric resamples; no model refitting')
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = dt.datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S')
        args.output_dir = ROOT / 'results' / f'rf_any_switch_{stamp}' if args.train_forest else ROOT / 'data/features/previsit_20261016'
    if args.train_forest:
        from forest_workflow import run_training
        run_training(args.feature_dir, args.output_dir, snapshot_dir=args.snapshot_dir, panel=args.panel,
                     seed=args.seed, trees=args.trees, jobs=args.jobs, bootstrap=args.bootstrap,
                     top_features=args.top_features, progress=lambda s: print(s, flush=True))
        return
    if args.output_dir.exists():
        parser.error('Output already exists; choose a new --output-dir to preserve previous runs')
    snapshot = read_snapshot(args.snapshot_dir)
    as_of = dt.datetime.fromisoformat(snapshot.schema['extracted_at_utc']).astimezone(ZoneInfo('Asia/Taipei')).date()
    features, visits, dictionary, audit = build_features(snapshot.tables, as_of,
        progress=lambda s: print(s, flush=True), male_code=args.male_code, include_doses=args.dose_features)
    split = make_patient_split(snapshot.tables['dbo.Patients']['PatientId'].astype(str), snapshot.tables['dbo.PatientMonthlyDreStatus'], args.seed)
    visits['split'] = visits.patient_id.map({pid: part for part in ('train', 'test') for pid in split[part]})
    audit['split_patient_counts'] = {part: len(split[part]) for part in ('train', 'test')}
    for horizon in (6, 12):
        observed = visits[f'target_dre_{horizon}m'].notna()
        counts = visits.loc[observed].groupby('patient_id')['patient_id'].transform('size')
        visits[f'sample_weight_{horizon}m'] = np.nan
        visits.loc[observed, f'sample_weight_{horizon}m'] = 1 / counts
        audit['outcomes'][f'{horizon}m']['contributing_patients_by_split'] = {
            part: int(visits.loc[visits[f'target_dre_{horizon}m'].notna() & visits['split'].eq(part), 'patient_id'].nunique())
            for part in ('train', 'test')}
        pairs = visits.loc[observed & visits.current_dre.notna()].groupby(['current_dre', f'target_dre_{horizon}m']).size()
        audit['outcomes'][f'{horizon}m']['current_to_future_counts'] = {
            f'{int(current)}->{int(future)}': int(n) for (current, future), n in pairs.items()}
    audit['snapshot_directory'] = str(args.snapshot_dir.resolve())
    save_bundle(features, visits, dictionary, audit, split, args.output_dir, args.snapshot_dir)
    print(json.dumps({k: v for k, v in audit.items() if k not in {'source_issues', 'largest_stored_daily_doses_for_review', 'eeg_report_audit'}}, indent=2))
    print('Saved: ' + str(args.output_dir.resolve()))


if __name__ == '__main__':
    main()
