"""Prepare visit-status features for separate prospective 3/6/9-month history models.

Reads a saved SQL snapshot and creates a reproducible 50/50 patient split.
Records a 12-calendar-month holdout for evaluation.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import pandas as pd

from longitudinal_features import add_months, build_features, day
from feature_schema import FEATURE_SCHEMA, validate_active_features
from sql_snapshot import digest, read_snapshot
from visit_status_targets import HISTORY_WINDOWS, TARGET_DEFINITION, build_visit_status_targets

ROOT = Path(__file__).resolve().parent
BUNDLE_FORMAT = 3


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def validate_split(split, patient_ids):
    train, test = split['train'], split['test']
    if not train or not test or len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Patient split must contain unique patients in both groups')
    if set(train) & set(test) or set(train + test) != set(patient_ids):
        raise ValueError('Patient split must partition the snapshot patient population exactly')


def make_patient_split(patient_ids, seed=42):
    """Split sorted snapshot IDs without consulting outcomes or visit counts."""
    ids = sorted(str(pid) for pid in patient_ids)
    if len(ids) < 2:
        raise ValueError('Patient splitting requires at least two patients')
    if len(ids) != len(set(ids)):
        raise ValueError('Patient splitting requires unique patient IDs')
    random.Random(seed).shuffle(ids)
    midpoint = len(ids) // 2
    split = dict(train=sorted(ids[:midpoint]), test=sorted(ids[midpoint:]))
    validate_split(split, ids)
    return split


def temporal_holdout(as_of):
    """Anchor the holdout to the snapshot's Asia/Taipei extraction date."""
    return dict(months=12, start_inclusive=add_months(as_of, -12).isoformat(),
                end_inclusive=as_of.isoformat())


def visit_periods(dates, holdout):
    start = dt.date.fromisoformat(holdout['start_inclusive'])
    end = dt.date.fromisoformat(holdout['end_inclusive'])
    parsed = dates.map(day)
    if parsed.isna().any() or parsed.gt(end).any():
        raise ValueError('Visit dates must be valid and on or before the snapshot date')
    return parsed.ge(start).map({True: 'recent', False: 'historical'})


def build_status_bundle(snapshot_dir, output_dir, male_code='1', include_doses=True, progress=print, *, seed=42):
    snapshot_dir, output = Path(snapshot_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError('Output already exists; choose a new visit-status bundle directory')
    snapshot = read_snapshot(snapshot_dir)
    split = make_patient_split(snapshot.tables['dbo.Patients'].PatientId, seed)
    if progress:
        progress(f'Created patient split (seed={seed}): {len(split["train"])} training, {len(split["test"])} test patients')
    as_of = dt.datetime.fromisoformat(snapshot.schema['extracted_at_utc']).astimezone(ZoneInfo('Asia/Taipei')).date()
    holdout = temporal_holdout(as_of)
    if progress:
        progress(f'Recent evaluation window: {holdout["start_inclusive"]} through {holdout["end_inclusive"]} inclusive')
    output.mkdir(parents=True, exist_ok=False)
    assignment = {pid: part for part in ('train', 'test') for pid in split[part]}
    summaries, common_visits, targets = {}, None, None
    for months in HISTORY_WINDOWS:
        if progress:
            progress(f'Building {months}-month visit-status history starting 12 months before prediction')
        X, visits, dictionary, audit = build_features(snapshot.tables, as_of, progress=progress,
            male_code=male_code, include_doses=include_doses, history_months=months, include_status_history=False)
        validate_active_features(X.columns, months)
        visits = visits[['patient_id', 'prediction_date', 'source_asm_record_id']].copy()
        visits['split'] = visits.patient_id.map(assignment)
        visits['period'] = visit_periods(visits.prediction_date, holdout)
        if common_visits is None:
            common_visits = visits
            targets = build_visit_status_targets(visits, snapshot.tables, as_of)
            visits.to_json(output / 'visits.jsonl', orient='records', lines=True, date_format='iso')
            targets.to_json(output / 'status_targets.jsonl', orient='records', lines=True, date_format='iso')
            audit_records = audit.pop('eeg_report_audit')
            pd.DataFrame(audit_records).to_json(output / 'eeg_report_audit.jsonl', orient='records', lines=True)
        elif not visits.equals(common_visits):
            raise ValueError('History-window visit alignment mismatch')
        audit.pop('eeg_report_audit', None)
        X.to_pickle(output / f'features_{months}m.pkl.gz', compression='gzip')
        write_json(output / f'feature_dictionary_{months}m.json', dictionary)
        write_json(output / f'audit_{months}m.json', audit)
        summaries[f'{months}m'] = dict(features=len(X.columns), visits=len(X),
            start_months_before_prediction=12, end_months_before_prediction=12 - months)
    write_json(output / 'patient_split.json', split)
    counts = {part: dict(visits=int(common_visits['split'].eq(part).sum()),
        dre=int((targets.status_at_visit.eq(1) & common_visits['split'].eq(part)).sum()),
        ndre=int((targets.status_at_visit.eq(0) & common_visits['split'].eq(part)).sum()),
        unknown=int((targets.status_at_visit.isna() & common_visits['split'].eq(part)).sum())) for part in ('train', 'test')}
    code = [Path(__file__), ROOT / 'feature_schema.py', ROOT / 'visit_status_targets.py', ROOT / 'longitudinal_features.py', ROOT / 'eeg_features.py',
            ROOT / 'sql_snapshot.py', ROOT.parent / 'Database/dre_onset_analysis/monthly_dre.py',
            ROOT.parent / 'Database/dre_onset_analysis/analyze_onset.py', ROOT.parent / 'docx_parsers/docx_parser_eeg_timeline.py']
    write_json(output / 'provenance.json', dict(snapshot=str(snapshot_dir.resolve()),
        snapshot_manifest_sha256=digest(snapshot_dir / 'manifest.json'),
        snapshot_as_of_taipei=as_of.isoformat(), temporal_holdout=holdout,
        split_generation=dict(method='seeded shuffle of sorted snapshot patient IDs; no outcome stratification',
            seed=seed, train_patients=len(split['train']), test_patients=len(split['test'])),
        target_definition=TARGET_DEFINITION,
        feature_schema=FEATURE_SCHEMA,
        target_seizure_window_months=12, history_anchor_months=12,
        predictor_timing='[visit minus 12 months, visit minus (12 minus history months)); availability must precede window end',
        code_sha256={p.relative_to(ROOT.parent).as_posix(): digest(p) for p in code}))
    manifest = dict(format_version=BUNDLE_FORMAT, task='visit_status', feature_schema=FEATURE_SCHEMA,
        history_windows_months=list(HISTORY_WINDOWS), history_anchor_months=12,
        rows=len(common_visits), windows=summaries, label_counts=counts, temporal_holdout=holdout,
        files={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / 'manifest.json', manifest)
    if progress:
        progress('Saved visit-status bundle: ' + str(output.resolve()))
        progress('Visit label counts: ' + json.dumps(counts))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, default=ROOT / 'data/sql_snapshots/20261019')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the reproducible 50/50 patient split')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--male-code', choices=['0', '1'], default='1')
    parser.add_argument('--dose-features', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    stamp = dt.datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S')
    output = args.output_dir or ROOT / 'data/features' / f'visit_status_{stamp}'
    build_status_bundle(args.snapshot_dir, output, male_code=args.male_code, include_doses=args.dose_features,
                        progress=lambda message: print(message, flush=True), seed=args.seed)


if __name__ == '__main__':
    main()
