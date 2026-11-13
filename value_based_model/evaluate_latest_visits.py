"""Evaluate saved forests at the latest visits in the 12-month holdout.

Training, feature selection, and the patient split remain fixed. Latest visits
are selected before excluding unknown outcomes; earlier known visits cannot
replace an unknown latest visit.
"""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from build_visit_status import BUNDLE_FORMAT, digest, write_json
from forest_workflow import _fmt, format_interval, performance
from longitudinal_features import HISTORY_WINDOWS, history_bounds, day
from plot_visit_status import write_confusion_matrices, write_roc_comparison
from visit_status_workflow import load_status_bundle


def latest_visit_indices(visits):
    dates = pd.to_datetime(visits.prediction_date, errors='raise')
    if dates.isna().any() or visits.patient_id.isna().any():
        raise ValueError('Latest-visit selection requires patient IDs and valid dates')
    if visits.duplicated(['patient_id', 'prediction_date']).any():
        raise ValueError('Same-day visits need reconciliation')
    ordered = visits.assign(_date=dates).sort_values(['patient_id', '_date'])
    return ordered.groupby('patient_id', sort=False).tail(1).index


def apply_folder_labels(targets, label_root, overrides_file=None, *, eligible_indices=None):
    """Use provided DRE/NDRE folder membership for latest visits only.

    Preserve computed evidence and labels for audit. Unmatched patients retain
    the computed label; conflicting or duplicate folder assignments fail.
    """
    root = Path(label_root).resolve()
    references = {}
    for group, status in (('DRE', 1), ('NDRE', 0)):
        directory = root / group
        if not directory.is_dir():
            raise ValueError('Missing label directory: ' + str(directory))
        for path in sorted(directory.glob('*.docx')):
            match = re.match(r'^(\d{4})(?:\s|$)', path.stem)
            if not match:
                continue
            pid = match.group(1)
            if pid in references:
                raise ValueError('Duplicate folder label for patient ' + pid)
            references[pid] = dict(status=status, source=str(path), sha256=digest(path),
                                   source_type='provided_folder_label', note='Provided ' + group + ' folder label.')
    if not references:
        raise ValueError('No patient labels found in DRE/NDRE folders')
    folder_references = references.copy()
    override_ids, override_provenance = [], None
    if overrides_file is not None:
        path = Path(overrides_file).resolve()
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('scope') != 'latest_recorded_visit' or not isinstance(payload.get('labels'), list):
            raise ValueError('Label overrides must specify latest_recorded_visit scope and a labels list')
        for entry in payload['labels']:
            pid, status, note = entry.get('patient_id'), entry.get('status'), entry.get('source')
            if (not isinstance(pid, str) or not re.fullmatch(r'\d{4}', pid)
                    or status not in ('DRE', 'NDRE') or not isinstance(note, str) or not note.strip()):
                raise ValueError('Each override requires a four-digit patient ID, DRE/NDRE status and source')
            if pid in override_ids or pid not in set(targets.patient_id.astype(str)):
                raise ValueError('Duplicate or unmatched override patient ID: ' + pid)
            override_ids.append(pid)
            references[pid] = dict(status=int(status == 'DRE'), source=str(path), sha256=digest(path),
                                   source_type='user_confirmed_label', note=note)
        override_provenance = dict(file=str(path), sha256=digest(path), patient_ids=override_ids)
    if eligible_indices is not None:
        targets = targets.loc[eligible_indices]
        if override_provenance is not None:
            eligible_patients = set(targets.patient_id.astype(str))
            override_provenance['patient_ids'] = [
                pid for pid in override_ids if pid in eligible_patients]
    result = targets.copy()
    result['automated_status_at_visit'] = result.status_at_visit.copy()
    result['automated_target_reason'] = result.target_reason.copy()
    result['reference_label_source'] = 'computed_visit_status'
    result['reference_label_file'] = pd.Series(None, index=result.index, dtype=object)
    result['reference_label_note'] = pd.Series(None, index=result.index, dtype=object)
    records = []
    for index, row in result.iterrows():
        reference = references.get(str(row.patient_id))
        old = None if pd.isna(row.automated_status_at_visit) else int(row.automated_status_at_visit)
        if reference:
            status = reference['status']
            result.at[index, 'status_at_visit'] = status
            result.at[index, 'target_reason'] = reference['note']
            result.at[index, 'reference_label_source'] = reference['source_type']
            result.at[index, 'reference_label_file'] = reference['source']
            result.at[index, 'reference_label_note'] = reference['note']
        records.append(dict(patient_id=str(row.patient_id), prediction_date=row.prediction_date,
            source_asm_record_id=str(row.source_asm_record_id), automated_status_at_visit=old,
            reference_status_at_visit=reference['status'] if reference else old,
            reference_label_source=reference['source_type'] if reference else 'computed_visit_status',
            reference_label_file=reference['source'] if reference else None,
            reference_label_note=reference['note'] if reference else None,
            reference_label_file_sha256=reference['sha256'] if reference else None,
            folder_status_at_visit=folder_references[str(row.patient_id)]['status'] if str(row.patient_id) in folder_references else None,
            filled_unknown=reference is not None and old is None,
            changed_known=reference is not None and old is not None and old != reference['status']))
    audit = pd.DataFrame(records, index=result.index)
    summary = dict(label_root=str(root), priority='Explicit user-confirmed override, then provided folder label, then computed latest status',
        user_confirmed_overrides=override_provenance,
        patients_with_folder_labels=int(targets.patient_id.astype(str).isin(folder_references).sum()),
        unknown_labels_resolved=int(audit.filled_unknown.sum()),
        known_labels_changed=int(audit.changed_known.sum()),
        known_label_disagreement_patient_ids=audit.loc[audit.changed_known, 'patient_id'].tolist(),
        patients_without_folder_labels=targets.loc[~targets.patient_id.astype(str).isin(folder_references), 'patient_id'].tolist(),
        patients_remaining_unknown=result.loc[result.status_at_visit.isna(), 'patient_id'].tolist())
    return result, audit, summary


def write_latest_report(report, output):
    active_tasks = {f'visit_status_history_{months}m' for months in HISTORY_WINDOWS}
    cohort = report['latest_visit_cohort']
    holdout = report.get('temporal_holdout')
    reference = report.get('reference_labels')
    labeling = ('Provided DRE/NDRE folder labels are the reference for each patient\'s latest visit. '
        'Explicit user-confirmed labels take precedence, followed by folder labels and computed status. '
        'Patients without a folder label or user confirmation retain their computed status. '
        'Both label versions and source paths are preserved in label_audit.jsonl.' if reference else
        'Computed visit status supplies the reference label.')
    lines = ['# DRE/NDRE status at each patient\'s most recent visit', '']
    if holdout:
        lines += [f'The evaluation window runs from {holdout["start_inclusive"]} through '
            f'{holdout["end_inclusive"]} inclusive, ending on the snapshot extraction date in Asia/Taipei.',
            'The saved forests were fitted using labeled visits before the window from training patients.', '']
    lines += [labeling, '', 'Select the latest recorded medication visit per patient before checking its label. '
        'Performance uses held-out patients whose latest visit falls in the evaluation cohort and has a known status.', '',
        f'Source model run: `{report["source_results_directory"]}`.',
        f'Source snapshot: `{report["source_snapshot"]}`.', '',
        f'Original training/test patients: {report["split_patient_counts"]["train"]}/'
        f'{report["split_patient_counts"]["test"]}. '
        f'Original labeled training visits: {report["labeled_train_visits"]}.', '',
        '| Group | Latest visits | DRE | NDRE | Unknown | Patients without an eligible visit |',
        '|---|---:|---:|---:|---:|---:|']
    for part in ('train', 'test'):
        c = cohort[part]
        lines.append(f'| {part} | {c["visits"]} | {c["dre"]} | {c["ndre"]} | '
                     f'{c["unknown"]} | {c["patients_without_visits"]} |')
    lines += ['', f'Test latest-visit dates: {report["test_visit_date_range"]["earliest"]} to '
              f'{report["test_visit_date_range"]["latest"]}.', '',
        'All histories start 12 calendar months before the selected visit. '
        'The 3m/6m/9m histories end 9/6/3 months before that visit, respectively. '
        'Starts are inclusive and ends exclusive. The target is the provided latest status when '
        'folder labels are supplied; otherwise it is computed status at that visit.', '',
        '| History | Model | Test AUC | Brier | Brier 95% CI | Accuracy | Precision | F1 | NPV | Sensitivity | Specificity |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for task, row in report['tasks'].items():
        if task not in active_tasks:
            continue
        for variant, m in row['models'].items():
            lines.append(f'| {task.rsplit("_", 1)[-1]} | {variant} | {_fmt(m["roc_auc"])} | '
                f'{m["brier"]:.4f} | {format_interval(m["bootstrap_95"].get("brier"))} | '
                f'{_fmt(m["accuracy_at_0_5"])} | {_fmt(m["precision_at_0_5"])} | '
                f'{_fmt(m["f1_at_0_5"])} | '
                f'{_fmt(m["npv_at_0_5"])} | '
                f'{_fmt(m["sensitivity_at_0_5"])} | {_fmt(m["specificity_at_0_5"])} |')
    lines += ['', 'DRE is positive; classification threshold is 0.5. Each evaluated patient '
        'has one visit and equal weight. F1 is 2*TP / (2*TP + FP + FN), undefined when there are '
        'neither actual nor predicted DRE outcomes. Specificity is TN / (TN + FP), undefined when '
        'there are no actual NDRE outcomes. NPV is TN / (TN + FN) and is undefined '
        'when there are no negative predictions. Brier intervals use '
        f'{report["bootstrap_replicates"]:,} patient bootstrap samples. '
        'NA denotes an undefined metric. The prevalence baseline is computed from the original '
        'patient-weighted training visits, matching the source run.', '',
        '## ROC comparison', '', '![Latest held-out visit ROC comparison](roc_comparison.png)', '',
        '[ROC data](roc_curves.json)', '', '## Confusion matrices', '',
        'Rows show actual status and columns predicted status, both in DRE/NDRE order. '
        'Cells show patient counts and percentages within each actual class.', '']
    for task, models in report['confusion_matrix_plots'].items():
        if task not in active_tasks:
            continue
        lines.append(f'- {task.rsplit("_", 1)[-1]} history: ' + ' | '.join(
            f'[{variant.replace("_", " ")}]({path})' for variant, path in models.items()))
    lines += ['', '## Interpretation', '',
        'The evaluation uses the latest eligible visit per held-out patient in the saved snapshot. ' +
        ('Training uses the original project-derived labels. Latest-visit evaluation uses the supplied '
         'folder annotations, whose provenance is recorded without assuming independent adjudication. ' if reference else
         'Labels use the project operational rules and are not independently adjudicated. ') +
        'Unknown latest outcomes are unvalidated. Probabilities remain uncalibrated.', '',
        ('The combined prediction file also contains recent visits from training patients, marked as excluded '
         'from fitting and performance evaluation. Use heldout_status_predictions.jsonl for the latest-visit '
         'evaluation cohort.' if holdout else
         'The combined prediction file includes in-sample training patients. Use heldout_status_predictions.jsonl '
         'for the latest-visit evaluation cohort.'), '']
    if reference:
        lines += ['## Reference-label audit', '',
            f'Folder labels cover {reference["patients_with_folder_labels"]} latest visits. '
            f'Combined reference labels resolve {reference["unknown_labels_resolved"]} unknown labels and change '
            f'{reference["known_labels_changed"]} known computed labels.', '',
            'Known-label disagreements: ' + (', '.join(reference['known_label_disagreement_patient_ids']) or 'None') + '.',
            'Patients without folder labels: ' + (', '.join(reference['patients_without_folder_labels']) or 'None') + '.',
            'Patients remaining unknown: ' + (', '.join(reference['patients_remaining_unknown']) or 'None') + '.', '',
            '[Patient-level label audit](label_audit.jsonl)', '']
        if reference.get('user_confirmed_overrides') and reference['user_confirmed_overrides']['patient_ids']:
            lines += ['User-confirmed label overrides: ' + ', '.join(reference['user_confirmed_overrides']['patient_ids']) + '.', '',
                      'Override source: `' + reference['user_confirmed_overrides']['file'] + '`.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def evaluate_latest(results_dir, output_dir, bootstrap=1000, seed=42, progress=print, reference_label_root=None,
                    label_overrides=None):
    source, output = Path(results_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Choose a new evaluation output directory')
    if bootstrap < 0:
        raise ValueError('Bootstrap replicates must be nonnegative')
    if label_overrides is not None and reference_label_root is None:
        raise ValueError('Label overrides require --reference-label-root')
    original = json.loads((source / 'metrics.json').read_text())
    provenance = json.loads((source / 'provenance.json').read_text())
    feature_dir = Path(provenance['feature_directory'])
    if digest(feature_dir / 'manifest.json') != provenance['feature_manifest_sha256']:
        raise ValueError('Feature bundle differs from the trained model input')
    frames, visits, targets, _, split, manifest, _ = load_status_bundle(feature_dir)
    holdout = manifest.get('temporal_holdout')
    if (manifest['format_version'] != BUNDLE_FORMAT or not holdout or original.get('temporal_holdout') != holdout
            or provenance.get('temporal_holdout') != holdout
            or original.get('evaluation_scope') != 'recent_visits_of_heldout_patients'):
        raise ValueError('Latest-visit evaluation requires forests trained with the 12-month temporal holdout')
    if split != json.loads((source / 'patient_split.json').read_text()):
        raise ValueError('Model and feature bundle patient splits differ')
    indices = latest_visit_indices(visits)
    recent = indices[visits.loc[indices, 'period'].eq('recent')]
    latest = visits.loc[recent].copy()
    latest_targets = targets.loc[recent].copy()
    if not latest['split'].eq('test').any():
        raise ValueError('No latest visits from held-out patients in the 12-month window')
    audit, reference = None, None
    if reference_label_root is not None:
        latest_targets, audit, reference = apply_folder_labels(
            targets.loc[indices], reference_label_root, label_overrides, eligible_indices=recent)
        audit['split'] = latest['split']
    labels = latest_targets.status_at_visit
    test = latest['split'].eq('test') & labels.notna()
    if not test.any():
        raise ValueError('No known latest-visit outcomes in held-out patients')
    predictions = latest.copy()
    predictions['observed_status_at_visit'] = labels
    predictions['target_reason'] = latest_targets.target_reason
    if reference:
        for column in ('automated_status_at_visit', 'automated_target_reason', 'reference_label_source', 'reference_label_file',
                       'reference_label_note'):
            predictions[column] = latest_targets[column]
    predictions['prediction_scope'] = np.where(latest['split'].eq('test'), 'heldout_patient', 'excluded_training_patient')
    cohort = {}
    for part in ('train', 'test'):
        group = latest['split'].eq(part)
        values = labels.loc[group]
        cohort[part] = dict(visits=int(group.sum()), dre=int(values.eq(1).sum()),
            ndre=int(values.eq(0).sum()), unknown=int(values.isna().sum()),
            patients_without_visits=len(set(split[part]) - set(latest.loc[group, 'patient_id'])))
    y = labels.loc[test].to_numpy(dtype=int)
    ids = latest.loc[test, 'patient_id']
    prevalence = original['training_prevalence']
    dates = pd.to_datetime(latest.loc[latest['split'].eq('test'), 'prediction_date'])
    report = copy.deepcopy(original)
    report.update(generated_at_taipei=dt.datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        history_windows_months=list(HISTORY_WINDOWS),
        evaluation_scope='latest_recent_visit_per_heldout_patient', retrained=False,
        source_results_directory=str(source), source_snapshot=provenance['feature_provenance']['snapshot'],
        latest_visit_cohort=cohort, labeled_test_visits=int(test.sum()),
        excluded_unknown_status_visits=cohort['test']['unknown'],
        label_counts=cohort, bootstrap_replicates=bootstrap, bootstrap_seed=seed,
        test_visit_date_range=dict(earliest=dates.min().date().isoformat(), latest=dates.max().date().isoformat()),
        training_prevalence=prevalence, tasks={},
        limitations=['Retrospective temporal evaluation on held-out patients in the source snapshot.',
            'Patients whose latest visits predate the holdout window are outside the evaluation cohort.',
            'Only known latest outcomes contribute to performance.',
            'Labels use project operational rules, not independent adjudication.',
            'Probabilities are uncalibrated; recording timestamps are unavailable.'])
    if reference:
        report['reference_labels'] = reference
        report['training_target_definition'] = original['target_definition']
        report['target_definition'] = ('User-confirmed latest-status overrides take precedence, followed by provided '
            'DRE/NDRE folder classification; computed latest status is retained only without either provided label.')
        report['limitations'][3] = 'Training uses computed labels; latest-visit evaluation uses provided folder annotations.'
    output.mkdir(parents=True)
    registry = json.loads((source / 'models.json').read_text())
    model_hashes = {}
    for months in HISTORY_WINDOWS:
        task = f'visit_status_history_{months}m'
        row = copy.deepcopy(original['tasks'][task])
        row['models'] = {}
        for variant in ('all_features', 'selected_features'):
            name = f'{task}_{variant}'
            path = source / registry[name]
            model_hashes[registry[name]] = digest(path)
            model = joblib.load(path)
            if model.history_months != months or model.variant != variant or model.history_anchor_months != 12:
                raise ValueError('Saved model metadata mismatch')
            p = model.predict_dre_probability(frames[months].loc[recent])
            row['models'][variant] = performance(y, p[test.to_numpy()], ids, bootstrap, seed)
            if variant == 'selected_features':
                predictions[f'risk_dre_pct_history_{months}m'] = p * 100
                predictions[f'risk_ndre_pct_history_{months}m'] = (1 - p) * 100
                predictions[f'predicted_status_history_{months}m'] = np.where(p >= .5, 'DRE', 'NDRE')
                for field, bound in (('history_start', 0), ('history_end_exclusive', 1)):
                    predictions[f'{field}_{months}m'] = [history_bounds(day(d), months)[bound].isoformat()
                                                         for d in latest.prediction_date]
        row['models']['training_prevalence_baseline'] = performance(y, np.full(len(y), prevalence), ids, bootstrap, seed)
        report['tasks'][task] = row
        if progress:
            progress(f'{months}m: {len(y)} latest held-out visits; selected-feature AUC='
                     f'{_fmt(row["models"]["selected_features"]["roc_auc"])}; '
                     f'NPV={_fmt(row["models"]["selected_features"]["npv_at_0_5"])}')
    heldout = predictions.loc[latest['split'].eq('test')]
    predictions.to_json(output / 'status_predictions.jsonl', orient='records', lines=True, date_format='iso')
    heldout.to_json(output / 'heldout_status_predictions.jsonl', orient='records', lines=True, date_format='iso')
    latest_targets.to_json(output / 'status_targets.jsonl', orient='records', lines=True, date_format='iso')
    if audit is not None:
        audit.to_json(output / 'label_audit.jsonl', orient='records', lines=True, date_format='iso')
    report['confusion_matrix_plots'] = write_confusion_matrices(report, output)
    report['roc_comparison'] = write_roc_comparison(heldout, output)
    write_json(output / 'metrics.json', report)
    write_json(output / 'patient_split.json', split)
    write_json(output / 'provenance.json', dict(source_results_directory=str(source),
        source_metrics_sha256=digest(source / 'metrics.json'),
        source_provenance_sha256=digest(source / 'provenance.json'),
        feature_directory=str(feature_dir), feature_manifest_sha256=digest(feature_dir / 'manifest.json'),
        model_sha256=model_hashes, retrained=False, temporal_holdout=holdout,
        reference_labels=reference,
        label_audit_sha256=digest(output / 'label_audit.jsonl') if audit is not None else None,
        selection='Maximum prediction_date per patient within the 12-month holdout, before outcome filtering',
        code_sha256={name: digest(Path(__file__).parent / name) for name in
                     ('evaluate_latest_visits.py', 'forest_workflow.py', 'plot_visit_status.py')}))
    write_latest_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-label-root', type=Path,
                        help='Parent directory containing authoritative DRE and NDRE folders for latest status')
    parser.add_argument('--label-overrides', type=Path, help='JSON file of explicit user-confirmed latest-status labels')
    args = parser.parse_args()
    evaluate_latest(args.results_dir, args.output_dir, args.bootstrap, args.seed,
                    reference_label_root=args.reference_label_root, label_overrides=args.label_overrides)
    print('Saved latest-visit evaluation: ' + str(args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
    main()
