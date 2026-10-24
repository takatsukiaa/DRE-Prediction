"""Train and evaluate status-at-visit models from a prepared bundle only."""
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn

from build_visit_status import digest, validate_split, write_json
from longitudinal_features import day, history_bounds
from plot_confusion_matrices import write_confusion_matrices
from plot_roc_curves import write_roc_comparison
from forest_workflow import (FOREST_DEFAULTS, _fmt, core_feature, fit_forest,
    format_interval, patient_weights, performance, source_importance)
from visit_status_targets import HISTORY_WINDOWS, TARGET_DEFINITION

ROOT = Path(__file__).resolve().parent


@dataclass
class VisitStatusModel:
    pipeline: object
    history_months: int
    variant: str
    target_definition: str = TARGET_DEFINITION
    history_anchor_months: int = 12

    def predict_dre_probability(self, X):
        classes = self.pipeline.named_steps['forest'].classes_
        return self.pipeline.predict_proba(X)[:, int(np.flatnonzero(classes == 1)[0])]


def load_status_bundle(feature_dir):
    root = Path(feature_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    required = {'visits.jsonl', 'status_targets.jsonl', 'patient_split.json', 'provenance.json', 'eeg_report_audit.jsonl'}
    required.update(f'{stem}_{months}m{ext}' for months in HISTORY_WINDOWS
                    for stem, ext in [('features', '.pkl.gz'), ('feature_dictionary', '.json'), ('audit', '.json')])
    if (manifest.get('format_version') != 2 or manifest.get('task') != 'visit_status'
            or manifest.get('history_anchor_months') != 12
            or manifest.get('history_windows_months') != list(HISTORY_WINDOWS)
            or set(manifest.get('files', {})) != required):
        raise ValueError('Unsupported visit-status bundle; run build_visit_status.py first')
    for name, expected in manifest['files'].items():
        if digest(root / name) != expected:
            raise ValueError('Visit-status bundle checksum mismatch: ' + name)
    dtypes = {'patient_id': 'string', 'source_asm_record_id': 'string'}
    visits = pd.read_json(root / 'visits.jsonl', lines=True, dtype=dtypes, convert_dates=False)
    targets = pd.read_json(root / 'status_targets.jsonl', lines=True, dtype=dtypes, convert_dates=False)
    split = json.loads((root / 'patient_split.json').read_text())
    validate_split(split, split['train'] + split['test'])
    assignment = {pid: part for part in ('train', 'test') for pid in split[part]}
    keys = ['patient_id', 'prediction_date', 'source_asm_record_id']
    if (len(visits) != manifest['rows'] or not visits[keys].equals(targets[keys])
            or visits.duplicated(['patient_id', 'prediction_date']).any()
            or visits.patient_id.map(assignment).isna().any()
            or visits.patient_id.map(assignment).tolist() != visits['split'].tolist()):
        raise ValueError('Visit-status target, visit, or patient split alignment mismatch')
    if not targets.status_at_visit.dropna().isin([0, 1]).all():
        raise ValueError('Status labels must be 0, 1, or unknown')
    targets['status_at_visit'] = targets.status_at_visit.astype('Int8')
    frames, dictionaries = {}, {}
    for months in HISTORY_WINDOWS:
        X = pd.read_pickle(root / f'features_{months}m.pkl.gz')
        dictionary = json.loads((root / f'feature_dictionary_{months}m.json').read_text())
        if (len(X) != len(visits) or X.index.tolist() != visits.index.tolist()
                or list(X) != list(dictionary) or X.columns.duplicated().any()
                or len(X.columns) != manifest['windows'][f'{months}m']['features']):
            raise ValueError('Visit-status feature alignment mismatch')
        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes) or np.isinf(X.to_numpy()).any():
            raise ValueError('Status features must be numeric and finite or missing')
        forbidden = ('target_', 'outcome_', 'risk_', 'current_', 'monthly_', 'status_', 'switch_', 'followup_')
        if any(c.startswith(forbidden) or c == 'history_days_available' for c in X):
            raise ValueError('Status outcome or unrestricted history present in predictors')
        frames[months], dictionaries[months] = X, dictionary
    provenance = json.loads((root / 'provenance.json').read_text())
    return frames, visits, targets, dictionaries, split, manifest, provenance


def run_status_training(feature_dir, output_dir, seed=42, trees=500, jobs=6,
                        bootstrap=1000, top_features=20, panel='all', progress=print):
    feature_dir, output = Path(feature_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError('Output already exists; choose a new visit-status results directory')
    if trees < 1 or top_features < 1 or bootstrap < 0 or panel not in ('core', 'all'):
        raise ValueError('Invalid visit-status training settings')
    frames, visits, targets, dictionaries, split, manifest, provenance = load_status_bundle(feature_dir)
    observed = targets.status_at_visit.notna()
    train = observed & visits['split'].eq('train')
    test = observed & visits['split'].eq('test')
    if not train.any() or not test.any():
        raise ValueError('Status training needs labeled visits in both patient groups')
    y_train, y_test = targets.loc[train, 'status_at_visit'].to_numpy(dtype=int), targets.loc[test, 'status_at_visit'].to_numpy(dtype=int)
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError('Status training needs DRE and NDRE visits in training patients')
    train_ids, test_ids = visits.loc[train, 'patient_id'], visits.loc[test, 'patient_id']
    settings = dict(FOREST_DEFAULTS, random_state=seed, n_estimators=trees, n_jobs=jobs)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    predictions = visits.copy()
    predictions['observed_status_at_visit'] = targets.status_at_visit
    predictions['target_reason'] = targets.target_reason
    predictions['prediction_scope'] = np.where(visits['split'].eq('train'), 'in_sample_training_patient', 'heldout_patient')
    report = dict(generated_at_taipei=dt.datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        task='visit_status', target_definition=TARGET_DEFINITION, positive_class='DRE',
        history_windows_months=list(HISTORY_WINDOWS), forest_settings=settings, panel=panel,
        history_anchor_months=12,
        top_features=top_features, bootstrap_replicates=bootstrap, primary_variant='selected_features',
        split_patient_counts={part: len(split[part]) for part in ('train', 'test')},
        labeled_train_visits=int(train.sum()), labeled_test_visits=int(test.sum()),
        excluded_unknown_status_visits=int((~observed).sum()), label_counts=manifest['label_counts'],
        probability_semantics='P(DRE at visit); P(NDRE at visit) = 1 - P(DRE at visit)',
        calibration='uncalibrated forests; calibration diagnostics use held-out patients only',
        limitations=['Labels follow the project operational DRE rules, not independent adjudication.',
            'Observation and linked visit dates proxy availability; recording timestamps are unavailable.',
            'Visits with unknown status are excluded from fitting and evaluation; their scores are unvalidated.',
            'All visits are scored, including those with sparse or no dynamic history; full history is not required.',
            'Model comparisons use the same held-out patients and labeled visits; no cross-validation or test-set tuning.'],
        tasks={})
    importance, selections, registry, retained = {}, {}, {}, {}
    for months in HISTORY_WINDOWS:
        task, X, dictionary = f'visit_status_history_{months}m', frames[months], dictionaries[months]
        candidates = [c for c in X if panel == 'all' or core_feature(c, dictionary[c])]
        if progress:
            progress(f'Fitting {task}: {int(train.sum())} training visits, {train_ids.nunique()} patients')
        # fit_forest supplies the shared patient weighting and training-only
        # preprocessing. Saved status models carry their own target semantics.
        full = fit_forest(X.loc[train], y_train, train_ids, candidates, 'all_features', settings)
        ranking, columns = source_importance(full, candidates, top_features)
        selected = fit_forest(X.loc[train], y_train, train_ids, columns, 'selected_features', settings)
        importance[task], selections[task] = ranking, columns
        row = dict(input_features=len(X.columns), candidate_features=len(candidates), selected_feature_count=len(columns), models={})
        for variant, fitted in [('all_features', full), ('selected_features', selected)]:
            model = VisitStatusModel(fitted.pipeline, months, variant)
            p = model.predict_dre_probability(X.loc[test])
            row['models'][variant] = performance(y_test, p, test_ids, bootstrap, seed)
            name = f'{task}_{variant}'
            path = f'models/{name}.joblib'
            joblib.dump(model, output / path, compress=3)
            registry[name] = path
            filt = model.pipeline.named_steps['features']
            retained[name] = dict(retained=filt.selected_columns_, dropped_all_missing=filt.all_missing_, dropped_constant=filt.constant_)
            if variant == 'selected_features':
                p_all = model.predict_dre_probability(X)
                predictions[f'risk_dre_pct_history_{months}m'] = p_all * 100
                predictions[f'risk_ndre_pct_history_{months}m'] = 100 - p_all * 100
                predictions[f'predicted_status_history_{months}m'] = np.where(p_all >= .5, 'DRE', 'NDRE')
                predictions[f'history_start_{months}m'] = [
                    history_bounds(day(date), months)[0].isoformat() for date in visits.prediction_date]
                predictions[f'history_end_exclusive_{months}m'] = [
                    history_bounds(day(date), months)[1].isoformat() for date in visits.prediction_date]
        prevalence = float(np.average(y_train, weights=patient_weights(train_ids)))
        row['models']['training_prevalence_baseline'] = performance(y_test, np.full(len(y_test), prevalence), test_ids, bootstrap, seed)
        report['tasks'][task] = row
        if progress:
            metrics = row['models']['selected_features']
            progress(f'Finished {task}: held-out AUC={metrics["roc_auc"]}, Brier={metrics["brier"]:.4f}')
    predictions.to_json(output / 'status_predictions.jsonl', orient='records', lines=True, date_format='iso')
    predictions.loc[visits['split'].eq('test')].to_json(output / 'heldout_status_predictions.jsonl', orient='records', lines=True, date_format='iso')
    targets.to_json(output / 'status_targets.jsonl', orient='records', lines=True, date_format='iso')
    report['confusion_matrix_plots'] = write_confusion_matrices(report, output)
    report['roc_comparison'] = write_roc_comparison(predictions.loc[visits['split'].eq('test')], output)
    for name, value in {'metrics.json': report, 'feature_importance.json': importance, 'selected_features.json': selections,
                        'models.json': registry, 'retained_features.json': retained, 'patient_split.json': split}.items():
        write_json(output / name, value)
    write_json(output / 'provenance.json', dict(feature_directory=str(feature_dir.resolve()),
        feature_manifest_sha256=digest(feature_dir / 'manifest.json'), feature_provenance=provenance,
        input_sha256=manifest['files'], code_sha256={name: digest(ROOT / name) for name in
            ('train.py', 'visit_status_workflow.py', 'forest_workflow.py', 'visit_status_targets.py', 'build_visit_status.py',
             'plot_confusion_matrices.py', 'plot_roc_curves.py')},
        python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__))
    write_status_report(report, output)
    return report


def write_status_report(report, output):
    lines = ['# DRE/NDRE status at each visit', '', report['target_definition'], '',
        'All input windows start 12 calendar months before prediction: 3m uses [visit - 12m, visit - 9m), '
        '6m uses [visit - 12m, visit - 6m), 9m uses [visit - 12m, visit - 3m), '
        'and 12m uses [visit - 12m, visit). '
        'Evidence must be available before the respective window end. Visit-day evidence defines the target only.',
        'All four windows use the same visit labels, patient assignments, and evaluation visits.',
        'Features are selected and imputed using training patients only. Each patient has equal total fitting and evaluation weight.', '',
        f'Training/test patients: {report["split_patient_counts"]["train"]}/{report["split_patient_counts"]["test"]}. '
        f'Labeled training/test visits: {report["labeled_train_visits"]}/{report["labeled_test_visits"]}. '
        f'Unknown status visits excluded: {report["excluded_unknown_status_visits"]}.', '',
        f"Primary model: top {report['top_features']} source features per history window. Forest settings: {report['forest_settings']}.", '',
        '| History | Model | Test AUC | Brier | Brier 95% CI | Accuracy at 0.5 | Precision at 0.5 | Sensitivity at 0.5 |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    for task, row in report['tasks'].items():
        for variant, values in row['models'].items():
            interval = format_interval(values.get('bootstrap_95', {}).get('brier'))
            lines.append(f'| {task.rsplit("_", 1)[-1]} | {variant} | {_fmt(values["roc_auc"])} | {values["brier"]:.4f} | {interval} | {_fmt(values["accuracy_at_0_5"])} | {_fmt(values["precision_at_0_5"])} | {_fmt(values["sensitivity_at_0_5"])} |')
    lines += ['', 'Accuracy, precision and sensitivity are proportions at probability threshold 0.5. The positive class is DRE. Precision is positive predictive value at that threshold. Brier 95% CIs use patient-cluster bootstrap resamples. NA means the metric is undefined or its interval was not computed.', '',
              'Percentages are uncalibrated estimates. DRE and NDRE percentages sum to 100 for each history model.',
              'History windows describe separate models; their probabilities have no required monotonic ordering.',
              'Combined predictions identify in-sample training patients; use heldout_status_predictions.jsonl for independent evaluation.', '']
    lines += ['- ' + limitation for limitation in report['limitations']]
    if report.get('roc_comparison'):
        roc = report['roc_comparison']
        lines += ['', '## ROC comparison', '',
            'The 3-, 6-, 9-, and 12-month selected-feature models are compared on the same labeled held-out visits. '
            'DRE is positive. Curves and AUROCs use the same equal-patient weights as the performance table. '
            'The dashed diagonal shows chance discrimination. AUROC is undefined when only one observed class is present.', '',
            f'![Held-out ROC comparison of four history models]({roc["plot"]})', '',
            f'[ROC curve data]({roc["data"]})']
    if report.get('confusion_matrix_plots'):
        lines += ['', '## Confusion matrices', '',
            'Held-out labeled visits at threshold 0.5. Rows are actual status; columns are predicted status, labeled at the top. '
            'Both axes use DRE, NDRE order, with DRE as positive. '
            'Cells show raw visit counts and percentages within each actual class. '
            'N/A means no observed visits for that class. These plots are unweighted; the evaluation metrics above use patient weights.', '']
        for task, models in report['confusion_matrix_plots'].items():
            links = [f'[{variant.replace("_", " ")}]({path})' for variant, path in models.items()]
            lines.append(f'- {task.rsplit("_", 1)[-1]} history: ' + ' | '.join(links))
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
