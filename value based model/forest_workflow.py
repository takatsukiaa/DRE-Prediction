"""Fixed-split random forests for any observed monthly state switch.

All fitting uses saved training patients only. Test outcomes evaluate fixed
models; they never select features, tune thresholds, or fit calibration.
"""
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from risk_targets import build_switch_targets, risk_percentages
from sql_snapshot import read_snapshot

ROOT = Path(__file__).resolve().parent
CORE_LABS = {'wbc', 'rbc', 'hb', 'hct', 'plt', 'na', 'k', 'ca', 'mg', 'bun', 'crea',
             'egfr', 'ast', 'alt', 'alb', 'hba1c', 'cbz', 'ptn', 'valp', 'phno'}
FOREST_DEFAULTS = dict(n_estimators=500, max_depth=8, min_samples_leaf=10,
                       max_features='sqrt', class_weight=None, random_state=42, n_jobs=4)


def core_feature(name, spec):
    """Clinical panel defined from names/sources, never observed test values."""
    if name in {'age_years', 'sex_male', 'history_days_available', 'current_dre', 'current_dre_unknown', 'current_state_age_days'}:
        return True
    if name.startswith(('asm_', 'asm__', 'seizure_', 'eeg_', 'eeg__', 'brain_image_', 'monthly_')):
        return True
    field = str(spec.get('measurement', ''))
    if name.startswith('mmse__'):
        return field == 'TotalScore'
    if name.startswith('casi__'):
        return field == 'TotalScore' or bool(re.fullmatch('Q[1-9]', field))
    if name.startswith('wais__'):
        return field == 'TotalScore'
    base = re.sub(r'\([^)]*\)|\s+', '', field).casefold()
    return name.startswith(('biochem__', 'hematology__', 'immuno__', 'druglevel__')) and base in CORE_LABS


def change_feature(name):
    # "Delta waves" is an EEG morphology, not a visit-to-visit difference.
    return (name.endswith(('_delta', '__delta_score', '__change_per30d')) or
        name.startswith(('asm_drugs_added', 'asm_drugs_stopped', 'asm_composition_changes',
            'asm_dose_increased', 'asm_dose_decreased', 'asm_days_since_last_observed_composition_change',
            'asm_regimen_changed', 'eeg_ed_newly_detected', 'eeg_ed_no_longer_detected',
            'monthly_observed_switches')))


class TrainingFeatureFilter(TransformerMixin, BaseEstimator):
    """Learn only all-empty/constant exclusions from the provided training rows."""
    def __init__(self, candidate_columns):
        self.candidate_columns = candidate_columns

    def fit(self, X, y=None):
        absent = set(self.candidate_columns) - set(X.columns)
        if absent:
            raise ValueError('Missing feature columns: ' + str(sorted(absent)))
        frame = X.loc[:, self.candidate_columns]
        self.all_missing_ = frame.columns[frame.isna().all()].tolist()
        # A constant observed value with variable missingness is retained so
        # the training-fitted missingness indicator can capture availability.
        self.constant_ = [c for c in frame if c not in self.all_missing_ and frame[c].nunique(dropna=False) <= 1]
        self.selected_columns_ = [c for c in self.candidate_columns if c not in set(self.all_missing_ + self.constant_)]
        if not self.selected_columns_:
            raise ValueError('No usable training features')
        return self

    def transform(self, X):
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_columns_, dtype=object)


@dataclass
class SwitchRiskModel:
    pipeline: Pipeline
    starting_state: int
    horizon_months: int
    variant: str
    target_definition: str = 'any opposite stored monthly state during exact calendar-month follow-up'

    def predict_switch_probability(self, X):
        classes = self.pipeline.named_steps['forest'].classes_
        return self.pipeline.predict_proba(X)[:, int(np.where(classes == 1)[0][0])]


def patient_weights(patient_ids):
    ids = pd.Series(patient_ids, copy=False)
    return (1 / ids.map(ids.value_counts())).to_numpy(dtype=float)


def fit_forest(X, y, patient_ids, candidates, state, horizon, variant, settings=None):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Training requires observed switch and no-switch outcomes')
    pipeline = Pipeline([('features', TrainingFeatureFilter(candidates)),
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('forest', RandomForestClassifier(**(settings or FOREST_DEFAULTS)))])
    pipeline.fit(X, np.asarray(y, dtype=np.int8), forest__sample_weight=patient_weights(patient_ids))
    return SwitchRiskModel(pipeline, state, horizon, variant)


def source_importance(model, candidates, top_features):
    """Rank original features using the training forest, grouping missing flags.

    No test inputs or outcomes are accepted. Excluded candidates remain in the
    audit with zero importance and an explicit exclusion reason.
    """
    if top_features < 1:
        raise ValueError('top_features must be positive')
    filt = model.pipeline.named_steps['features']
    names = model.pipeline.named_steps['imputer'].get_feature_names_out(filt.selected_columns_)
    scores = {name: 0. for name in candidates}
    indicators = {name: 0. for name in candidates}
    for name, score in zip(names, model.pipeline.named_steps['forest'].feature_importances_):
        indicator = str(name).startswith('missingindicator_')
        source = str(name)[len('missingindicator_'):] if indicator else str(name)
        scores[source] += float(score)
        if indicator:
            indicators[source] += float(score)
    ranked = sorted(filt.selected_columns_, key=lambda name: (-scores[name], name))
    selected = ranked[:top_features]
    ranks = {name: i + 1 for i, name in enumerate(ranked)}
    rows = []
    for name in ranked + [c for c in candidates if c not in ranks]:
        rows.append(dict(feature=name, rank=ranks.get(name), impurity_importance=scores[name],
            missing_indicator_importance=indicators[name], selected_top_k=name in selected,
            status='all_missing_in_training' if name in filt.all_missing_ else
                   'constant_in_training' if name in filt.constant_ else 'usable'))
    return rows, selected


def calibration_bins(y, probability, weights, patient_ids):
    bins = np.minimum((probability * 10).astype(int), 9)
    result = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            result.append(dict(lower=b / 10, upper=(b + 1) / 10, visits=int(mask.sum()),
                patients=int(pd.Series(np.asarray(patient_ids)[mask]).nunique()),
                mean_probability=float(np.average(probability[mask], weights=weights[mask])),
                observed_fraction=float(np.average(y[mask], weights=weights[mask]))))
    return result


def calibration_diagnostics(y, p, weights):
    """Diagnostic regression only: never applied back to reported predictions."""
    if len(np.unique(y)) < 2:
        return dict(intercept=None, slope=None, converged=False)
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped))
    if np.ptp(logits) < 1e-10:
        observed = np.average(y, weights=weights)
        intercept = np.log(observed / (1 - observed)) - logits[0]
        return dict(intercept=float(intercept), slope=None, converged=True)
    design = np.column_stack([np.ones(len(y)), logits])
    w = weights / weights.sum()

    def loss(beta):
        z = design @ beta
        return np.sum(w * (np.logaddexp(0, z) - y * z))

    def gradient(beta):
        z = np.clip(design @ beta, -700, 700)
        return design.T @ (w * (1 / (1 + np.exp(-z)) - y))

    fitted = minimize(loss, np.array([0., 1.]), jac=gradient, method='BFGS')
    return dict(intercept=float(fitted.x[0]) if fitted.success else None,
                slope=float(fitted.x[1]) if fitted.success else None, converged=bool(fitted.success))


def performance(y, p, patient_ids, bootstrap=1000, seed=42):
    y, p, ids = np.asarray(y, dtype=int), np.asarray(p, dtype=float), np.asarray(patient_ids)
    w = patient_weights(ids)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    squared = (p - y) ** 2
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    classified = (p >= .5).astype(int)
    cm = confusion_matrix(y, classified, labels=[0, 1], sample_weight=w)
    tn, fp, fn, tp = cm.ravel()
    means = pd.DataFrame(dict(patient=ids, brier=squared, log_loss=losses)).groupby('patient')[['brier', 'log_loss']].mean()
    intervals = {}
    if bootstrap > 0 and len(means) >= 2:
        draws = np.random.default_rng(seed).integers(0, len(means), size=(bootstrap, len(means)))
        for name in means:
            scores = means[name].to_numpy()[draws].mean(axis=1)
            intervals[name] = [float(v) for v in np.quantile(scores, [.025, .975])]
    two_classes = len(np.unique(y)) == 2
    return dict(visits=len(y), patients=len(means), events=int(y.sum()),
        patient_weighted_event_fraction=float(np.average(y, weights=w)),
        mean_predicted_probability=float(np.average(p, weights=w)),
        brier=float(np.average(squared, weights=w)), log_loss=float(np.average(losses, weights=w)),
        roc_auc=float(roc_auc_score(y, p, sample_weight=w)) if two_classes else None,
        average_precision=float(average_precision_score(y, p, sample_weight=w)) if two_classes else None,
        accuracy_at_0_5=float(np.average(classified == y, weights=w)),
        sensitivity_at_0_5=float(tp / (tp + fn)) if tp + fn else None,
        specificity_at_0_5=float(tn / (tn + fp)) if tn + fp else None,
        precision_at_0_5=float(tp / (tp + fp)) if tp + fp else None,
        confusion_matrix_raw=confusion_matrix(y, classified, labels=[0, 1]).tolist(),
        bootstrap_95=intervals, calibration=calibration_diagnostics(y, p, w),
        calibration_bins=calibration_bins(y, p, w, ids))


def compare_brier(y, first, second, ids, bootstrap=1000, seed=42):
    frame = pd.DataFrame(dict(patient=np.asarray(ids), difference=(np.asarray(first) - y) ** 2 - (np.asarray(second) - y) ** 2))
    values = frame.groupby('patient').difference.mean().to_numpy()
    result = dict(brier_with_changes_minus_without_changes=float(values.mean()))
    if bootstrap and len(values) >= 2:
        draws = np.random.default_rng(seed).integers(0, len(values), size=(bootstrap, len(values)))
        result['bootstrap_95'] = [float(v) for v in np.quantile(values[draws].mean(axis=1), [.025, .975])]
    return result


def load_training_bundle(feature_dir, snapshot_dir=None):
    feature_dir = Path(feature_dir)
    X = pd.read_pickle(feature_dir / 'features.pkl.gz')
    visits = pd.read_json(feature_dir / 'visits.jsonl', lines=True,
                         dtype={'patient_id': 'string', 'source_asm_record_id': 'string'})
    dictionary = json.loads((feature_dir / 'feature_dictionary.json').read_text())
    split = json.loads((feature_dir / 'patient_split.json').read_text())
    feature_provenance = json.loads((feature_dir / 'provenance.json').read_text())
    audit = json.loads((feature_dir / 'audit.json').read_text())
    source = Path(snapshot_dir or feature_provenance['snapshot'])
    digest = hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest()
    if digest != feature_provenance['snapshot_manifest_sha256']:
        raise ValueError('Features and monthly-label snapshot are from different extractions')
    if len(X) != len(visits) or list(X) != list(dictionary) or X.index.tolist() != visits.index.tolist():
        raise ValueError('Feature/visit/dictionary row or column alignment mismatch')
    if X.columns.duplicated().any() or visits.duplicated(['patient_id', 'prediction_date']).any():
        raise ValueError('Duplicate features or patient prediction dates')
    if np.isinf(X.to_numpy()).any() or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError('Features must be finite numeric values or missing')
    if set(split['train']) & set(split['test']):
        raise ValueError('Patient overlap between training and test')
    assignment = {pid: part for part in ('train', 'test') for pid in split[part]}
    if visits.patient_id.map(assignment).isna().any() or not visits.patient_id.map(assignment).equals(visits['split'].astype('string')):
        # Compare values separately to avoid pandas string/object dtype differences.
        if visits.patient_id.map(assignment).tolist() != visits['split'].tolist():
            raise ValueError('Visit split does not match saved patient assignment')
    if any(c.startswith(('target_', 'outcome_', 'sample_weight_')) for c in X):
        raise ValueError('Outcome or audit column present in model inputs')
    snapshot = read_snapshot(source)
    targets = build_switch_targets(visits, snapshot.tables['dbo.PatientMonthlyDreStatus'], dt.date.fromisoformat(audit['as_of']))
    return X, visits, dictionary, split, targets, source


def write_plots(curves, output):
    os.environ.setdefault('MPLCONFIGDIR', str((output / 'plot_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ('calibration', 'roc'):
        figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        for ax, (task, data) in zip(axes.flat, curves.items()):
            for variant, row in data.items():
                if kind == 'calibration':
                    bins = row['metrics']['calibration_bins']
                    ax.plot([b['mean_probability'] * 100 for b in bins], [b['observed_fraction'] * 100 for b in bins], marker='o', label=variant)
                elif len(np.unique(row['y'])) == 2:
                    fpr, tpr, _ = roc_curve(row['y'], row['p'], sample_weight=patient_weights(row['ids']))
                    ax.plot(fpr, tpr, label=variant)
            limit = 100 if kind == 'calibration' else 1
            ax.plot([0, limit], [0, limit], '--', color='gray', linewidth=1)
            ax.set(xlim=(0, limit), ylim=(0, limit), title=task.replace('_', ' '),
                xlabel='Predicted switch risk (%)' if kind == 'calibration' else 'False positive rate',
                ylabel='Observed switch fraction (%)' if kind == 'calibration' else 'True positive rate')
            ax.legend(fontsize=8)
        figure.suptitle('Patient-weighted held-out results: any opposite monthly state')
        figure.savefig(output / f'{kind}.png', dpi=160)
        plt.close(figure)


def _fmt(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def write_report(report, output):
    lines = ['# Random forest: any observed monthly state switch', '',
        f"Generated: {report['generated_at_taipei']}. One saved 50-50 patient split; no cross-validation or tuning.", '',
        'The outcome is at least one opposite stored monthly state strictly after prediction and within the exact 6- or 12-calendar-month anniversary. A later reversal does not erase a switch. Transitions that reverse between monthly assessments are not observed.', '',
        'DRE-to-NDRE and NDRE-to-DRE risks are separate conditional tasks. Only the direction matching the starting state is applicable. No-switch risk is 100% minus switch risk; the two direction columns are not complementary class probabilities.', '',
        f"Feature panel: {report['panel']}; {report['panel_candidate_count']} candidates. Primary model uses the top {report['top_features']} source features per task, selected only from training-forest importance. Forest settings: {report['forest_settings']}.", '',
        '## Held-out performance', '',
        '| Task | Test patients | Test visits | Observed switch % | RF Brier | Baseline Brier | RF ROC-AUC | RF average precision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task, row in report['tasks'].items():
        m, baseline = row['models']['selected_features'], row['models']['state_only_baseline']
        lines.append(f"| {task} | {m['patients']} | {m['visits']} | {m['patient_weighted_event_fraction'] * 100:.1f} | {m['brier']:.4f} | {baseline['brier']:.4f} | {_fmt(m['roc_auc'])} | {_fmt(m['average_precision'])} |")
    lines += ['', f"RF columns above refer to the prespecified top-{report['top_features']} model. Brier is the primary probability score; lower is better. Metrics use equal total weight per patient. The state-only baseline uses the training-patient-weighted event fraction for the relevant starting state.", '',
        '## Full versus selected features', '',
        '| Task | Usable full features | Selected features | Full Brier | Selected Brier | Full ROC-AUC | Selected ROC-AUC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task, row in report['tasks'].items():
        full, small = row['models']['all_features'], row['models']['selected_features']
        lines.append(f"| {task} | {row['usable_full_features']} | {row['selected_feature_count']} | {full['brier']:.4f} | {small['brier']:.4f} | {_fmt(full['roc_auc'])} | {_fmt(small['roc_auc'])} |")
    lines += ['', 'The number of retained features was fixed before this comparison; test performance does not choose the feature count, ranking, or primary model.', '',
        '## Effect of change features', '',
        '| Task | Brier with changes | Brier without changes | Paired difference (with minus without) |',
        '|---|---:|---:|---:|']
    for task, row in report['tasks'].items():
        lines.append(f"| {task} | {row['models']['all_features']['brier']:.4f} | {row['models']['without_changes']['brier']:.4f} | {row['change_comparison']['brier_with_changes_minus_without_changes']:.4f} |")
    lines += ['', 'Patient-cluster bootstrap intervals for Brier/log loss and paired Brier differences are in metrics.json. Bootstrap resamples saved test predictions only; no forests are refitted.', '',
        '## Probability outputs and limits', '',
        '- `heldout_risk_predictions.jsonl` contains forecasts for test patients only, including forecasts without an observed future outcome.',
        '- `risk_predictions.jsonl` also includes training-patient forecasts, explicitly marked as in-sample. Do not evaluate generalization from those rows.',
        '- `switch_targets.jsonl` records event labels, observation coverage, and exclusions separately from model features.',
        '- Forest percentages are uncalibrated estimates. Calibration is evaluated, not fitted on test outcomes. Diagnostic calibration intercept/slope do not alter forecasts.',
        '- Unknown starting states have no assigned transition direction or risk. Unknown/censored outcomes are excluded from the affected task, not labeled no-switch.',
        '- Independent 6- and 12-month estimates can cross; raw estimates are preserved and `horizon_order_violation` flags those cases. They are not silently reordered.',
        '- Unreviewed EEG encodings and unusually large doses remain as previously documented in the feature bundle. This initial run is not independent clinical validation.',
        '- `feature_importance.json` ranks every candidate per task using the full training forest. Missingness-indicator importance is added to its source feature; excluded features have zero importance and an exclusion reason. `selected_features.json` lists the chosen source columns.',
        '- Impurity importance can favor some feature types and distribute credit across correlated variables. These exploratory rankings do not establish clinical or causal importance, or selection stability.',
        f"- The top-{report['top_features']} model supplies the risk files, regardless of whether the full model scores better on the test set. Change-feature comparisons use the full panel versus that panel with explicit changes removed.", '',
        '![Calibration](calibration.png)', '', '![ROC](roc.png)', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_training(feature_dir, output_dir, snapshot_dir=None, panel='all', seed=42,
                 trees=500, jobs=4, bootstrap=1000, top_features=30, progress=print):
    output, feature_dir = Path(output_dir), Path(feature_dir)
    if output.exists():
        raise FileExistsError('Choose a new training output directory')
    if panel not in ('core', 'all') or trees < 1 or jobs < 1 or bootstrap < 0 or top_features < 1:
        raise ValueError('Invalid training settings')
    X, visits, dictionary, split, targets, source = load_training_bundle(feature_dir, snapshot_dir)
    candidates = [name for name in X if panel == 'all' or core_feature(name, dictionary[name])]
    without_changes = [name for name in candidates if not change_feature(name)]
    if not candidates or not without_changes:
        raise ValueError('Empty predefined feature panel')
    settings = dict(FOREST_DEFAULTS, n_estimators=trees, n_jobs=jobs, random_state=seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    predictions = visits[['patient_id', 'prediction_date', 'current_dre', 'split']].copy()
    predictions['transition_direction'] = targets.transition_direction
    predictions['prediction_context'] = np.where(visits['split'].eq('train'), 'in_sample_training_patient', 'heldout_test_patient')
    predictions['probability_status'] = 'uncalibrated_random_forest'
    for months in (6, 12):
        predictions[f'followup_end_{months}m'] = targets[f'followup_end_{months}m']
        predictions[f'observed_switch_{months}m'] = targets[f'switch_within_{months}m']
        predictions[f'target_reason_{months}m'] = targets[f'target_reason_{months}m']
        for name in ('risk_any_switch_pct', 'risk_no_switch_pct', 'risk_ndre_to_dre_pct', 'risk_dre_to_ndre_pct'):
            predictions[f'{name}_{months}m'] = np.nan
    report = dict(generated_at_taipei=dt.datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        target='any opposite stored monthly state, including subsequent reversals', panel=panel,
        forest_settings=settings, bootstrap_replicates=bootstrap, split_patient_counts={part: len(split[part]) for part in ('train', 'test')},
        input_feature_count=len(X.columns), panel_candidate_count=len(candidates), top_features=top_features,
        primary_variant='selected_features', feature_selection='training-only grouped impurity importance; fixed top-k',
        starting_state_unknown_visits=int(targets.current_dre.isna().sum()), tasks={})
    feature_audit, importances, curves, selected_lists = {}, {}, {}, {}
    models_registry = {}
    for horizon in (6, 12):
        for current, direction in ((0, 'NDRE_to_DRE'), (1, 'DRE_to_NDRE')):
            task = f'{direction}_{horizon}m'
            origin = targets.current_dre.eq(current).fillna(False)
            observed = targets[f'switch_within_{horizon}m'].notna()
            training = origin & observed & visits['split'].eq('train')
            test = origin & observed & visits['split'].eq('test')
            if not training.any() or not test.any():
                raise ValueError('No train/test observations for ' + task)
            y_train = targets.loc[training, f'switch_within_{horizon}m'].to_numpy(dtype=int)
            y_test = targets.loc[test, f'switch_within_{horizon}m'].to_numpy(dtype=int)
            train_ids, test_ids = visits.loc[training, 'patient_id'], visits.loc[test, 'patient_id']
            row = dict(train_patients=int(train_ids.nunique()), train_visits=int(training.sum()),
                       train_events=int(y_train.sum()), models={})
            report['tasks'][task], curves[task] = row, {}
            # Fit and freeze every model and the subset before test evaluation.
            progress(f'Fitting {task} / all_features: {len(train_ids)} visits, {train_ids.nunique()} training patients')
            full = fit_forest(X.loc[training], y_train, train_ids, candidates, current, horizon, 'all_features', settings)
            ranking, selected_columns = source_importance(full, candidates, top_features)
            importances[task], selected_lists[task] = ranking, selected_columns
            row['usable_full_features'] = len(full.pipeline.named_steps['features'].selected_columns_)
            row['selected_feature_count'] = len(selected_columns)
            fitted = {'all_features': full}
            for variant, columns in (('selected_features', selected_columns), ('without_changes', without_changes)):
                progress(f'Fitting {task} / {variant}: {len(train_ids)} visits, {train_ids.nunique()} training patients')
                fitted[variant] = fit_forest(X.loc[training], y_train, train_ids, columns, current, horizon, variant, settings)
            test_probabilities = {}
            for variant, model in fitted.items():
                p = model.predict_switch_probability(X.loc[test])
                test_probabilities[variant] = p
                metrics = performance(y_test, p, test_ids, bootstrap, seed)
                row['models'][variant] = metrics
                curves[task][variant] = dict(y=y_test, p=p, ids=test_ids.to_numpy(), metrics=metrics)
                name = f'{task}_{variant}'
                path = f'models/{name}.joblib'
                joblib.dump(model, output / path, compress=3)
                models_registry[name] = path
                selected = model.pipeline.named_steps['features']
                transformed_names = model.pipeline.named_steps['imputer'].get_feature_names_out(selected.selected_columns_)
                feature_audit[name] = dict(retained=selected.selected_columns_, dropped_all_missing=selected.all_missing_,
                    dropped_constant=selected.constant_, transformed_columns=transformed_names.tolist())
                if variant == 'selected_features':
                    risk = model.predict_switch_probability(X.loc[origin])
                    percentages = risk_percentages(risk, targets.loc[origin, 'current_dre'])
                    for field, values in percentages.items():
                        predictions.loc[origin, f'{field}_{horizon}m'] = values
                progress(f'Finished {task} / {variant}: test Brier={metrics["brier"]:.4f}, AUC={_fmt(metrics["roc_auc"])}')
            prevalence = float(np.average(y_train, weights=patient_weights(train_ids)))
            baseline_p = np.full(len(y_test), prevalence)
            baseline_metrics = performance(y_test, baseline_p, test_ids, bootstrap, seed)
            row['models']['state_only_baseline'] = baseline_metrics
            row['models']['no_switch_persistence'] = performance(y_test, np.zeros(len(y_test)), test_ids, 0, seed)
            curves[task]['state_only_baseline'] = dict(y=y_test, p=baseline_p, ids=test_ids.to_numpy(), metrics=baseline_metrics)
            row['change_comparison'] = compare_brier(y_test, test_probabilities['all_features'], test_probabilities['without_changes'], test_ids, bootstrap, seed)
    p6, p12 = predictions.risk_any_switch_pct_6m, predictions.risk_any_switch_pct_12m
    predictions['horizon_order_violation'] = (p6 > p12).where(p6.notna() & p12.notna())
    report['horizon_order_violations'] = dict(all_known_start_rows=int((p6 > p12).sum()),
        heldout_rows=int(((p6 > p12) & predictions['split'].eq('test')).sum()))
    targets.to_json(output / 'switch_targets.jsonl', orient='records', lines=True, date_format='iso')
    predictions.to_json(output / 'risk_predictions.jsonl', orient='records', lines=True, date_format='iso')
    predictions.loc[predictions['split'].eq('test')].to_json(output / 'heldout_risk_predictions.jsonl', orient='records', lines=True, date_format='iso')
    file_hashes = {name: hashlib.sha256((feature_dir / name).read_bytes()).hexdigest()
        for name in ('features.pkl.gz', 'visits.jsonl', 'feature_dictionary.json', 'patient_split.json', 'provenance.json')}
    provenance = dict(feature_directory=str(feature_dir.resolve()), snapshot_directory=str(source.resolve()),
        input_sha256=file_hashes, source_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
        code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('train.py', 'forest_workflow.py', 'risk_targets.py')},
        python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__,
        probability_semantics='Applicable directional any-switch probability; opposite direction is not applicable',
        calibration='not fitted; test calibration diagnostics only')
    for name, data in {'metrics.json': report, 'retained_features.json': feature_audit, 'feature_importance.json': importances, 'selected_features.json': selected_lists,
                       'models.json': models_registry, 'provenance.json': provenance, 'patient_split.json': split}.items():
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    write_plots(curves, output)
    write_report(report, output)
    progress('Saved random forest run: ' + str(output.resolve()))
    return report
