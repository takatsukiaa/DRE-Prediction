"""Shared random-forest fitting and evaluation for visit-status models.

All fitting uses saved training patients only. Test outcomes evaluate fixed
models; they never select features, tune thresholds, or fit calibration.
"""
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline


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
class ForestModel:
    pipeline: Pipeline
    variant: str

    def predict_probability(self, X):
        classes = self.pipeline.named_steps['forest'].classes_
        return self.pipeline.predict_proba(X)[:, int(np.where(classes == 1)[0][0])]


def patient_weights(patient_ids):
    ids = pd.Series(patient_ids, copy=False)
    return (1 / ids.map(ids.value_counts())).to_numpy(dtype=float)


def fit_forest(X, y, patient_ids, candidates, variant, settings=None):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Training requires both binary outcome classes')
    pipeline = Pipeline([('features', TrainingFeatureFilter(candidates)),
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('forest', RandomForestClassifier(**(settings or FOREST_DEFAULTS)))])
    pipeline.fit(X, np.asarray(y, dtype=np.int8), forest__sample_weight=patient_weights(patient_ids))
    return ForestModel(pipeline, variant)


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
        f1_at_0_5=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        npv_at_0_5=float(tn / (tn + fn)) if tn + fn else None,
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


def _fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def format_interval(interval, digits=4):
    """Format a saved confidence interval without inventing missing bounds."""
    if interval is None or len(interval) != 2 or any(value is None for value in interval):
        return 'NA'
    return f'[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]'
