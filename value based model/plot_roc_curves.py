"""Compare all four primary visit-status models using held-out ROC curves."""
import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from build_visit_status import write_json
from forest_workflow import patient_weights
from longitudinal_features import HISTORY_WINDOWS


def roc_comparison_figure(predictions):
    """Use one common labeled cohort and the evaluation's equal-patient weights."""
    columns = {months: f'risk_dre_pct_history_{months}m' for months in HISTORY_WINDOWS}
    required = {'patient_id', 'split', 'prediction_scope', 'observed_status_at_visit', *columns.values()}
    if not required.issubset(predictions.columns):
        raise ValueError('ROC comparison requires held-out predictions for all four history models')
    if not (predictions['split'].eq('test') & predictions.prediction_scope.eq('heldout_patient')).all():
        raise ValueError('ROC comparison accepts held-out test patients only')
    labeled = predictions.loc[predictions.observed_status_at_visit.notna()]
    if labeled.empty or not labeled.observed_status_at_visit.isin([0, 1]).all():
        raise ValueError('ROC comparison needs known binary visit labels')
    if labeled.patient_id.isna().any():
        raise ValueError('ROC comparison needs patient IDs for evaluation weights')
    probabilities = labeled[list(columns.values())].to_numpy(dtype=float) / 100
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('All history models need finite DRE percentages between 0 and 100 on the common cohort')
    y = labeled.observed_status_at_visit.to_numpy(dtype=int)
    weights = patient_weights(labeled.patient_id)
    two_classes = len(np.unique(y)) == 2
    data = dict(variant='selected_features', positive_class='DRE',
        history_windows_months=list(HISTORY_WINDOWS),
        weighting='Each patient has equal total weight across labeled held-out visits',
        labeled_visits=len(labeled), patients=int(labeled.patient_id.nunique()), curves={})

    figure = Figure(figsize=(7.6, 7.2), facecolor='white', layout='constrained')
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    for index, (months, color) in enumerate(zip(HISTORY_WINDOWS, colors)):
        if two_classes:
            probability = probabilities[:, index]
            fpr, tpr, _ = roc_curve(y, probability, sample_weight=weights, drop_intermediate=False)
            area = float(roc_auc_score(y, probability, sample_weight=weights))
        else:
            fpr, tpr, area = np.array([]), np.array([]), None
        data['curves'][f'{months}m'] = dict(roc_auc=area, false_positive_rate=fpr.tolist(), true_positive_rate=tpr.tolist())
        label = f'{months}-month history (AUROC = {area:.3f})' if area is not None else f'{months}-month history (AUROC undefined)'
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color='#77828c', linestyle='--', linewidth=1.2, label='Chance (AUROC = 0.500)')
    if not two_classes:
        ax.text(.5, .55, 'ROC curves are undefined:\nheld-out labels contain only one class.',
                ha='center', va='center', transform=ax.transAxes)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate (1 - specificity)',
           ylabel='True positive rate (sensitivity)', title='Held-out ROC comparison\nSelected-feature models | DRE positive')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', framealpha=.95, fontsize=10)
    figure.supxlabel(f'{len(labeled):,} labeled visits | {labeled.patient_id.nunique():,} held-out patients | Equal patient weights', fontsize=10)
    return figure, data


def write_roc_comparison(predictions, output_dir):
    output = Path(output_dir)
    figure, data = roc_comparison_figure(predictions)
    paths = dict(plot='roc_comparison.png', data='roc_curves.json', variant='selected_features')
    try:
        figure.savefig(output / paths['plot'], dpi=330, facecolor='white')
    finally:
        figure.clear()
    write_json(output / paths['data'], data)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.results_dir / 'metrics.json').read_text(encoding='utf-8'))
    if report.get('task') != 'visit_status' or report.get('history_windows_months') != list(HISTORY_WINDOWS):
        raise ValueError('ROC comparison requires visit-status results for all four history models')
    predictions = pd.read_json(args.results_dir / 'heldout_status_predictions.jsonl', lines=True,
                               dtype={'patient_id': 'string'}, convert_dates=False)
    report['roc_comparison'] = write_roc_comparison(predictions, args.results_dir)
    write_json(args.results_dir / 'metrics.json', report)
    from visit_status_workflow import write_status_report
    write_status_report(report, args.results_dir)
    print('Saved ROC comparison: ' + str((args.results_dir / report['roc_comparison']['plot']).resolve()))


if __name__ == '__main__':
    main()
