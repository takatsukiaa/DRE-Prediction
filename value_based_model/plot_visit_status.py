"""Render visit-status ROC curves and confusion matrices from saved results."""
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


CLASS_NAMES = ('DRE', 'NDRE')
MODEL_NAMES = {'all_features': 'All features', 'selected_features': 'Selected features',
               'training_prevalence_baseline': 'Training prevalence baseline'}


def confusion_matrix_figure(metrics, title):
    """Show raw held-out visit counts and percentages within each observed class."""
    matrix = np.asarray(metrics['confusion_matrix_raw'])
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or (matrix < 0).any() or (matrix != np.floor(matrix)).any()):
        raise ValueError('Expected a 2-by-2 confusion matrix of nonnegative visit counts')
    # Saved metrics use [NDRE, DRE]; display DRE first on both axes.
    matrix = matrix.astype(np.int64)[np.ix_([1, 0], [1, 0])]
    totals = matrix.sum(axis=1)
    figure = Figure(figsize=(6.6, 6.2), facecolor='white')
    FigureCanvasAgg(figure)
    ax = figure.add_axes([.21, .07, .60, .72])
    image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(int(matrix.max()), 1),
                      aspect='auto', interpolation='nearest')
    colorbar = figure.colorbar(image, cax=figure.add_axes([.84, .07, .025, .72]))
    colorbar.set_label('Visit count', fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel='Predicted', ylabel='Actual')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.xaxis.label.set(size=14)
    ax.yaxis.label.set(size=14)
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.tick_params(axis='both', which='major', length=0, pad=8, labelsize=12)
    for tick in ax.get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment('center')
    ax.set_xticks([-.5, .5, 1.5], minor=True)
    ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for row in range(2):
        for column in range(2):
            percent = f'{matrix[row, column] / totals[row]:.1%}' if totals[row] else 'N/A'
            color = 'white' if matrix[row, column] > image.norm.vmax / 2 else '#17324d'
            ax.text(column, row, f'{matrix[row, column]:,}\n({percent})', ha='center', va='center',
                    fontsize=14, color=color)
    figure.text(.58, .98, title, ha='center', va='top', fontsize=12, color='#334e68')
    return figure


def write_confusion_matrices(report, output_dir):
    """Save one PNG per history/model, including the prevalence baselines."""
    if report.get('task') != 'visit_status':
        raise ValueError('Confusion matrix plotting requires visit-status metrics')
    output = Path(output_dir)
    plots = output / 'confusion_matrices'
    plots.mkdir(parents=True, exist_ok=True)
    paths = {}
    active_tasks = {f'visit_status_history_{months}m' for months in HISTORY_WINDOWS}
    for task, result in report['tasks'].items():
        if task not in active_tasks:
            continue
        paths[task] = {}
        history = task.rsplit('_', 1)[-1]
        for variant, metrics in result['models'].items():
            path = plots / f'{task}_{variant}.png'
            title = f'{history} history | {MODEL_NAMES.get(variant, variant)}'
            figure = confusion_matrix_figure(metrics, title)
            try:
                figure.savefig(path, dpi=330, facecolor='white')
            finally:
                figure.clear()
            paths[task][variant] = path.relative_to(output).as_posix()
    return paths


def roc_comparison_figure(predictions):
    """Use one common labeled cohort and the evaluation's equal-patient weights."""
    columns = {months: f'risk_dre_pct_history_{months}m' for months in HISTORY_WINDOWS}
    required = {'patient_id', 'split', 'prediction_scope', 'observed_status_at_visit', *columns.values()}
    if not required.issubset(predictions.columns):
        raise ValueError('ROC comparison requires held-out predictions for all three history models')
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
    colors = ('#0072B2', '#D55E00', '#009E73')
    for index, (months, color) in enumerate(zip(HISTORY_WINDOWS, colors)):
        if two_classes:
            probability = probabilities[:, index]
            fpr, tpr, _ = roc_curve(y, probability, sample_weight=weights, drop_intermediate=False)
            area = float(roc_auc_score(y, probability, sample_weight=weights))
        else:
            fpr, tpr, area = np.array([]), np.array([]), None
        data['curves'][f'{months}m'] = dict(roc_auc=area, false_positive_rate=fpr.tolist(), true_positive_rate=tpr.tolist())
        label = f'{months}-month history (AUROC = {area:.4f})' if area is not None else f'{months}-month history (AUROC undefined)'
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color='#77828c', linestyle='--', linewidth=1.2, label='Chance (AUROC = 0.5000)')
    if not two_classes:
        ax.text(.5, .55, 'ROC curves are undefined:\nheld-out labels contain only one class.',
                ha='center', va='center', transform=ax.transAxes)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate (1 - specificity)',
           ylabel='True positive rate (sensitivity)', title='Held-out ROC comparison\nSelected-feature models | DRE positive')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', framealpha=.95, fontsize=14)
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
    parser.add_argument('--results-dir', type=Path, required=True, help='Directory containing visit-status metrics.json')
    parser.add_argument('--plot', choices=('all', 'confusion', 'roc'), default='all')
    args = parser.parse_args()
    root = args.results_dir
    report = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
    if args.plot != 'confusion':
        if report.get('task') != 'visit_status' or report.get('history_windows_months') != list(HISTORY_WINDOWS):
            raise ValueError('ROC comparison requires visit-status results for all three history models')
        predictions = pd.read_json(root / 'heldout_status_predictions.jsonl', lines=True,
                                   dtype={'patient_id': 'string'}, convert_dates=False)
        report['roc_comparison'] = write_roc_comparison(predictions, root)
        print(f'Saved ROC comparison to {(root / report["roc_comparison"]["plot"]).resolve()}')
    if args.plot != 'roc':
        paths = write_confusion_matrices(report, root)
        print(f'Saved {sum(len(models) for models in paths.values())} confusion matrices to '
              f'{(root / "confusion_matrices").resolve()}')
        if args.plot == 'all':
            report['confusion_matrix_plots'] = paths
    if args.plot != 'confusion':
        write_json(root / 'metrics.json', report)
        if 'latest_visit_cohort' in report:
            from evaluate_latest_visits import write_latest_report
            write_latest_report(report, root)
        else:
            from visit_status_workflow import write_status_report
            write_status_report(report, root)


if __name__ == '__main__':
    main()
