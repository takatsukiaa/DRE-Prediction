"""Plot every visit-status result from saved metrics, without retraining."""
import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

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
    for task, result in report['tasks'].items():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True, help='Directory containing visit-status metrics.json')
    args = parser.parse_args()
    report = json.loads((args.results_dir / 'metrics.json').read_text(encoding='utf-8'))
    paths = write_confusion_matrices(report, args.results_dir)
    print(f'Saved {sum(len(models) for models in paths.values())} confusion matrices to '
          f'{(args.results_dir / "confusion_matrices").resolve()}')


if __name__ == '__main__':
    main()
