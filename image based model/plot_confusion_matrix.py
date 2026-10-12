"""Draw a confusion matrix from an out-of-fold prediction CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ("NDRE", "DRE")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="results/first12_final_status_nested_out_of_fold_predictions.csv",
        help="CSV containing label and prediction columns.",
    )
    parser.add_argument(
        "--output",
        default="results/first12_final_status_confusion_matrix_python.png",
        help="Destination PNG path.",
    )
    return parser.parse_args()


def load_confusion_matrix(csv_path):
    predictions = pd.read_csv(csv_path)
    required = {"label", "prediction"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")

    labels = pd.to_numeric(predictions["label"], errors="raise").astype(int)
    predicted = pd.to_numeric(predictions["prediction"], errors="raise").astype(int)
    observed_values = set(labels.unique()) | set(predicted.unique())
    if not observed_values.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, found {sorted(observed_values)}")
    return confusion_matrix(labels, predicted, labels=[0, 1])


def draw_confusion_matrix(matrix, output_path):
    matrix = np.asarray(matrix, dtype=int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    tn, fp, fn, tp = matrix.ravel()
    total = matrix.sum()
    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Patients", rotation=270, labelpad=18)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted status",
        ylabel="True status",
        title="DRE Prediction Confusion Matrix",
    )
    ax.xaxis.set_label_position("bottom")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]}\n{row_percent[row, column]:.1%} of true {CLASS_NAMES[row]}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="semibold",
                color="white" if matrix[row, column] > threshold else "#17324d",
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=11)
    fig.text(
        0.5,
        0.025,
        f"N = {total}   Accuracy = {accuracy:.1%}   Sensitivity = {sensitivity:.1%}   Specificity = {specificity:.1%}",
        ha="center",
        fontsize=10.5,
        color="#334e68",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    matrix = load_confusion_matrix(args.predictions)
    output_path = draw_confusion_matrix(matrix, args.output)
    print(matrix)
    print(f"Saved confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
