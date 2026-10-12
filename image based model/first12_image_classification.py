import argparse
import gc
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from structured_features import STRUCTURED_FEATURE_NAMES


MODALITIES = ("asm", "cognitive", "eeg", "lab", "seizure")
TARGET_SIZE = (384, 768)
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
EXPECTED_OUTER_FOLDS = 4
EXPECTED_PATIENTS_PER_CLASS = 72
EXPECTED_TEST_PATIENTS_PER_CLASS = 18


def resize_with_padding(image_path, target_height, target_width):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    interpolation = cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def image_to_tensor(image):
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255.0)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def load_plot(image_path):
    height, width = TARGET_SIZE
    image = resize_with_padding(image_path, target_height=height, target_width=width)
    return image_to_tensor(image)


class FivePlotDataset(Dataset):
    def __init__(self, manifest, image_root, structured_mean, structured_std):
        self.data = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.structured_mean = np.asarray(structured_mean, dtype=np.float32)
        self.structured_std = np.asarray(structured_std, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sample = {
            modality: load_plot(self.image_root / row[modality])
            for modality in MODALITIES
        }
        structured = row.loc[list(STRUCTURED_FEATURE_NAMES)].to_numpy(dtype=np.float32)
        structured = (structured - self.structured_mean) / self.structured_std
        sample["structured"] = torch.from_numpy(structured)
        sample["target"] = torch.tensor(row["label"], dtype=torch.float32)
        sample["patient_id"] = str(row["patient_id"])
        return sample


class FiveImageStructuredFusionModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_size = self.encoder.num_features
        self.structured_encoder = nn.Sequential(
            nn.Linear(len(STRUCTURED_FEATURE_NAMES), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * len(MODALITIES) + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode(self, batch):
        images = torch.stack([batch[modality] for modality in MODALITIES], dim=1)
        batch_size, modality_count, channels, height, width = images.shape
        images = images.reshape(batch_size * modality_count, channels, height, width)
        features = self.encoder(images)
        return features.reshape(batch_size, modality_count, self.feature_size)

    def forward(self, batch):
        image_features = self.encode(batch).flatten(start_dim=1)
        structured_features = self.structured_encoder(batch["structured"])
        fused = torch.cat([image_features, structured_features], dim=1)
        return self.classifier(fused).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_structured_scaler(table):
    values = table.loc[:, list(STRUCTURED_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Structured feature table contains non-finite values")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def make_loader(
    table,
    image_root,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    seed,
    structured_mean,
    structured_std,
):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FivePlotDataset(table, image_root, structured_mean, structured_std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
    )


def move_inputs_to_device(batch, device):
    inputs = {
        modality: batch[modality].to(device, non_blocking=True)
        for modality in MODALITIES
    }
    inputs["structured"] = batch["structured"].to(device, non_blocking=True)
    return inputs


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp_enabled):
    model.train()
    model.encoder.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        inputs = move_inputs_to_device(batch, device)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


@torch.inference_mode()
def predict_loader(model, loader, criterion, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    patient_ids = []
    targets_list = []
    probabilities_list = []
    for batch in loader:
        inputs = move_inputs_to_device(batch, device)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)
        probabilities = torch.sigmoid(logits)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        patient_ids.extend(batch["patient_id"])
        targets_list.extend(targets.cpu().tolist())
        probabilities_list.extend(probabilities.cpu().tolist())
    return {
        "loss": total_loss / total_samples,
        "patient_ids": patient_ids,
        "targets": np.asarray(targets_list, dtype=int),
        "probabilities": np.asarray(probabilities_list, dtype=float),
    }


def choose_youden_threshold(targets, probabilities):
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    scores = np.where(np.isfinite(thresholds), tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(scores))])


def classification_metrics(targets, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "auc": roc_auc_score(targets, probabilities),
        "average_precision": average_precision_score(targets, probabilities),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predictions": predictions,
    }


def bootstrap_intervals(out_of_fold, iterations, seed):
    rng = np.random.default_rng(seed)
    targets = out_of_fold["label"].to_numpy(dtype=int)
    probabilities = out_of_fold["probability"].to_numpy(dtype=float)
    predictions = out_of_fold["prediction"].to_numpy(dtype=int)
    values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(out_of_fold), len(out_of_fold))
        sample_targets = targets[indices]
        if np.unique(sample_targets).size < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(
            sample_targets,
            predictions[indices],
            labels=[0, 1],
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        values.append(
            (
                roc_auc_score(sample_targets, probabilities[indices]),
                average_precision_score(sample_targets, probabilities[indices]),
                (sensitivity + specificity) / 2,
                sensitivity,
                specificity,
            )
        )
    names = ("auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity")
    array = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "metric": names,
            "ci_low": np.quantile(array, 0.025, axis=0),
            "ci_high": np.quantile(array, 0.975, axis=0),
        }
    )


def validate_manifest(manifest, image_root, require_images):
    required = {
        "patient_id",
        "label",
        "outer_fold",
        *MODALITIES,
        *STRUCTURED_FEATURE_NAMES,
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate patient IDs")
    structured = manifest.loc[:, list(STRUCTURED_FEATURE_NAMES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(structured.to_numpy(dtype=float)).all():
        raise ValueError("Manifest contains missing or non-finite structured features")
    expected_total = EXPECTED_PATIENTS_PER_CLASS * 2
    if len(manifest) != expected_total:
        raise ValueError(f"Expected {expected_total} patients, found {len(manifest)}")
    counts = manifest["label"].value_counts().to_dict()
    if counts.get(0) != EXPECTED_PATIENTS_PER_CLASS or counts.get(1) != EXPECTED_PATIENTS_PER_CLASS:
        raise ValueError(
            f"Expected {EXPECTED_PATIENTS_PER_CLASS} DRE and "
            f"{EXPECTED_PATIENTS_PER_CLASS} NDRE patients, found {counts}"
        )

    fold_table = pd.crosstab(manifest["outer_fold"], manifest["label"])
    if fold_table.shape != (EXPECTED_OUTER_FOLDS, 2):
        raise ValueError(f"Expected four outer folds and two labels, found\n{fold_table}")
    if not (fold_table.to_numpy() == EXPECTED_TEST_PATIENTS_PER_CLASS).all():
        raise ValueError(
            f"Every fold must contain {EXPECTED_TEST_PATIENTS_PER_CLASS} DRE and "
            f"{EXPECTED_TEST_PATIENTS_PER_CLASS} NDRE patients; found\n{fold_table}"
        )
    if require_images:
        image_root = Path(image_root)
        missing_paths = [
            image_root / row[modality]
            for _, row in manifest.iterrows()
            for modality in MODALITIES
            if not (image_root / row[modality]).exists()
        ]
        if missing_paths:
            examples = "\n".join(str(path) for path in missing_paths[:10])
            raise FileNotFoundError(
                f"Missing {len(missing_paths)} dynamic plot files. First examples:\n{examples}"
            )
    return fold_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Four-fold nested validation for earliest-12-month final DRE status prediction."
    )
    parser.add_argument("--manifest", default="first12_final_status_manifest.csv")
    parser.add_argument("--image-root", default="12-month_plots")
    parser.add_argument("--output-prefix", default="results/first12_final_status")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Explicitly authorize the time-consuming model training run.",
    )
    return parser.parse_args()


def run_training(args, manifest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    pin_memory = device.type == "cuda"
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    fold_results = []
    out_of_fold_rows = []
    criterion = nn.BCEWithLogitsLoss()

    for outer_fold in range(EXPECTED_OUTER_FOLDS):
        fold_seed = args.seed + outer_fold
        set_seed(fold_seed)
        development_table = manifest[manifest["outer_fold"] != outer_fold].copy()
        outer_test_table = manifest[manifest["outer_fold"] == outer_fold].copy()
        inner_train_table, inner_validation_table = train_test_split(
            development_table,
            test_size=args.inner_validation_fraction,
            stratify=development_table["label"],
            random_state=fold_seed,
        )
        train_counts = inner_train_table["label"].value_counts().to_dict()
        validation_counts = inner_validation_table["label"].value_counts().to_dict()
        test_counts = outer_test_table["label"].value_counts().to_dict()
        if train_counts.get(0) != train_counts.get(1):
            raise AssertionError(f"Inner training split is not balanced: {train_counts}")
        if validation_counts.get(0) != validation_counts.get(1):
            raise AssertionError(f"Inner validation split is not balanced: {validation_counts}")
        expected_test_counts = {
            0: EXPECTED_TEST_PATIENTS_PER_CLASS,
            1: EXPECTED_TEST_PATIENTS_PER_CLASS,
        }
        if test_counts != expected_test_counts:
            raise AssertionError(
                f"Outer test split is not "
                f"{EXPECTED_TEST_PATIENTS_PER_CLASS}+{EXPECTED_TEST_PATIENTS_PER_CLASS}: "
                f"{test_counts}"
            )

        structured_mean, structured_std = fit_structured_scaler(inner_train_table)

        train_loader = make_loader(
            inner_train_table,
            args.image_root,
            args.batch_size,
            True,
            args.num_workers,
            pin_memory,
            fold_seed,
            structured_mean,
            structured_std,
        )
        inner_validation_loader = make_loader(
            inner_validation_table,
            args.image_root,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
            fold_seed,
            structured_mean,
            structured_std,
        )
        outer_test_loader = make_loader(
            outer_test_table,
            args.image_root,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
            fold_seed,
            structured_mean,
            structured_std,
        )

        model = FiveImageStructuredFusionModel(backbone="resnet18", pretrained=True).to(device)
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        trainable_parameters = [
            *model.structured_encoder.parameters(),
            *model.classifier.parameters(),
        ]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_auc = float("-inf")
        epochs_without_improvement = 0
        checkpoint_path = Path(f"{output_prefix}_best_model_outer_fold_{outer_fold}.pt")

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                amp_enabled,
            )
            inner_results = predict_loader(
                model,
                inner_validation_loader,
                criterion,
                device,
                amp_enabled,
            )
            inner_auc = roc_auc_score(inner_results["targets"], inner_results["probabilities"])
            print(
                f"Outer {outer_fold} epoch {epoch + 1:02d} | "
                f"train loss {train_loss:.4f} | inner loss {inner_results['loss']:.4f} | "
                f"inner AUC {inner_auc:.4f}"
            )
            if inner_auc > best_auc:
                best_auc = inner_auc
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "outer_fold": outer_fold,
                        "epoch": epoch + 1,
                        "inner_validation_auc": inner_auc,
                        "modalities": MODALITIES,
                        "structured_feature_names": STRUCTURED_FEATURE_NAMES,
                        "structured_mean": structured_mean.tolist(),
                        "structured_std": structured_std.tolist(),
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    break

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        inner_results = predict_loader(
            model,
            inner_validation_loader,
            criterion,
            device,
            amp_enabled,
        )
        threshold = choose_youden_threshold(
            inner_results["targets"],
            inner_results["probabilities"],
        )
        outer_results = predict_loader(
            model,
            outer_test_loader,
            criterion,
            device,
            amp_enabled,
        )
        metrics = classification_metrics(
            outer_results["targets"],
            outer_results["probabilities"],
            threshold,
        )
        fold_results.append(
            {
                "fold": outer_fold,
                "best_epoch": checkpoint["epoch"],
                "threshold": threshold,
                "structured_feature_count": len(STRUCTURED_FEATURE_NAMES),
                **{key: value for key, value in metrics.items() if key != "predictions"},
            }
        )
        for patient_id, target, probability, prediction in zip(
            outer_results["patient_ids"],
            outer_results["targets"],
            outer_results["probabilities"],
            metrics["predictions"],
        ):
            out_of_fold_rows.append(
                {
                    "patient_id": patient_id,
                    "fold": outer_fold,
                    "label": int(target),
                    "probability": float(probability),
                    "threshold": threshold,
                    "prediction": int(prediction),
                }
            )
        print(
            f"Outer {outer_fold} | threshold {threshold:.4f} | AUC {metrics['auc']:.4f} | "
            f"AP {metrics['average_precision']:.4f} | sensitivity {metrics['sensitivity']:.4f} | "
            f"specificity {metrics['specificity']:.4f}"
        )
        del model, optimizer, scaler, train_loader, inner_validation_loader, outer_test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_results = pd.DataFrame(fold_results)
    out_of_fold = pd.DataFrame(out_of_fold_rows)
    if len(fold_results) != EXPECTED_OUTER_FOLDS:
        raise AssertionError("Expected four completed outer folds")
    if len(out_of_fold) != len(manifest):
        raise AssertionError("Every selected patient must have one outer-fold prediction")
    if out_of_fold["patient_id"].duplicated().any():
        raise AssertionError("A patient received more than one outer-fold prediction")

    fold_path = Path(f"{output_prefix}_nested_four_fold_results.csv")
    oof_path = Path(f"{output_prefix}_nested_out_of_fold_predictions.csv")
    ci_path = Path(f"{output_prefix}_bootstrap_confidence_intervals.csv")
    fold_results.to_csv(fold_path, index=False)
    out_of_fold.to_csv(oof_path, index=False)
    intervals = bootstrap_intervals(out_of_fold, args.bootstrap_iterations, args.seed)
    intervals.to_csv(ci_path, index=False)
    metric_columns = ["auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity"]
    print("\nOuter-fold mean and standard deviation:")
    print(fold_results[metric_columns].agg(["mean", "std"]))
    print(f"Pooled OOF AUC: {roc_auc_score(out_of_fold['label'], out_of_fold['probability']):.4f}")
    print(f"Pooled OOF AP: {average_precision_score(out_of_fold['label'], out_of_fold['probability']):.4f}")
    print(intervals)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    fold_table = validate_manifest(manifest, args.image_root, require_images=args.run_training)
    print("Validated balanced four-fold manifest:")
    print(fold_table)
    if not args.run_training:
        print("Training was not started. Pass --run-training explicitly when ready.")
        return
    run_training(args, manifest)


if __name__ == "__main__":
    main()
