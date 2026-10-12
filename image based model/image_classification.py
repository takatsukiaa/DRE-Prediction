import argparse
import gc
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import timm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

MODALITIES = [
    "asm",
    "cognitive",
    "eeg",
    "lab",
    "seizure",
]
target_size = (384,768)  # Target size for resizing images

def resize_with_padding(
    image_path,
    target_height,
    target_width,
):
    # OpenCV loads images as BGR
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    # White background matches your plots
    canvas = np.full(
        (target_height, target_width, 3),
        fill_value=255,
        dtype=np.uint8,
    )

    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2

    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    # Convert BGR to RGB for pretrained PyTorch models
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return canvas

def image_to_tensor(image):
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    tensor = tensor / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return (tensor - mean) / std


def load_plot(image_path):
    height, width = target_size

    image = resize_with_padding(
        image_path,
        target_height=height,
        target_width=width,
    )

    return image_to_tensor(image)

class FivePlotDataset(Dataset):
    def __init__(self, manifest, image_root):
        self.data = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        sample = {}
        for modality in MODALITIES:
            image_path = self.image_root / row[modality]
            sample[modality] = load_plot(image_path)


        sample["target"] = torch.tensor(
            row["label"],
            dtype=torch.float32,
        )

        # Keep ID for grouping and reporting; don't feed it to the model.
        sample["patient_id"] = str(row["patient_id"])

        return sample
from torch import nn

class FiveImageFusionModel(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        pretrained=True,
        dropout=0.3,
    ):
        super().__init__()

        # num_classes=0 removes the ImageNet classifier and returns features.
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_size = self.encoder.num_features

        # Five encoder outputs are concatenated.
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * len(MODALITIES), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode(self, batch):
        # Each item: [B, 3, 384, 768]
        images = torch.stack(
            [batch[modality] for modality in MODALITIES],
            dim=1,
        )

        # [B, 5, 3, 384, 768]
        batch_size, modalities, channels, height, width = images.shape

        # Treat the five images as B*5 images for one encoder call.
        images = images.reshape(
            batch_size * modalities,
            channels,
            height,
            width,
        )

        # ResNet-18 output: [B*5, 512]
        features = self.encoder(images)

        # Restore modality dimension: [B, 5, 512]
        features = features.reshape(
            batch_size,
            modalities,
            self.feature_size,
        )

        return features

    def forward(self, batch):
        features = self.encode(batch)

        # Preserve modality identity through fixed concatenation order.
        # [B, 5, 512] -> [B, 2560]
        fused_features = features.flatten(start_dim=1)

        # Output raw binary logits: [B]
        logits = self.classifier(fused_features).squeeze(1)

        return logits
   

def main():
    from torch.utils.data import DataLoader

    manifest = pd.read_csv(
        "patient_manifest.csv",
        dtype={"patient_id": str},
    )

    out_of_fold_rows = []
    fold_results = []
    for outer_fold in range(5):
        print(f"\nOuter fold {outer_fold}")

        development_table = manifest[
            manifest["fold"] != outer_fold
        ].copy()

        outer_test_table = manifest[
            manifest["fold"] == outer_fold
        ].copy()

        inner_train_table, inner_validation_table = train_test_split(
            development_table,
            test_size=0.20,
            stratify=development_table["label"],
            random_state=42 + outer_fold,
        )

        print(
            "Inner train:",
            len(inner_train_table),
            inner_train_table["label"].value_counts().to_dict(),
        )
        print(
            "Inner validation:",
            len(inner_validation_table),
            inner_validation_table["label"].value_counts().to_dict(),
        )
        print(
            "Outer test:",
            len(outer_test_table),
            outer_test_table["label"].value_counts().to_dict(),
        )

        inner_train_dataset = FivePlotDataset(
            inner_train_table,
            "36-month_plots",
        )

        inner_validation_dataset = FivePlotDataset(
            inner_validation_table,
            "36-month_plots",
        )

        outer_test_dataset = FivePlotDataset(
            outer_test_table,
            "36-month_plots",
        )

        inner_train_loader = DataLoader(
            inner_train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        inner_validation_loader = DataLoader(
            inner_validation_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        outer_test_loader = DataLoader(
            outer_test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")



        model = FiveImageFusionModel(
            backbone="resnet18",
            pretrained=True,
        ).to(device)

        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        positive_count = (inner_train_table["label"] == 1).sum()
        negative_count = (inner_train_table["label"] == 0).sum()

        positive_weight = torch.tensor(
            negative_count / positive_count,
            dtype=torch.float32,
            device=device,
        )

        train_criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight,
        )

        validation_criterion = nn.BCEWithLogitsLoss()

        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            confusion_matrix,
            roc_auc_score,
        )

        amp_enabled = device.type == "cuda"

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
        def move_images_to_device(batch):
            return {
                modality: batch[modality].to(
                    device,
                    non_blocking=True,
                )
                for modality in MODALITIES
            }


        def train_one_epoch():
            model.train()

            # Encoder is frozen, so BatchNorm statistics should not change.
            model.encoder.eval()

            total_loss = 0.0
            total_samples = 0

            for batch in train_loader:
                images = move_images_to_device(batch)

                targets = batch["target"].to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                    loss = train_criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            return total_loss / total_samples


        @torch.inference_mode()
        def predict_loader(model, loader, criterion):
            model.eval()

            total_loss = 0.0
            total_samples = 0

            patient_ids = []
            targets_list = []
            probabilities_list = []

            for batch in loader:
                images = move_images_to_device(batch)

                targets = batch["target"].to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
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
                "probabilities": np.asarray(
                    probabilities_list,
                    dtype=float,
                ),
            }



        number_of_epochs = 15
        early_stopping_patience = 5

        best_auc = float("-inf")
        epochs_without_improvement = 0

        for epoch in range(number_of_epochs):
            train_loss = train_one_epoch()

            inner_results = predict_loader(
                model,
                inner_validation_loader,
                validation_criterion,
            )

            inner_auc = roc_auc_score(
                inner_results["targets"],
                inner_results["probabilities"],
            )

            print(
                f"Epoch {epoch + 1:02d} | "
                f"train loss: {train_loss:.4f} | "
                f"inner val loss: {inner_results['loss']:.4f} | "
                f"inner val AUC: {inner_auc:.4f}"
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
                    },
                    f"best_model_outer_fold_{outer_fold}.pt",
                )
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    break

        checkpoint = torch.load(
                f"best_model_outer_fold_{outer_fold}.pt",
                map_location=device,
            )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        inner_results = predict_loader(
            model,
            inner_validation_loader,
            validation_criterion,
        )
    from sklearn.metrics import roc_curve


    def choose_youden_threshold(targets, probabilities):
        fpr, tpr, thresholds = roc_curve(
            targets,
            probabilities,
        )

        youden_scores = tpr - fpr

        # roc_curve can include an infinite first threshold.
        valid = np.isfinite(thresholds)

        best_index = np.argmax(
            np.where(valid, youden_scores, -np.inf)
        )

        return float(thresholds[best_index])


    selected_threshold = choose_youden_threshold(
        inner_results["targets"],
        inner_results["probabilities"],
    )

    print(
        f"Selected threshold: {selected_threshold:.4f}"
    )
    outer_results = predict_loader(
        model,
        outer_test_loader,
        validation_criterion,
    )

    outer_targets = outer_results["targets"]
    outer_probabilities = outer_results["probabilities"]

    outer_predictions = (
        outer_probabilities >= selected_threshold
    ).astype(int)

    outer_auc = roc_auc_score(
        outer_targets,
        outer_probabilities,
    )

    outer_ap = average_precision_score(
        outer_targets,
        outer_probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        outer_targets,
        outer_predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2

    print(
        f"Outer fold {outer_fold} | "
        f"threshold: {selected_threshold:.4f} | "
        f"AUC: {outer_auc:.4f} | "
        f"AP: {outer_ap:.4f} | "
        f"balanced accuracy: {balanced_accuracy:.4f} | "
        f"sensitivity: {sensitivity:.4f} | "
        f"specificity: {specificity:.4f}"
    )

    #     best_results = validate()

    #     fold_results.append({
    #         "fold": validation_fold,
    #         "best_epoch": checkpoint["epoch"],
    #         "auc": best_results["auc"],
    #         "average_precision": best_results["average_precision"],
    #         "balanced_accuracy": best_results["balanced_accuracy"],
    #         "sensitivity": best_results["sensitivity"],
    #         "specificity": best_results["specificity"],
    #     })

    #     for patient_id, target, probability in zip(
    #         best_results["patient_ids"],
    #         best_results["targets"],
    #         best_results["probabilities"],
    #     ):
    #         out_of_fold_rows.append({
    #             "patient_id": patient_id,
    #             "fold": validation_fold,
    #             "label": int(target),
    #             "probability": probability,
    #         })
    # fold_results = pd.DataFrame(fold_results)
    # out_of_fold = pd.DataFrame(out_of_fold_rows)

    # fold_results.to_csv("five_fold_results.csv", index=False)
    # out_of_fold.to_csv("out_of_fold_predictions.csv", index=False)

    # print(fold_results)
    # print("\nMean fold results:")
    # print(fold_results.mean(numeric_only=True))
    # print("\nStandard deviations:")
    # print(fold_results.std(numeric_only=True))

    # overall_auc = roc_auc_score(
    #     out_of_fold["label"],
    #     out_of_fold["probability"],
    # )

    # overall_ap = average_precision_score(
    #     out_of_fold["label"],
    #     out_of_fold["probability"],
    # )

    # print(f"Overall out-of-fold AUC: {overall_auc:.4f}")
    # print(f"Overall out-of-fold AP: {overall_ap:.4f}")
if __name__ == '__main__':
    main()
