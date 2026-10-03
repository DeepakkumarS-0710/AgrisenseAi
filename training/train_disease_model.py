"""Fine-tune the leaf classifier with full PlantVillage and optional PlantDoc data."""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PLANTDOC_ALIASES = {
    "applescableaf": "Apple___Apple_scab",
    "appleleafscab": "Apple___Apple_scab",
    "applerustleaf": "Apple___Cedar_apple_rust",
    "bellpepperleafspot": "Pepper,_bell___Bacterial_spot",
    "corngrayleafspot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "cornleafblight": "Corn_(maize)___Northern_Leaf_Blight",
    "cornrustleaf": "Corn_(maize)___Common_rust_",
    "grapeleafblackrot": "Grape___Black_rot",
    "potatoleafearlyblight": "Potato___Early_blight",
    "potatoleaflateblight": "Potato___Late_blight",
    "squashpowderymildewleaf": "Squash___Powdery_mildew",
    "tomatoearlyblightleaf": "Tomato___Early_blight",
    "tomatoleafbacterialspot": "Tomato___Bacterial_spot",
    "tomatoleaflateblight": "Tomato___Late_blight",
    "tomatoleafmosaicvirus": "Tomato___Tomato_mosaic_virus",
    "tomatoleafyellowvirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomatomoldleaf": "Tomato___Leaf_Mold",
    "tomatoseptorialeafspot": "Tomato___Septoria_leaf_spot",
    "tomatotwospottedspidermitesleaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_plantvillage_root(data_dir):
    candidates = (
        data_dir / "plantvillage" / "PlantVillage",
        data_dir / "plantvillage",
        data_dir / "PlantVillage",
    )
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
        if candidate.is_dir() and any(
            child.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in child.iterdir() if p.is_file())
            for child in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(
        "PlantVillage class folders were not found. Expected "
        "data/plantvillage/PlantVillage/train/<class> or "
        "data/plantvillage/<class>."
    )


def list_class_images(root, include_validation_split=True):
    classes = {}
    if not root or not root.is_dir():
        return classes
    split_roots = [root]
    if (root / "train").is_dir():
        # Some PlantVillage copies include a separate validation split. Merge
        # it into the image pool, then create fresh stratified train/val/test
        # partitions below so the supplied split is not silently discarded.
        split_roots = [root / "train"]
        if include_validation_split and (root / "val").is_dir():
            split_roots.append(root / "val")
    gathered = defaultdict(list)
    for split_root in split_roots:
        for folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
            files = sorted(
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if files:
                gathered[folder.name].extend(files)
    for name, files in gathered.items():
        classes[name] = sorted(files)
    return classes


def split_by_class(class_files, class_to_index, cap_per_class=0, seed=SEED):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    rng = random.Random(seed)

    for name in sorted(class_files):
        if name not in class_to_index:
            continue
        files = list(class_files[name])
        rng.shuffle(files)
        if cap_per_class > 0:
            files = files[:cap_per_class]
        if len(files) < 5:
            raise ValueError(f"Class {name!r} has too few images for safe train/validation/test splits.")
        train, holdout = train_test_split(files, test_size=0.20, random_state=seed)
        val, test = train_test_split(holdout, test_size=0.50, random_state=seed)
        label = class_to_index[name]
        train_paths.extend(train)
        train_labels.extend([label] * len(train))
        val_paths.extend(val)
        val_labels.extend([label] * len(val))
        test_paths.extend(test)
        test_labels.extend([label] * len(test))
    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def normalize_plantdoc_class(name):
    folded = re.sub(r"[^a-z0-9]", "", name.lower())
    return PLANTDOC_ALIASES.get(folded)


def plantdoc_data(data_dir, class_to_index, cap_per_class=0, seed=SEED):
    candidates = (
        data_dir / "plantdoc" / "PlantDoc-Dataset",
        data_dir / "plantdoc",
        data_dir / "PlantDoc-Dataset",
    )
    root = next((path for path in candidates if (path / "train").is_dir()), None)
    if root is None:
        return None, None, {"path": None, "train_images": 0, "test_images": 0, "mapped_classes": []}

    train_dirs = list_class_images(root / "train")
    external_test_dirs = list_class_images(root / "test")

    mapped_train = defaultdict(list)
    mapped_test = defaultdict(list)
    ignored = []
    for source, destination in ((train_dirs, mapped_train), (external_test_dirs, mapped_test)):
        for raw_name, files in source.items():
            canonical = normalize_plantdoc_class(raw_name)
            if canonical in class_to_index:
                destination[canonical].extend(files)
            elif raw_name not in ignored:
                ignored.append(raw_name)

    rng = random.Random(seed)
    field_train_paths, field_train_labels = [], []
    field_val_paths, field_val_labels = [], []
    field_test_paths, field_test_labels = [], []

    for name in sorted(mapped_train):
        files = list(mapped_train[name])
        rng.shuffle(files)
        if cap_per_class > 0:
            files = files[:cap_per_class]
        if len(files) < 3:
            continue
        train, val = train_test_split(files, test_size=max(1, round(len(files) * 0.15)), random_state=seed)
        label = class_to_index[name]
        field_train_paths.extend(train)
        field_train_labels.extend([label] * len(train))
        field_val_paths.extend(val)
        field_val_labels.extend([label] * len(val))

    for name, files in sorted(mapped_test.items()):
        label = class_to_index[name]
        field_test_paths.extend(files)
        field_test_labels.extend([label] * len(files))

    metadata = {
        "path": str(root),
        "train_images": len(field_train_paths),
        "validation_images": len(field_val_paths),
        "test_images": len(field_test_paths),
        "mapped_classes": sorted({
            *[name for name, files in mapped_train.items() if files],
            *[name for name, files in mapped_test.items() if files],
        }),
        "unmapped_classes": sorted(ignored),
    }
    return (field_train_paths, field_train_labels), (field_val_paths, field_val_labels), {
        **metadata,
        "test": (field_test_paths, field_test_labels),
    }


class PlantImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), self.labels[index]


def build_model(num_classes, checkpoint_path, class_names, model_dir, device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    mapping_path = model_dir / "class_names.json"
    old_names = []
    if mapping_path.is_file():
        with mapping_path.open(encoding="utf-8") as handle:
            mapping = json.load(handle)
        old_names = [mapping[key] for key in sorted(mapping, key=lambda item: int(item))]
    if checkpoint_path.is_file() and old_names == class_names:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"Starting from the existing trained checkpoint: {checkpoint_path}", flush=True)
    else:
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        print("Starting from ImageNet-pretrained MobileNetV3-Small.", flush=True)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss, total, predictions, targets = 0.0, 0, [], []
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            targets.extend(labels.detach().cpu().numpy().tolist())
            total += labels.size(0)
    return (
        total_loss / max(total, 1),
        accuracy_score(targets, predictions) if total else 0.0,
        f1_score(targets, predictions, average="macro", zero_division=0) if total else 0.0,
        predictions,
        targets,
    )


def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels)
    if not all_logits:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits, labels):
    if not len(labels):
        return 1.0
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=60)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.5, 5.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.5, 5.0).item())


def report_for(logits, labels, class_names, temperature=1.0):
    if not len(labels):
        return None, None, []
    predictions = (logits / temperature).argmax(dim=1).numpy()
    targets = labels.numpy()
    evaluated_labels = sorted(set(int(label) for label in targets))
    report = classification_report(
        targets,
        predictions,
        labels=evaluated_labels,
        target_names=[class_names[index] for index in evaluated_labels],
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "images": int(len(targets)),
        "classes_evaluated": len(evaluated_labels),
        "accuracy": round(float(accuracy_score(targets, predictions)) * 100, 2),
        "macro_f1": round(float(f1_score(
            targets,
            predictions,
            labels=evaluated_labels,
            average="macro",
            zero_division=0,
        )) * 100, 2),
    }
    return summary, report, confusion_matrix(
        targets, predictions, labels=evaluated_labels
    ).tolist()


def train_disease_model(data_dir, epochs=5, batch_size=32, max_samples_per_class=0,
                        max_field_samples_per_class=0, num_workers=0,
                        field_sample_weight=4.0,
                        include_plantvillage_val=True):
    seed_everything()
    data_dir = Path(data_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models" / "disease_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    plantvillage_root = find_plantvillage_root(data_dir)
    pv_files = list_class_images(
        plantvillage_root,
        include_validation_split=include_plantvillage_val,
    )
    class_names = sorted(pv_files)
    if not class_names:
        raise ValueError(f"No labeled image folders were found under {plantvillage_root}.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    pv_train, pv_val, pv_test = split_by_class(
        pv_files, class_to_index, cap_per_class=max_samples_per_class
    )
    field_train, field_val, field_metadata = plantdoc_data(
        data_dir, class_to_index, cap_per_class=max_field_samples_per_class
    )
    if field_train is None:
        field_train = ([], [])
        field_val = ([], [])
        field_test = ([], [])
    else:
        field_test = field_metadata.pop("test")

    train_paths = pv_train[0] + field_train[0]
    train_labels = pv_train[1] + field_train[1]
    val_paths = pv_val[0] + field_val[0]
    val_labels = pv_val[1] + field_val[1]
    print(
        "Images — "
        f"PlantVillage train/val/test: {len(pv_train[0])}/{len(pv_val[0])}/{len(pv_test[0])}; "
        f"field train/val/test: {len(field_train[0])}/{len(field_val[0])}/{len(field_test[0])}",
        flush=True,
    )
    print(f"Recognized {len(class_names)} PlantVillage labels.", flush=True)
    if field_metadata.get("unmapped_classes"):
        print(
            "Leaving ambiguous PlantDoc categories out of training: "
            + ", ".join(field_metadata["unmapped_classes"]),
            flush=True,
        )

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.68, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = PlantImageDataset(train_paths, train_labels, train_transform)
    val_set = PlantImageDataset(val_paths, val_labels, validation_transform)
    field_val_set = PlantImageDataset(field_val[0], field_val[1], validation_transform)
    pv_test_set = PlantImageDataset(pv_test[0], pv_test[1], validation_transform)
    field_test_set = PlantImageDataset(field_test[0], field_test[1], validation_transform)

    label_counts = np.bincount(train_labels, minlength=len(class_names))
    sample_weights = [
        (1.0 / np.sqrt(label_counts[label]))
        * (field_sample_weight if index >= len(pv_train[0]) else 1.0)
        for index, label in enumerate(train_labels)
    ]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    loader_options = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, **loader_options
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, **loader_options
    )
    field_val_loader = DataLoader(
        field_val_set, batch_size=batch_size, shuffle=False, **loader_options
    )
    pv_test_loader = DataLoader(
        pv_test_set, batch_size=batch_size, shuffle=False, **loader_options
    )
    field_test_loader = DataLoader(
        field_test_set, batch_size=batch_size, shuffle=False, **loader_options
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    print(f"Training on {device}; batch size {batch_size}.", flush=True)
    if field_train[0]:
        print(
            f"PlantDoc samples receive {field_sample_weight:g}x sampling weight; "
            "checkpoint selection balances PlantVillage and field validation macro F1.",
            flush=True,
        )
    checkpoint_path = model_dir / "best_model.pth"
    model = build_model(
        len(class_names), checkpoint_path, class_names, model_dir, device
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    warmup_epochs = min(1, max(epochs - 1, 0))
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=2e-4, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    history = []
    best_macro_f1 = -1.0
    best_validation_summary = {}

    for epoch in range(1, epochs + 1):
        if epoch == warmup_epochs + 1 and warmup_epochs:
            for parameter in model.features[-5:].parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.features[-5:].parameters(), "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ], weight_decay=1e-4)

        train_loss, train_accuracy, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_accuracy, val_macro_f1, _, _ = run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        if len(field_val_set):
            field_val_loss, field_val_accuracy, field_val_macro_f1, _, _ = run_epoch(
                model, field_val_loader, criterion, None, device, training=False
            )
            selection_macro_f1 = (val_macro_f1 + field_val_macro_f1) / 2
        else:
            field_val_loss = field_val_accuracy = field_val_macro_f1 = None
            selection_macro_f1 = val_macro_f1
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_accuracy * 100, 2),
            "validation_loss": round(val_loss, 5),
            "validation_accuracy": round(val_accuracy * 100, 2),
            "validation_macro_f1": round(val_macro_f1 * 100, 2),
            "field_validation_loss": round(field_val_loss, 5) if field_val_loss is not None else None,
            "field_validation_accuracy": round(field_val_accuracy * 100, 2) if field_val_accuracy is not None else None,
            "field_validation_macro_f1": round(field_val_macro_f1 * 100, 2) if field_val_macro_f1 is not None else None,
            "checkpoint_selection_macro_f1": round(selection_macro_f1 * 100, 2),
        })
        epoch_message = (
            f"Epoch {epoch}/{epochs} — train accuracy {train_accuracy:.3f}; "
            f"PlantVillage validation accuracy {val_accuracy:.3f}; macro F1 {val_macro_f1:.3f}"
        )
        if field_val_macro_f1 is not None:
            epoch_message += (
                f"; field validation accuracy {field_val_accuracy:.3f}; "
                f"macro F1 {field_val_macro_f1:.3f}"
            )
        print(epoch_message, flush=True)
        if selection_macro_f1 >= best_macro_f1:
            best_macro_f1 = selection_macro_f1
            best_validation_summary = history[-1]
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), model_dir / "disease_model.pth")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    pv_logits, pv_labels = collect_logits(model, pv_test_loader, device)
    field_logits, field_labels = collect_logits(model, field_test_loader, device)
    pv_summary, pv_report, pv_matrix = report_for(
        pv_logits, pv_labels, class_names, temperature
    )
    field_summary, field_report, field_matrix = report_for(
        field_logits, field_labels, class_names, temperature
    )

    index_to_class = {str(index): name for index, name in enumerate(class_names)}
    for name in ("class_names.json", "class_mapping.json"):
        (model_dir / name).write_text(
            json.dumps(index_to_class, indent=2), encoding="utf-8"
        )
    if pv_report:
        (model_dir / "classification_report.json").write_text(
            json.dumps(pv_report, indent=2), encoding="utf-8"
        )
        (model_dir / "confusion_matrix.json").write_text(
            json.dumps(pv_matrix, indent=2), encoding="utf-8"
        )
    (model_dir / "field_classification_report.json").write_text(
        json.dumps(field_report or {}, indent=2), encoding="utf-8"
    )
    (model_dir / "field_confusion_matrix.json").write_text(
        json.dumps(field_matrix or [], indent=2), encoding="utf-8"
    )
    config = {
        "architecture": "MobileNetV3-Small, fine-tuned",
        "num_classes": len(class_names),
        "image_size": [224, 224],
        "normalization": {"mean": MEAN, "std": STD},
        "class_mapping": index_to_class,
        "temperature_scale": temperature,
        "datasets": {
            "plantvillage": {
                "training_images": len(pv_train[0]),
                "test": pv_summary,
            },
            "plantdoc": {
                key: value for key, value in field_metadata.items()
                if key not in ("unmapped_classes",)
            },
        },
        "validation": best_validation_summary,
        "field_test": field_summary,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if pv_summary:
        config["test_accuracy"] = pv_summary["accuracy"]
        config["macro_f1"] = round(pv_summary["macro_f1"] / 100, 4)
    model_config_path = model_dir / "model_config.json"
    model_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([item["epoch"] for item in history],
             [item["train_accuracy"] for item in history], label="Train")
    plt.plot([item["epoch"] for item in history],
             [item["validation_accuracy"] for item in history], label="Validation")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([item["epoch"] for item in history],
             [item["train_loss"] for item in history], label="Train")
    plt.plot([item["epoch"] for item in history],
             [item["validation_loss"] for item in history], label="Validation")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_metrics.png", dpi=150)
    plt.close()

    summary = {
        "plantvillage_test": pv_summary,
        "plantdoc_field_test": field_summary,
        "temperature_scale": temperature,
        "checkpoint": str(checkpoint_path),
    }
    (project_root / "reports" / "disease_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Saved disease model, calibrated probabilities, and separate lab/field reports.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV3 with all available labeled crop-disease images."
    )
    default_data_dir = Path(__file__).resolve().parents[1] / "final_data" / "data"
    parser.add_argument("--data-dir", default=os.getenv("AGRISENSE_DATA_DIR", str(default_data_dir)))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Optional PlantVillage sample cap; 0 uses every image.",
    )
    parser.add_argument(
        "--max-field-samples-per-class",
        type=int,
        default=0,
        help="Optional PlantDoc sample cap; 0 uses every mapped field image.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--field-sample-weight",
        type=float,
        default=4.0,
        help="Relative sampling boost for PlantDoc field photos during training.",
    )
    parser.add_argument(
        "--exclude-plantvillage-val",
        action="store_true",
        help="Keep the supplied PlantVillage validation split out of training.",
    )
    args = parser.parse_args()
    train_disease_model(
        args.data_dir,
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        max_samples_per_class=max(0, args.max_samples_per_class),
        max_field_samples_per_class=max(0, args.max_field_samples_per_class),
        num_workers=args.num_workers,
        field_sample_weight=max(1.0, args.field_sample_weight),
        include_plantvillage_val=not args.exclude_plantvillage_val,
    )
