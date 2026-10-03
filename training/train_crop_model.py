"""Compare, calibrate and save the crop classifier using a stratified holdout."""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42


def find_crop_csv(data_dir):
    candidates = [
        data_dir / "crop_recommendation" / "Crop_recommendation.csv",
        data_dir / "crop_recommendation" / "crop_recommendation.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    folder = data_dir / "crop_recommendation"
    if folder.is_dir():
        matches = list(folder.glob("*.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No crop recommendation CSV was found beneath {data_dir / 'crop_recommendation'}."
    )


def load_crop_data(csv_path):
    frame = pd.read_csv(csv_path)
    columns = {str(name).strip().lower(): name for name in frame.columns}
    missing = [name for name in [x.lower() for x in FEATURE_NAMES] if name not in columns]
    target = columns.get("label") or columns.get("crop")
    if missing or target is None:
        raise ValueError(
            "Crop CSV must contain N, P, K, temperature, humidity, ph, rainfall, and label columns."
        )
    features = frame[[columns[name.lower()] for name in FEATURE_NAMES]].copy()
    features.columns = FEATURE_NAMES
    features = features.apply(pd.to_numeric, errors="coerce")
    labels = frame[target].astype(str).str.strip().str.lower()
    valid = features.notna().all(axis=1) & labels.ne("")
    features, labels = features.loc[valid], labels.loc[valid]
    if features.empty:
        raise ValueError("The crop dataset contains no complete numeric rows.")
    return features, labels


def training_profiles(features, labels):
    profiles = {}
    for crop in sorted(labels.unique()):
        crop_rows = features.loc[labels == crop]
        profiles[crop] = {
            feature: {
                "p10": float(crop_rows[feature].quantile(0.10)),
                "p50": float(crop_rows[feature].median()),
                "p90": float(crop_rows[feature].quantile(0.90)),
            }
            for feature in FEATURE_NAMES
        }
    feature_ranges = {
        feature: {
            "p01": float(features[feature].quantile(0.01)),
            "p99": float(features[feature].quantile(0.99)),
        }
        for feature in FEATURE_NAMES
    }
    return profiles, feature_ranges


def candidate_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def train_crop_model(data_dir):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir).resolve()
    csv_path = find_crop_csv(data_dir)
    print(f"Loading labeled crop data from {csv_path}")
    features, labels = load_crop_data(csv_path)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    class_names = [str(name) for name in encoder.classes_]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=0.20,
        random_state=SEED,
        stratify=encoded_labels,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparisons = {}
    best_name = None
    best_model = None
    best_macro_f1 = -1.0

    for name, model in candidate_models().items():
        print(f"Comparing {name} with 5-fold macro-F1 validation...", flush=True)
        scores = cross_val_score(
            model, x_train, y_train, cv=folds, scoring="f1_macro", n_jobs=1
        )
        mean_score = float(scores.mean())
        comparisons[name] = {
            "fold_macro_f1": [round(float(value), 4) for value in scores],
            "mean_macro_f1": round(mean_score, 4),
        }
        print(f"  Mean macro-F1: {mean_score:.4f}", flush=True)
        if mean_score > best_macro_f1:
            best_name, best_model, best_macro_f1 = name, model, mean_score

    print(f"Calibrating the selected {best_name} model...", flush=True)
    calibrated_model = CalibratedClassifierCV(
        estimator=best_model,
        method="sigmoid",
        cv=5,
        ensemble=False,
        n_jobs=1,
    )
    calibrated_model.fit(x_train, y_train)
    probabilities = calibrated_model.predict_proba(x_test)
    predictions = calibrated_model.classes_[np.argmax(probabilities, axis=1)]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=calibrated_model.classes_)), 4),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "cross_validation": comparisons,
    }
    print(json.dumps(metrics, indent=2), flush=True)

    class_report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=list(range(len(class_names)))).tolist()
    profiles, feature_ranges = training_profiles(features, labels)
    scaler = StandardScaler().fit(x_train)
    model_dir = project_root / "models"
    report_dir = project_root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": calibrated_model,
        "model_name": best_name,
        "label_encoder": encoder,
        "feature_names": FEATURE_NAMES,
        "crop_profiles": profiles,
        "feature_ranges": feature_ranges,
        "source_rows": int(len(features)),
    }
    with (model_dir / "crop_model.pkl").open("wb") as handle:
        pickle.dump(model_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (model_dir / "crop_preprocessor.pkl").open("wb") as handle:
        pickle.dump({"scaler": scaler, "feature_names": FEATURE_NAMES}, handle)
    (report_dir / "crop_model_report.json").write_text(
        json.dumps({
            "model": best_name,
            "dataset": str(csv_path),
            "classes": class_names,
            "metrics": metrics,
            "note": "Holdout score is from a small public crop recommendation dataset; it does not measure farm yield or district-level Tamil Nadu suitability.",
        }, indent=2),
        encoding="utf-8",
    )
    (report_dir / "crop_classification_report.json").write_text(
        json.dumps(class_report, indent=2), encoding="utf-8"
    )
    (report_dir / "crop_confusion_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    print(f"Saved calibrated model to {model_dir / 'crop_model.pkl'}", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a calibrated crop recommendation classifier.")
    default_data_dir = Path(__file__).resolve().parents[1] / "final_data" / "data"
    parser.add_argument("--data-dir", default=os.getenv("AGRISENSE_DATA_DIR", str(default_data_dir)))
    args = parser.parse_args()
    train_crop_model(args.data_dir)
