import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import xgboost as xgb

def train_crop_model(data_dir):
    csv_path = os.path.join(data_dir, "crop_recommendation", "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Crop dataset not found at {csv_path}. "
            "Pass the dataset root with --data-dir or AGRISENSE_DATA_DIR."
        )
    print(f"Loading crop recommendation dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Target and Features
    target_col = 'label' if 'label' in df.columns else df.columns[-1]
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Preprocessing: Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train / Test split (80% train, 20% test, random_state=42, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Training Model 1: Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train) # Tree models work great on raw or scaled features
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average='weighted')
    print(f"Random Forest Accuracy:  {rf_acc:.4f}")
    print(f"Random Forest Weighted F1: {rf_f1:.4f}")
    
    print("\n--- Training Model 2: XGBoost Classifier ---")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_preds, average='weighted')
    print(f"XGBoost Accuracy:     {xgb_acc:.4f}")
    print(f"XGBoost Weighted F1:    {xgb_f1:.4f}")
    
    # Model selection
    if xgb_f1 >= rf_f1:
        best_model = xgb_model
        best_name = "XGBoost"
        best_acc = xgb_acc
        best_f1 = xgb_f1
        best_preds = xgb_preds
    else:
        best_model = rf
        best_name = "Random Forest"
        best_acc = rf_acc
        best_f1 = rf_f1
        best_preds = rf_preds
        
    print(f"\n==========================================")
    print(f"BEST MODEL SELECTED: {best_name} (Accuracy: {best_acc:.4f}, F1: {best_f1:.4f})")
    print(f"==========================================")
    print("\nClassification Report:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_test, best_preds, target_names=target_names))
    
    # Save models directory
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "crop_model.pkl")
    preprocessor_path = os.path.join("models", "crop_preprocessor.pkl")
    
    pipeline_data = {
        'model': best_model,
        'model_name': best_name,
        'label_encoder': label_encoder,
        'feature_names': feature_cols
    }
    
    preprocessor_data = {
        'scaler': scaler,
        'feature_names': feature_cols
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline_data, f)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor_data, f)
        
    print(f"Saved crop model to {model_path}")
    print(f"Saved preprocessor to {preprocessor_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the crop recommendation model")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("AGRISENSE_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")),
        help="Dataset root containing crop_recommendation/Crop_recommendation.csv",
    )
    args = parser.parse_args()
    train_crop_model(os.path.abspath(args.data_dir))
