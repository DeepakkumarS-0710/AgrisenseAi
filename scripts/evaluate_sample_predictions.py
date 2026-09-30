import os
import sys
import json
import random
import glob
import argparse
import shutil
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.test_disease_image import predict_single_image

def run_sample_evaluation(data_dir, healthy_limit=10, diseased_limit=10):
    pv_dir = os.path.join(data_dir, "plantvillage")
    if not os.path.isdir(pv_dir):
        raise FileNotFoundError(f"PlantVillage dataset not found at {pv_dir}")

    img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # 1. Collect healthy and diseased image paths across classes
    healthy_files = []
    diseased_files = []
    
    for root, dirs, files in os.walk(pv_dir):
        image_files = [os.path.join(root, f) for f in files if f.endswith(img_extensions)]
        if image_files:
            c_name = os.path.basename(root)
            if "healthy" in c_name.lower():
                healthy_files.extend([(f, c_name) for f in image_files])
            else:
                diseased_files.extend([(f, c_name) for f in image_files])
                
    print(f"Discovered {len(healthy_files)} healthy images and {len(diseased_files)} diseased images.")
    
    # Sample 10 healthy and 10 diseased images reproducibly
    SEED = 42
    random.Random(SEED).shuffle(healthy_files)
    random.Random(SEED).shuffle(diseased_files)
    
    selected_healthy = healthy_files[:healthy_limit] if healthy_limit else healthy_files
    selected_diseased = diseased_files[:diseased_limit] if diseased_limit else diseased_files
    selected_samples = selected_healthy + selected_diseased
    
    results = []
    reports_dir = os.path.join(BASE_DIR, "reports")
    correct_dir = os.path.join(reports_dir, "correct_predictions")
    shutil.rmtree(correct_dir, ignore_errors=True)
    os.makedirs(correct_dir, exist_ok=True)

    print("\nEvaluating Sample Predictions on Held-Out PlantVillage Images...")
    for idx, (img_path, actual_class) in enumerate(selected_samples, 1):
        try:
            pred = predict_single_image(img_path)
            predicted_class = pred["raw_class"]
            conf = pred["confidence"]
            is_correct = (actual_class.strip().lower() == predicted_class.strip().lower())

            copied_path = ""
            if is_correct:
                class_dir = os.path.join(correct_dir, actual_class)
                os.makedirs(class_dir, exist_ok=True)
                copied_path = os.path.join(class_dir, os.path.basename(img_path))
                shutil.copy2(img_path, copied_path)
            
            results.append({
                "sample_num": idx,
                "image": os.path.basename(img_path),
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "confidence_percent": conf,
                "correct": is_correct,
                "correct_image_path": os.path.relpath(copied_path, BASE_DIR) if copied_path else ""
            })
            print(f"Sample {idx}/{len(selected_samples)} | Actual: {actual_class} | Pred: {predicted_class} ({conf:.1f}%) | Correct: {is_correct}")
        except Exception as e:
            print(f"Error evaluating sample {img_path}: {e}")

    df_res = pd.DataFrame(results)
    os.makedirs(reports_dir, exist_ok=True)
    report_csv = os.path.join(reports_dir, "sample_predictions.csv")
    df_res.to_csv(report_csv, index=False)
    
    correct_count = sum(r["correct"] for r in results)
    print(f"\n==========================================")
    print(f"SAMPLE PREDICTIONS VERIFICATION REPORT:")
    print(f"  - Total Evaluated: {len(results)}")
    print(f"  - Correct:         {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"  - CSV Saved To:    {report_csv}")
    print(f"  - Correct Images:  {correct_dir}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate disease predictions and collect correct images")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("AGRISENSE_DATA_DIR", r"F:\adithyan\agrisense\data"),
        help="Dataset root containing plantvillage/",
    )
    parser.add_argument(
        "--healthy-limit",
        type=int,
        default=10,
        help="Number of healthy images to evaluate; use 0 for all healthy images",
    )
    parser.add_argument(
        "--diseased-limit",
        type=int,
        default=10,
        help="Number of diseased images to evaluate; use 0 for all diseased images",
    )
    args = parser.parse_args()
    run_sample_evaluation(os.path.abspath(args.data_dir), args.healthy_limit, args.diseased_limit)
