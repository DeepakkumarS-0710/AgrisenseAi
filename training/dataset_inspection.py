import os
import glob
import pandas as pd
from PIL import Image

def inspect_csv(csv_path):
    print("==================================================")
    print("STEP 1: CROP RECOMMENDATION CSV INSPECTION")
    print("==================================================")
    print(f"Filename: {os.path.basename(csv_path)}")
    
    df = pd.read_csv(csv_path)
    rows, cols = df.shape
    print(f"Rows: {rows}")
    print(f"Columns: {cols}")
    print(f"Column Names: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    
    missing = df.isnull().sum().sum()
    print(f"\nMissing Values: {missing}")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    unique_labels = df[label_col].nunique()
    print(f"\nUnique Crop Labels ({unique_labels}):")
    print(df[label_col].unique().tolist())
    
    print("\nSamples per Crop:")
    print(df[label_col].value_counts())
    
    print("\nNumerical Statistics:")
    print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
    print("==================================================\n")
    return df

def inspect_plantvillage(pv_dir):
    print("==================================================")
    print("STEP 1: PLANTVILLAGE DATASET INSPECTION")
    print("==================================================")
    print(f"Directory: {pv_dir}")
    
    img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    all_image_paths = []
    class_counts = {}
    formats = set()
    corrupted_count = 0
    
    # Traverse directories to count classes and images
    for root, dirs, files in os.walk(pv_dir):
        image_files = [f for f in files if f.endswith(img_extensions)]
        if image_files:
            class_name = os.path.basename(root)
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += len(image_files)
            for f in image_files:
                full_path = os.path.join(root, f)
                all_image_paths.append(full_path)
    
    total_images = len(all_image_paths)
    print(f"Total Images Found: {total_images}")
    print(f"Number of Classes: {len(class_counts)}")
    print("\nClass Names & Counts:")
    for c_name, count in sorted(class_counts.items()):
        print(f"  - {c_name}: {count} images")
        
    print("\nChecking image formats and integrity (sample check if large)...")
    # Sample check up to 5000 images for speed if dataset is huge, or all
    check_sample = all_image_paths[:5000] if len(all_image_paths) > 5000 else all_image_paths
    for p in check_sample:
        ext = os.path.splitext(p)[1].lower()
        formats.add(ext)
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as e:
            corrupted_count += 1
            print(f"Corrupted image found: {p} ({e})")
            
    print(f"\nImage Formats Detected: {list(formats)}")
    print(f"Corrupted Images (Sample of {len(check_sample)}): {corrupted_count}")
    print("==================================================\n")

if __name__ == "__main__":
    data_root = r"F:\adithyan\agrisense\data"
    csv_file = os.path.join(data_root, "crop_recommendation", "Crop_recommendation.csv")
    pv_dir = os.path.join(data_root, "plantvillage")
    
    if os.path.exists(csv_file):
        inspect_csv(csv_file)
    else:
        print(f"CSV file not found at: {csv_file}")
        
    if os.path.exists(pv_dir):
        inspect_plantvillage(pv_dir)
    else:
        print(f"PlantVillage directory not found at: {pv_dir}")
