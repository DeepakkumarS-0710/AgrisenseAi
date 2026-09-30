import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# AGRISENSE AI - CROP MODEL TRAINING
# ============================================================

print("=" * 70)
print("        AGRISENSE AI - CROP MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# 2. FIND CROP DATASET FOLDER
# ============================================================

CROP_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "final_data",
    "data",
    "crop_recommendation"
)


print("\n[1/7] Checking dataset folder...")


if not os.path.exists(CROP_DATA_DIR):

    raise FileNotFoundError(
        f"""
Crop dataset folder not found:

{CROP_DATA_DIR}

Please check your project structure.
"""
    )


print("Dataset folder found:")
print(CROP_DATA_DIR)


# ============================================================
# 3. FIND CSV FILE
# ============================================================

csv_files = []

for filename in os.listdir(CROP_DATA_DIR):

    if filename.lower().endswith(".csv"):

        full_path = os.path.join(
            CROP_DATA_DIR,
            filename
        )

        csv_files.append(
            full_path
        )


if not csv_files:

    raise FileNotFoundError(
        f"""
No CSV file was found inside:

{CROP_DATA_DIR}
"""
    )


print("\nCSV files found:")

for file in csv_files:

    print(
        " -",
        os.path.basename(file)
    )


# ============================================================
# 4. SELECT DATASET
# ============================================================

# Prefer a file containing "crop" in its name.

crop_csv_files = [

    file
    for file in csv_files
    if "crop" in os.path.basename(file).lower()

]


if crop_csv_files:

    DATA_PATH = crop_csv_files[0]

else:

    DATA_PATH = csv_files[0]


print("\nUsing dataset:")

print(DATA_PATH)


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\n[2/7] Loading dataset...")


df = pd.read_csv(
    DATA_PATH
)


print("\nDataset shape:")

print(df.shape)


print("\nOriginal columns:")

print(
    list(df.columns)
)


# ============================================================
# 6. CLEAN COLUMN NAMES
# ============================================================

df.columns = [

    str(column)
    .strip()
    .lower()

    for column in df.columns

]


print("\nStandardized columns:")

print(
    list(df.columns)
)


# ============================================================
# 7. REQUIRED FEATURES
# ============================================================

FEATURE_NAMES = [

    "n",

    "p",

    "k",

    "temperature",

    "humidity",

    "ph",

    "rainfall"

]


TARGET_COLUMN = "label"


# ============================================================
# 8. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (

    FEATURE_NAMES
    +
    [TARGET_COLUMN]

)


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(
        f"""
Required columns are missing.

Missing:
{missing_columns}

Available:
{list(df.columns)}
"""
    )


# ============================================================
# 9. KEEP REQUIRED COLUMNS
# ============================================================

df = df[
    required_columns
].copy()


# ============================================================
# 10. CONVERT NUMERIC FEATURES
# ============================================================

print("\n[3/7] Cleaning dataset...")


for column in FEATURE_NAMES:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# ============================================================
# 11. CLEAN CROP LABELS
# ============================================================

df[TARGET_COLUMN] = (

    df[TARGET_COLUMN]

    .astype(str)

    .str.strip()

    .str.lower()

)


# ============================================================
# 12. REMOVE INVALID ROWS
# ============================================================

before_rows = len(df)


df = df.dropna()


after_rows = len(df)


print(
    "\nRows before cleaning :",
    before_rows
)


print(
    "Rows after cleaning  :",
    after_rows
)


print(
    "Rows removed         :",
    before_rows - after_rows
)


# ============================================================
# 13. CROP DISTRIBUTION
# ============================================================

crop_counts = (

    df[TARGET_COLUMN]

    .value_counts()

    .sort_index()

)


print("\nCrop classes:")

print("-" * 50)


for crop, count in crop_counts.items():

    print(
        f"{crop:<20} : {count}"
    )


print("-" * 50)


print(
    "Total crop classes:",
    len(crop_counts)
)


# ============================================================
# 14. PREPARE X AND Y
# ============================================================

X = df[
    FEATURE_NAMES
]


y_text = df[
    TARGET_COLUMN
]


# ============================================================
# 15. LABEL ENCODER
# ============================================================

print(
    "\n[4/7] Encoding crop labels..."
)


label_encoder = LabelEncoder()


y = label_encoder.fit_transform(
    y_text
)


print("\nCrop label mapping:")

print("-" * 50)


for index, crop in enumerate(
    label_encoder.classes_
):

    print(
        f"{index:>2} -> {crop}"
    )


print("-" * 50)


# ============================================================
# 16. TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[5/7] Splitting dataset..."
)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "Training samples :",
    len(X_train)
)


print(
    "Testing samples  :",
    len(X_test)
)


# ============================================================
# 17. CREATE RANDOM FOREST
# ============================================================

print(
    "\nTraining Random Forest..."
)


model = RandomForestClassifier(

    n_estimators=500,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1,

    max_features="sqrt"

)


# ============================================================
# 18. TRAIN
# ============================================================

model.fit(

    X_train,

    y_train

)


print(
    "Training completed."
)


# ============================================================
# 19. PREDICT TEST DATA
# ============================================================

print(
    "\n[6/7] Evaluating model..."
)


y_pred = model.predict(
    X_test
)


# ============================================================
# 20. OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n")

print("=" * 70)

print(
    f"OVERALL ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print("=" * 70)


# ============================================================
# 21. CROP-WISE PERFORMANCE
# ============================================================

print("\nCROP-WISE PERFORMANCE")

print("=" * 70)


report = classification_report(

    y_test,

    y_pred,

    target_names=label_encoder.classes_,

    digits=4,

    zero_division=0

)


print(report)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")

print("=" * 70)


cm = confusion_matrix(

    y_test,

    y_pred

)


print(cm)


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

print("\nFEATURE IMPORTANCE")

print("=" * 70)


feature_importance = pd.DataFrame({

    "feature": FEATURE_NAMES,

    "importance": model.feature_importances_

})


feature_importance = (

    feature_importance

    .sort_values(

        "importance",

        ascending=False

    )

)


for _, row in feature_importance.iterrows():

    print(

        f"{row['feature']:<15} : "
        f"{row['importance']:.4f}"

    )


# ============================================================
# 24. CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR = os.path.join(

    PROJECT_ROOT,

    "models"

)


os.makedirs(

    MODEL_DIR,

    exist_ok=True

)


# ============================================================
# 25. MODEL OUTPUT PATH
# ============================================================

MODEL_PATH = os.path.join(

    MODEL_DIR,

    "crop_model.pkl"

)


# ============================================================
# 26. SAVE MODEL
# ============================================================

print(
    "\n[7/7] Saving model..."
)


model_data = {

    "model": model,

    "label_encoder": label_encoder,

    "feature_names": FEATURE_NAMES,

    "accuracy": accuracy,

    "classes": list(
        label_encoder.classes_
    )

}


with open(

    MODEL_PATH,

    "wb"

) as f:

    pickle.dump(

        model_data,

        f

    )


print(
    "\nNew crop model saved successfully:"
)


print(
    MODEL_PATH
)


# ============================================================
# 27. SAMPLE PREDICTION FUNCTION
# ============================================================

def test_sample(

    sample_number,

    values

):

    sample_df = pd.DataFrame(

        [values],

        columns=FEATURE_NAMES

    )


    prediction = model.predict(

        sample_df

    )[0]


    probabilities = model.predict_proba(

        sample_df

    )[0]


    predicted_crop = (

        label_encoder

        .inverse_transform(

            [prediction]

        )[0]

    )


    confidence = (

        probabilities[prediction]

        * 100

    )


    print(
        f"\nSample {sample_number}"
    )


    print(
        "Input:",
        values
    )


    print(
        "Predicted crop:",
        predicted_crop
    )


    print(
        "Model confidence:",
        f"{confidence:.2f}%"
    )


    # --------------------------------------------------------
    # Top 5 predictions
    # --------------------------------------------------------

    ranked_indices = (

        probabilities

        .argsort()[::-1][:5]

    )


    print(
        "\nTop 5 predictions:"
    )


    for index in ranked_indices:

        crop_name = (

            label_encoder

            .inverse_transform(

                [int(index)]

            )[0]

        )


        probability = (

            probabilities[index]

            * 100

        )


        print(

            f"  {crop_name:<15}"
            f" -> {probability:.2f}%"

        )


# ============================================================
# 28. SAMPLE TEST 1
# ============================================================

print("\n")

print("=" * 70)

print(
    "SAMPLE PREDICTION TESTS"
)

print("=" * 70)


# Maize-like test
test_sample(

    1,

    [

        90,     # N

        42,     # P

        20,     # K

        24,     # Temperature

        65,     # Humidity

        6.2,    # pH

        90      # Rainfall

    ]

)


# High rainfall test
test_sample(

    2,

    [

        90,     # N

        42,     # P

        43,     # K

        27,     # Temperature

        75,     # Humidity

        6.5,    # pH

        2000    # Rainfall

    ]

)


# ============================================================
# 29. FINISHED
# ============================================================

print("\n")

print("=" * 70)

print(
    "CROP MODEL TRAINING COMPLETED"
)

print("=" * 70)

print(
    "\nModel file:"
)

print(
    MODEL_PATH
)

print(
    "\nTotal crop classes:",
    len(label_encoder.classes_)
)

print(
    f"Overall accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "\nNext step: test the new model with crop_service.py."
)

print("=" * 70)