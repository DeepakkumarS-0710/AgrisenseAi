# AGriSense

AGriSense is a Flask agriculture assistant with crop recommendations, leaf-image checks, an advisory page, analysis history, and a weekly sensor dashboard.

## Run the application

From this folder, activate the project environment and start Flask:

    ..\..\.venv\Scripts\Activate.ps1
    python app.py

Open http://127.0.0.1:5000.

The final_data directory is ignored by Git. Place training files under final_data/data, or point the training commands at another dataset directory with --data-dir.

## Model training

The crop trainer compares Random Forest, Extra Trees and Histogram Gradient Boosting using stratified validation folds, then calibrates the selected model's class scores. It also saves training ranges so the app can flag inputs outside the data it learned from.

    python training/train_crop_model.py

The disease trainer uses PlantVillage train and validation folders by default. If PlantDoc is present under final_data/data/plantdoc/PlantDoc-Dataset, it adds only unambiguous matching labels for training, samples those field photos more often, and selects checkpoints using both PlantVillage and PlantDoc validation results. The separate PlantDoc test images remain held out for a field-photo report.

    python training/train_disease_model.py --epochs 5

Optional --max-samples-per-class N and --max-field-samples-per-class N arguments support shorter experiments. A value of 0 (the default) uses all images. The disease trainer writes separate PlantVillage and PlantDoc reports to models/disease_model/ and reports/disease_training_summary.json.

The training run records separate PlantVillage controlled-image and PlantDoc field-photo reports. The current mapped PlantDoc holdout has 146 images, so its score is a small-sample warning rather than a dependable estimate of field accuracy. Read both reports before interpreting results; neither guarantees performance on a particular farm.

The retrained `models/disease_model/best_model.pth` checkpoint is included with the project. The PlantVillage and PlantDoc image folders remain local data and are not bundled.

## Dataset layout

    final_data/data/
    ├── crop_recommendation/crop_recommendation.csv
    ├── plantvillage/PlantVillage/
    │   ├── train/[class]/[image]
    │   └── val/[class]/[image] (optional)
    └── plantdoc/PlantDoc-Dataset/
        ├── train/[class]/[image]
        └── test/[class]/[image]

PlantVillage contributes 38 classes of healthy and diseased leaves photographed in controlled conditions. PlantDoc contributes field-style photos, but the trainer maps only labels that clearly match a PlantVillage class; ambiguous and unsupported classes are left out. The downloaded PlantDoc working copy is kept under the ignored final_data folder and its upstream README and license are included with that local copy. See [dataset sources and attribution](reports/dataset_sources.md).

Paddy Doctor is a useful Tamil Nadu field dataset, but the authors' public dataset page points to an IEEE DataPort record and the released 10,407-image Kaggle subset requires accepting Kaggle competition rules. AGriSense does not silently fetch that gated data. The current disease classifier has no paddy/rice disease classes, so its disease page asks for the crop and declines to guess for paddy and other unsupported crops.

## Tamil Nadu crop reference

The crop page includes paddy, maize, millets, pulses, groundnut, sesame, cotton, sugarcane, coconut, banana, chilli, tomato and other local crops. The classifier itself still recognizes only the 22 labels in its crop training CSV. Crops outside those labels are shown as regional options, not as model predictions. District, season, soil test, water supply, variety and local advice remain important.

## Weekly soil monitoring

The dashboard's Soil Monitor page summarizes sensor packets from the last 7 days, plots a selected measurement, shows the latest received values, and refreshes while the page is open. It does not fill missing sensor values with defaults.

Send readings to POST /api/sensors as JSON. Only include fields that the device actually measures:

    {
      "soil_moisture": 42.0,
      "temperature": 29.4,
      "humidity": 72.0,
      "ph": 6.4,
      "nitrogen": 48,
      "phosphorus": 31,
      "potassium": 40,
      "rainfall": 12.0
    }

GET /api/sensors/latest returns the newest packet. GET /api/sensors/history returns recent readings. The signed-in dashboard uses GET /api/sensors/week for its seven-day review. Soil-moisture screening bands depend on sensor calibration, soil type and crop stage; NPK readings are shown as trends and are not converted into fertilizer doses.

Common device field names `N`, `P`, `K`, `soilMoisture`, `air_temperature`, `air_humidity`, and `rain` are accepted as aliases for the canonical fields above.

## Other project areas

- services/crop_service.py formats crop results and flags inputs outside the training range.
- services/disease_service.py applies crop-specific class filtering and calibrated probabilities.
- services/monitoring_service.py produces weekly reading summaries and conservative actions.
- services/tamil_nadu_crop_catalog.py provides a regional crop reference separate from model labels.
- rag/ contains the local agricultural reference index.

## Model limits

The crop dataset is small and does not contain district, season, market, soil texture or yield outcomes. A calibrated classifier score is a match to labeled examples, not the probability of a successful harvest.

The disease model can only distinguish labels represented in its training classes. Even with field-style training images, image results are screening suggestions rather than a confirmed diagnosis. Confirm uncertain results with an agricultural extension officer, especially before using a treatment.
