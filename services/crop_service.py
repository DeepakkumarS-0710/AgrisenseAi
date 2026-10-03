import os
import pickle
import numpy as np


# ============================================================
# FIELD GUIDANCE
# ============================================================

def build_field_guidance(
    crop,
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    planned_crop=None
):
    """
    Creates simple field-condition guidance.

    NOTE:
    This is guidance based on the input values.
    It is separate from the ML model confidence.
    """

    crop_name = str(crop).strip().lower()

    # --------------------------------------------------------
    # Temperature condition
    # --------------------------------------------------------

    if temperature < 18:
        temperature_status = "Cool"
    elif temperature > 32:
        temperature_status = "Hot"
    else:
        temperature_status = "Moderate"

    # --------------------------------------------------------
    # Humidity condition
    # --------------------------------------------------------

    if humidity < 50:
        humidity_status = "Dry"
    elif humidity > 75:
        humidity_status = "Humid"
    else:
        humidity_status = "Balanced"

    # --------------------------------------------------------
    # pH condition
    # --------------------------------------------------------

    if ph < 5.8:
        ph_status = "Acidic"
    elif ph > 7.2:
        ph_status = "Alkaline"
    else:
        ph_status = "Near-neutral"

    # --------------------------------------------------------
    # Rainfall condition
    # --------------------------------------------------------

    if rainfall < 500:
        rainfall_status = "Low"
    elif rainfall > 1500:
        rainfall_status = "High"
    else:
        rainfall_status = "Moderate"

    # --------------------------------------------------------
    # Nutrient condition
    # --------------------------------------------------------

    # The incoming sensor's units and crop stage are not known here, so do not
    # label a nutrient "low" or "adequate" from a universal threshold.
    nitrogen_status = "Reading recorded"
    phosphorus_status = "Reading recorded"
    potassium_status = "Reading recorded"

    # --------------------------------------------------------
    # Irrigation suggestion
    # --------------------------------------------------------

    if rainfall < 100:
        rainfall_status = "Very low"
    elif rainfall > 1500:
        rainfall_status = "Very high"

    irrigation = (
        "Use crop stage, root-zone soil moisture and the local rainfall forecast "
        "to decide irrigation. Rainfall alone is not enough to set a watering schedule."
    )

    # --------------------------------------------------------
    # Moisture / disease risk
    # --------------------------------------------------------

    if (
        humidity >= 80
        or rainfall >= 1500
        or (temperature >= 28 and humidity >= 75)
    ):
        disease_risk = "Weather conditions may favour some diseases"
    else:
        disease_risk = "No broad weather warning"

    # --------------------------------------------------------
    # Crop-specific advice
    # --------------------------------------------------------

    crop_advice = {
        "rice": "Maintain adequate water availability and monitor nitrogen levels.",
        "maize": "Monitor nitrogen and potassium and maintain adequate soil moisture.",
        "blackgram": "Check the local sowing window and scout for pests during early growth.",
        "mungbean": "Check the local sowing window and maintain drainage after heavy rain.",
        "pigeonpeas": "Use a locally recommended variety and monitor crop and soil moisture.",
        "cotton": "Monitor soil moisture and maintain balanced nutrient levels.",
        "potato": "Avoid waterlogging and maintain suitable soil moisture.",
        "tomato": "Maintain balanced nutrients and monitor humidity-related disease risk.",
        "soybean": "Maintain balanced nutrients and adequate soil moisture.",
        "groundnut": "Maintain suitable soil moisture and monitor calcium/nutrient availability.",
        "sugarcane": "Maintain adequate water and potassium availability.",
        "banana": "Maintain adequate moisture and balanced potassium nutrition.",
        "grape": "Avoid excessive humidity and monitor disease development.",
        "apple": "Monitor temperature and maintain balanced soil nutrients.",
        "coconut": "Maintain adequate moisture and monitor potassium and nutrient availability.",
    }

    crop_specific_advice = crop_advice.get(
        crop_name,
        "Monitor soil nutrients, moisture, temperature and rainfall regularly."
    )

    # --------------------------------------------------------
    # Planned crop information
    # --------------------------------------------------------

    if planned_crop:
        planned_crop_message = (
            f"Planned crop: {planned_crop}. "
            f"Compare this plan with the model output and district-specific advice before planting."
        )
    else:
        planned_crop_message = "No planned crop was specified."

    return {
        "temperature": {
            "value": round(float(temperature), 2),
            "status": temperature_status
        },

        "humidity": {
            "value": round(float(humidity), 2),
            "status": humidity_status
        },

        "ph": {
            "value": round(float(ph), 2),
            "status": ph_status
        },

        "rainfall": {
            "value": round(float(rainfall), 2),
            "status": rainfall_status
        },

        "nitrogen": {
            "value": round(float(N), 2),
            "status": nitrogen_status
        },

        "phosphorus": {
            "value": round(float(P), 2),
            "status": phosphorus_status
        },

        "potassium": {
            "value": round(float(K), 2),
            "status": potassium_status
        },

        "irrigation": irrigation,

        "disease_risk": disease_risk,

        "crop_advice": crop_specific_advice,

        "planned_crop": planned_crop_message
    }


# ============================================================
# CROP SERVICE
# ============================================================

class CropService:

    def __init__(self, model_path="models/crop_model.pkl"):

        self.model_path = model_path

        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.crop_profiles = {}
        self.feature_ranges = {}
        self.model_name = "Crop classifier"
        self.source_rows = None

        self._load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):

        # ----------------------------------------------------
        # Try the given path first
        # ----------------------------------------------------

        if not os.path.exists(self.model_path):

            # Find project root
            base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )

            alternative_path = os.path.join(
                base_dir,
                self.model_path
            )

            if os.path.exists(alternative_path):

                self.model_path = alternative_path

            else:

                raise FileNotFoundError(
                    f"Crop model not found at {self.model_path}"
                )

        # ----------------------------------------------------
        # Load pickle model
        # ----------------------------------------------------

        with open(self.model_path, "rb") as f:

            data = pickle.load(f)

            self.model = data["model"]

            self.label_encoder = data["label_encoder"]

            self.feature_names = data["feature_names"]
            self.crop_profiles = data.get("crop_profiles", {})
            self.feature_ranges = data.get("feature_ranges", {})
            self.model_name = data.get("model_name", self.model_name)
            self.source_rows = data.get("source_rows")


    # ========================================================
    # PREDICT CROP
    # ========================================================

    def predict_crop(
        self,
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        planned_crop=None
    ):

        import pandas as pd

        # ----------------------------------------------------
        # Convert inputs to float
        # ----------------------------------------------------

        N = float(N)
        P = float(P)
        K = float(K)

        temperature = float(temperature)
        humidity = float(humidity)

        ph = float(ph)

        rainfall = float(rainfall)


        # ----------------------------------------------------
        # Create input dataframe
        #
        # IMPORTANT:
        # The current trained model uses exactly 7 features:
        #
        # N
        # P
        # K
        # temperature
        # humidity
        # ph
        # rainfall
        #
        # Soil moisture is NOT used here because the current
        # model was not trained with soil moisture.
        # ----------------------------------------------------

        features_df = pd.DataFrame(
            [[
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=self.feature_names
        )


        # ====================================================
        # ML PREDICTION
        # ====================================================

        probs = self.model.predict_proba(features_df)[0]
        class_ids = list(getattr(self.model, "classes_", range(len(probs))))


        # ----------------------------------------------------
        # Find highest probability
        # ----------------------------------------------------

        top_idx = int(np.argmax(probs))

        confidence = float(probs[top_idx])


        # ----------------------------------------------------
        # Convert class number to crop name
        # ----------------------------------------------------

        recommended_label = str(
            self.label_encoder.inverse_transform(
                [int(class_ids[top_idx])]
            )[0]
        )
        recommended_crop = self._display_crop_name(recommended_label)


        # ====================================================
        # TOP 5 CROPS
        # ====================================================

        ranked_indices = np.argsort(probs)[::-1][:5]


        crop_options = []

        for index in ranked_indices:

            crop_label = str(
                self.label_encoder.inverse_transform(
                    [int(class_ids[index])]
                )[0]
            )
            crop_name = self._display_crop_name(crop_label)

            crop_confidence = round(
                float(probs[index]) * 100,
                2
            )

            crop_options.append(
                {
                    "crop": crop_name,
                    "model_label": crop_label,

                    # Clear name:
                    "model_confidence": crop_confidence
                }
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        input_summary = {

            "N": N,

            "P": P,

            "K": K,

            "temperature": temperature,

            "humidity": humidity,

            "ph": ph,

            "rainfall": rainfall
        }


        # ====================================================
        # CONDITION ANALYSIS
        # ====================================================

        field_guidance = build_field_guidance(

            recommended_crop,

            N,
            P,
            K,

            temperature,

            humidity,

            ph,

            rainfall,

            planned_crop=planned_crop
        )

        ranked_probabilities = np.sort(probs)[::-1]
        margin = float(ranked_probabilities[0] - ranked_probabilities[1]) if len(ranked_probabilities) > 1 else float(ranked_probabilities[0])
        out_of_range = []
        feature_values = {
            "N": N, "P": P, "K": K,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
        }
        for name, value in feature_values.items():
            observed = self.feature_ranges.get(name)
            if observed and not observed.get("p01", value) <= value <= observed.get("p99", value):
                out_of_range.append(name)

        if out_of_range:
            recommendation_signal = "Cautious — outside the training data range"
        elif confidence >= 0.55 and margin >= 0.20:
            recommendation_signal = "Clear model preference"
        elif confidence >= 0.30 and margin >= 0.08:
            recommendation_signal = "Moderate model preference"
        else:
            recommendation_signal = "Close call — compare the alternatives"

        profile_matches = []
        profile = self.crop_profiles.get(recommended_label, {})
        for name, value in feature_values.items():
            band = profile.get(name)
            if band and band.get("p10", value) <= value <= band.get("p90", value):
                profile_matches.append(name)

        evidence = (
            "This class was the closest match to the seven input values in the model's labeled crop examples."
        )
        if profile_matches:
            evidence += " Inputs within the observed range for this class: " + ", ".join(profile_matches) + "."
        evidence += " Similarity to training examples is not a yield or planting-success forecast."


        # ====================================================
        # CLEAR PREDICTION RESULT
        # ====================================================

        return {

            # -----------------------------------------------
            # Main recommendation
            # -----------------------------------------------

            "recommended_crop": recommended_crop,
            "recommended_crop_label": recommended_label,


            # -----------------------------------------------
            # Planned crop
            # -----------------------------------------------

            "planned_crop": planned_crop,


            # -----------------------------------------------
            # IMPORTANT:
            # This is ML probability, NOT scientific
            # agricultural suitability percentage.
            # -----------------------------------------------

            "model_confidence": round(
                confidence * 100,
                2
            ),
            "recommendation_signal": recommendation_signal,
            "input_outside_training_range": out_of_range,
            "recommendation_evidence": evidence,
            "model_name": self.model_name,
            "model_class_count": len(self.label_encoder.classes_),
            "training_examples": self.source_rows,


            # -----------------------------------------------
            # Top 5 predictions
            # -----------------------------------------------

            "crop_options": crop_options,


            # -----------------------------------------------
            # Input values
            # -----------------------------------------------

            "input_summary": input_summary,


            # -----------------------------------------------
            # Agricultural condition analysis
            # -----------------------------------------------

            "field_guidance": field_guidance
        }

    @staticmethod
    def _display_crop_name(label):
        known_names = {
            "blackgram": "Black gram",
            "mungbean": "Green gram",
            "pigeonpeas": "Pigeon pea",
            "kidneybeans": "Kidney beans",
            "mothbeans": "Moth beans",
            "grapes": "Grapes",
            "muskmelon": "Muskmelon",
            "rice": "Paddy (rice)",
        }
        return known_names.get(label.lower(), label.replace("_", " ").title())


# ============================================================
# GLOBAL SINGLETON
# ============================================================

_crop_service_instance = None


def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    planned_crop=None
):

    global _crop_service_instance


    if _crop_service_instance is None:

        _crop_service_instance = CropService()


    return _crop_service_instance.predict_crop(

        N=N,

        P=P,

        K=K,

        temperature=temperature,

        humidity=humidity,

        ph=ph,

        rainfall=rainfall,

        planned_crop=planned_crop
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = predict_crop(

        N=90,

        P=42,

        K=43,

        temperature=27,

        humidity=75,

        ph=6.5,

        rainfall=2000
    )


    print("\n========================================")
    print("        AGRISENSE AI PREDICTION")
    print("========================================")


    print(
        "\nRecommended Crop :",
        result["recommended_crop"]
    )


    print(
        "Model Confidence :",
        str(result["model_confidence"]) + "%"
    )


    print("\nTop 5 Predictions")
    print("----------------------------------------")

    for option in result["crop_options"]:

        print(
            option["crop"],
            "->",
            str(option["model_confidence"]) + "%"
        )


    print("\nCondition Analysis")
    print("----------------------------------------")

    guidance = result["field_guidance"]


    print(
        "Temperature :",
        guidance["temperature"]["value"],
        "°C -",
        guidance["temperature"]["status"]
    )


    print(
        "Humidity    :",
        guidance["humidity"]["value"],
        "% -",
        guidance["humidity"]["status"]
    )


    print(
        "pH          :",
        guidance["ph"]["value"],
        "-",
        guidance["ph"]["status"]
    )


    print(
        "Rainfall    :",
        guidance["rainfall"]["value"],
        "mm -",
        guidance["rainfall"]["status"]
    )


    print(
        "Nitrogen    :",
        guidance["nitrogen"]["value"],
        "-",
        guidance["nitrogen"]["status"]
    )


    print(
        "Phosphorus  :",
        guidance["phosphorus"]["value"],
        "-",
        guidance["phosphorus"]["status"]
    )


    print(
        "Potassium   :",
        guidance["potassium"]["value"],
        "-",
        guidance["potassium"]["status"]
    )


    print(
        "\nIrrigation :",
        guidance["irrigation"]
    )


    print(
        "Disease Risk :",
        guidance["disease_risk"]
    )


    print(
        "Advice :",
        guidance["crop_advice"]
    )


    print("\n========================================")
