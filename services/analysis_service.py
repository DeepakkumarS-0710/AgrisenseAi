
import os
import sys

# ---------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------
# Import project services
# ---------------------------------------------------------
from services.crop_service import predict_crop
from services.disease_service import predict_disease
from rag.retriever import retrieve_knowledge
from services.llm_service import get_agricultural_advice


# =========================================================
# MULTIMODAL ANALYSIS
# =========================================================

def run_multimodal_analysis(
    soil_moisture=None,
    temperature=25.0,
    humidity=70.0,
    ph=6.5,
    nitrogen=50,
    phosphorus=50,
    potassium=50,
    rainfall=100,
    image_path=None
):
    """
    Complete AgriSense multimodal analysis.

    Combines:

    1. Sensor data
    2. Crop recommendation
    3. Disease detection
    4. RAG knowledge retrieval
    5. Agricultural advisory

    Returns a dictionary that can be used by Flask/frontend.
    """

    # =====================================================
    # 1. CLEAN SENSOR DATA
    # =====================================================

    # Keep soil moisture separate from humidity.
    if (
        soil_moisture is not None
        and str(soil_moisture).strip() != ""
        and str(soil_moisture).strip().lower()
        not in ("none", "not available", "null")
    ):
        try:
            display_soil_moisture = float(soil_moisture)

        except (ValueError, TypeError):
            display_soil_moisture = "Not Available"

    else:
        display_soil_moisture = "Not Available"


    # -----------------------------------------------------
    # Safely convert numeric sensor values
    # -----------------------------------------------------

    def safe_float(value, default):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


    # -----------------------------------------------------
    # Sensor dictionary
    # -----------------------------------------------------

    sensor_dict = {

        "soil_moisture": display_soil_moisture,

        "temperature": safe_float(
            temperature,
            25.0
        ),

        "humidity": safe_float(
            humidity,
            70.0
        ),

        "ph": safe_float(
            ph,
            6.5
        ),

        "nitrogen": safe_float(
            nitrogen,
            50.0
        ),

        "phosphorus": safe_float(
            phosphorus,
            50.0
        ),

        "potassium": safe_float(
            potassium,
            50.0
        ),

        "rainfall": safe_float(
            rainfall,
            100.0
        )
    }


    # =====================================================
    # 2. CROP RECOMMENDATION
    # =====================================================

    try:

        crop_res = predict_crop(

            N=sensor_dict["nitrogen"],

            P=sensor_dict["phosphorus"],

            K=sensor_dict["potassium"],

            temperature=sensor_dict["temperature"],

            humidity=sensor_dict["humidity"],

            ph=sensor_dict["ph"],

            rainfall=sensor_dict["rainfall"]
        )


        # =================================================
        # IMPORTANT CONFIDENCE NORMALIZATION
        # =================================================
        #
        # crop_service.py returns:
        #
        # "model_confidence": 85.42
        #
        # Some frontend / LLM code may expect:
        #
        # "confidence": 85.42
        #
        # Therefore we provide BOTH keys.
        # =================================================

        model_confidence = crop_res.get(
            "model_confidence",
            0.0
        )

        try:
            model_confidence = float(
                model_confidence
            )
        except (ValueError, TypeError):
            model_confidence = 0.0


        crop_res["model_confidence"] = round(
            model_confidence,
            2
        )

        crop_res["confidence"] = round(
            model_confidence,
            2
        )


        # =================================================
        # DEBUG OUTPUT
        # =================================================

        print("\n========================================")
        print("CROP MODEL RESULT")
        print("========================================")

        print(
            "Recommended Crop :",
            crop_res.get(
                "recommended_crop",
                "Unknown"
            )
        )

        print(
            "Model Confidence :",
            str(
                crop_res.get(
                    "model_confidence",
                    0.0
                )
            ) + "%"
        )

        print(
            "Confidence       :",
            str(
                crop_res.get(
                    "confidence",
                    0.0
                )
            ) + "%"
        )

        print(
            "Crop Options     :",
            crop_res.get(
                "crop_options",
                []
            )
        )

        print("========================================\n")


    except Exception as e:

        print(
            "Crop prediction error:",
            str(e)
        )

        crop_res = {

            "recommended_crop": "Unknown",

            "model_confidence": 0.0,

            "confidence": 0.0,

            "crop_options": [],

            "error": str(e)
        }


    # =====================================================
    # 3. DISEASE DETECTION
    # =====================================================

    if image_path and os.path.exists(image_path):

        try:

            disease_res = predict_disease(
                image_path
            )

        except Exception as e:

            print(
                "Disease prediction error:",
                str(e)
            )

            disease_res = {

                "crop": crop_res.get(
                    "recommended_crop",
                    "General Crop"
                ),

                "disease": "Prediction Error",

                "health_status": "Unknown",

                "confidence": 0.0,

                "raw_class": "Error",

                "error": str(e)
            }

    else:

        disease_res = {

            "crop": crop_res.get(
                "recommended_crop",
                "General Crop"
            ),

            "disease": "No Image Uploaded",

            "health_status": "Unknown",

            "confidence": 0.0,

            "raw_class": "None"
        }


    # =====================================================
    # 4. RAG KNOWLEDGE RETRIEVAL
    # =====================================================

    recommended_crop = crop_res.get(
        "recommended_crop",
        ""
    )

    disease_name = disease_res.get(
        "disease",
        ""
    )

    soil_ph = sensor_dict["ph"]


    query = (

        f"{recommended_crop} "

        f"cultivation NPK requirements "

        f"{disease_name} disease management "

        f"soil pH {soil_ph}"
    )


    try:

        rag_sources = retrieve_knowledge(

            query,

            top_k=3
        )

    except Exception as e:

        print(
            "RAG retrieval error:",
            str(e)
        )

        rag_sources = []


    # =====================================================
    # 5. AGRICULTURAL ADVISOR
    # =====================================================

    try:

        advisor_res = get_agricultural_advice(

            sensor_dict,

            crop_res,

            disease_res,

            rag_sources
        )

    except Exception as e:

        print(
            "Advisor generation error:",
            str(e)
        )

        advisor_res = {

            "summary": (
                "Agricultural analysis completed, "
                "but advisor generation failed."
            ),

            "crop_advice": (
                f"Recommended crop: "
                f"{recommended_crop}"
            ),

            "disease_advice": (
                f"Detected condition: "
                f"{disease_name}"
            ),

            "irrigation_advice": (
                "Monitor soil moisture before irrigation."
            ),

            "soil_advice": (
                f"Current soil pH: {soil_ph}"
            ),

            "nutrient_advice": (
                "Monitor NPK values and follow "
                "soil-test-based recommendations."
            ),

            "risk_level": "Unknown",

            "risk_reason": str(e),

            "immediate_actions": [

                "Check sensor values.",

                "Monitor crop health.",

                "Try the analysis again."
            ],

            "short_term_actions": [

                "Monitor soil moisture.",

                "Monitor temperature and humidity.",

                "Check crop leaves for disease symptoms."
            ]
        }


    # =====================================================
    # 6. FINAL MULTIMODAL OUTPUT
    # =====================================================

    final_output = {

        "sensor_data": sensor_dict,

        "crop_recommendation": crop_res,

        "disease_detection": disease_res,

        "rag_sources": rag_sources,

        "advisor": advisor_res
    }


    return final_output


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Testing AgriSense Multimodal Analysis Service"
    )

    print("=" * 60)


    result = run_multimodal_analysis(

        soil_moisture=45,

        temperature=27.5,

        humidity=80.0,

        ph=6.2,

        nitrogen=90,

        phosphorus=42,

        potassium=43,

        rainfall=200,

        image_path=None
    )


    print("\nRESULT")

    print("-" * 60)


    print(
        "Sensor Data:",
        result.get(
            "sensor_data"
        )
    )


    print(
        "Crop Recommendation:",
        result.get(
            "crop_recommendation"
        )
    )


    print(
        "Disease Detection:",
        result.get(
            "disease_detection"
        )
    )


    print(
        "RAG Sources:",
        result.get(
            "rag_sources"
        )
    )


    print(
        "Advisor:",
        result.get(
            "advisor"
        )
    )


    print("-" * 60)

    print(
        "Multimodal Analysis Test Completed"
    )

