import os
import json
import urllib.request
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# LLM SERVICE
# =========================================================

class LLMService:

    def __init__(self):

        self.api_key = os.getenv(
            "LLM_API_KEY",
            ""
        ).strip()

        self.model = os.getenv(
            "LLM_MODEL",
            "gemini-1.5-flash"
        ).strip()

        self.base_url = os.getenv(
            "LLM_BASE_URL",
            ""
        ).strip()


    # =====================================================
    # GENERATE AGRICULTURAL RECOMMENDATION
    # =====================================================

    def generate_recommendation(
        self,
        sensor_data,
        crop_pred,
        disease_pred,
        rag_context
    ):

        """
        Generate agricultural recommendations using:

        1. Sensor data
        2. Crop prediction
        3. Disease prediction
        4. RAG knowledge
        5. LLM

        If the LLM fails or returns incomplete JSON,
        the deterministic expert fallback engine is used.
        """

        # -------------------------------------------------
        # Extract crop information
        # -------------------------------------------------

        crop_name = crop_pred.get(
            "recommended_crop",
            "Unknown Crop"
        )

        crop_conf = crop_pred.get(
            "confidence",
            0
        )


        # -------------------------------------------------
        # Extract disease information
        # -------------------------------------------------

        disease_name = disease_pred.get(
            "disease",
            "No image diagnosis"
        )

        disease_conf = disease_pred.get(
            "confidence",
            0
        )

        health_status = disease_pred.get(
            "health_status",
            "Unknown"
        )

        disease_reliability = disease_pred.get(
            "reliability",
            "Unknown"
        )


        # -------------------------------------------------
        # Prepare RAG context
        # -------------------------------------------------

        rag_text = "\n".join(
            [
                f"Source ({c.get('source_doc', 'Unknown')}): "
                f"{c.get('text', '')}"
                for c in rag_context
            ]
        )


        # =================================================
        # LLM GENERATION
        # =================================================

        if self.api_key:

            try:

                prompt = f"""
You are an expert AI Agricultural Advisor for the AGriSense smart farming system.

Analyze the following crop data, disease prediction, sensor readings,
and retrieved agricultural knowledge.

Provide practical agricultural recommendations.

SENSOR READINGS:
{json.dumps(sensor_data, indent=2)}

CROP RECOMMENDATION MODEL:
- Recommended Crop: {crop_name}
- Confidence: {crop_conf}%

PLANT DISEASE DETECTION MODEL:
- Health Status: {health_status}
- Detected Disease: {disease_name}
- Confidence: {disease_conf}%
- Reliability: {disease_reliability}

RETRIEVED RAG KNOWLEDGE:
{rag_text}

IMPORTANT CONSTRAINTS:

1. Do NOT invent or alter sensor readings.
2. Do NOT claim a disease was detected if the disease model predicted Healthy.
3. If disease reliability is not High, do NOT treat the predicted disease as confirmed or recommend disease-specific treatment. Ask the farmer to photograph more leaves and get local confirmation before treatment.
4. Describe the crop output as the closest match to the model's training examples, not as an optimal crop or a harvest-success probability.
5. NPK sensor units and calibration are not provided. Do NOT label nutrients low or high and do NOT suggest fertilizer doses from those values; recommend a soil test.
6. Base recommendations on the provided sensor data, predictions,
   and retrieved agricultural knowledge.
7. Do not invent information that contradicts the retrieved knowledge.
8. Return ONLY valid JSON.
9. Every required field must be present.
10. risk_level must be exactly one of:
   Low, Medium, High, Critical.

Return JSON with exactly these keys:

{{
    "overall_assessment": "...",
    "disease_explanation": "...",
    "possible_causes": "...",
    "treatment_recommendation": "...",
    "fertilizer_recommendation": "...",
    "irrigation_recommendation": "...",
    "soil_management": "...",
    "prevention": "...",
    "risk_level": "Low",
    "monitoring_advice": "..."
}}
"""


                # -------------------------------------------------
                # Build API URL
                # -------------------------------------------------

                if "googleapis" in self.base_url:

                    url = (
                        f"{self.base_url}"
                        f"/models/{self.model}"
                        f":generateContent"
                        f"?key={self.api_key}"
                    )

                else:

                    url = self.base_url


                # -------------------------------------------------
                # API payload
                # -------------------------------------------------

                payload = {

                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],

                    "generationConfig": {

                        "temperature": 0.2,

                        "response_mime_type":
                            "application/json"
                    }
                }


                # -------------------------------------------------
                # HTTP request
                # -------------------------------------------------

                req = urllib.request.Request(

                    url,

                    data=json.dumps(
                        payload
                    ).encode("utf-8"),

                    headers={
                        "Content-Type":
                            "application/json"
                    },

                    method="POST"
                )


                # -------------------------------------------------
                # Call LLM
                # -------------------------------------------------

                with urllib.request.urlopen(
                    req,
                    timeout=12
                ) as response:

                    response_text = (
                        response
                        .read()
                        .decode("utf-8")
                    )

                    res_body = json.loads(
                        response_text
                    )


                # -------------------------------------------------
                # Extract generated text
                # -------------------------------------------------

                text_out = (
                    res_body
                    ["candidates"]
                    [0]
                    ["content"]
                    ["parts"]
                    [0]
                    ["text"]
                )


                # -------------------------------------------------
                # Convert generated JSON
                # -------------------------------------------------

                result = json.loads(
                    text_out
                )


                # =================================================
                # VALIDATE REQUIRED FIELDS
                # =================================================

                required_keys = [

                    "overall_assessment",

                    "disease_explanation",

                    "possible_causes",

                    "treatment_recommendation",

                    "fertilizer_recommendation",

                    "irrigation_recommendation",

                    "soil_management",

                    "prevention",

                    "risk_level",

                    "monitoring_advice"
                ]


                missing_keys = [

                    key

                    for key in required_keys

                    if key not in result
                ]


                # -------------------------------------------------
                # Reject incomplete LLM response
                # -------------------------------------------------

                if missing_keys:

                    raise ValueError(
                        "LLM response missing required "
                        f"keys: {missing_keys}"
                    )


                # =================================================
                # VALIDATE EMPTY VALUES
                # =================================================

                empty_keys = [

                    key

                    for key in required_keys

                    if result.get(key) is None
                    or str(result.get(key)).strip() == ""
                ]


                if empty_keys:

                    raise ValueError(
                        "LLM response contains empty "
                        f"fields: {empty_keys}"
                    )


                # =================================================
                # VALIDATE RISK LEVEL
                # =================================================

                valid_risk_levels = {

                    "Low",
                    "Medium",
                    "High",
                    "Critical"
                }


                risk_level = result.get(
                    "risk_level"
                )


                if risk_level not in valid_risk_levels:

                    raise ValueError(
                        "Invalid risk level returned "
                        f"by LLM: {risk_level}"
                    )


                # =================================================
                # SUCCESS
                # =================================================

                print(
                    "LLM agricultural advisory "
                    "generated successfully."
                )


                return self._enforce_disease_confirmation(
                    result,
                    disease_pred,
                )


            # =====================================================
            # LLM FAILURE
            # =====================================================

            except Exception as e:

                print(
                    f"LLM API Call failed ({e}). "
                    "Falling back to Knowledge-Based "
                    "Expert Engine."
                )


        else:

            print(
                "LLM_API_KEY not configured. "
                "Using Knowledge-Based Expert Engine."
            )


        # =========================================================
        # FALLBACK ENGINE
        # =========================================================

        result = self._generate_expert_fallback(

            sensor_data,

            crop_name,

            crop_conf,

            disease_name,

            disease_conf,

            health_status,

            disease_reliability,

            rag_context
        )
        return self._enforce_disease_confirmation(result, disease_pred)


    @staticmethod
    def _enforce_disease_confirmation(result, disease_pred):
        """Keep uncertain image matches from producing disease treatment advice."""
        health_status = str(disease_pred.get("health_status", "")).lower()
        reliability = str(disease_pred.get("reliability", "")).lower()
        has_image_classification = health_status in {"healthy", "diseased"}
        if not has_image_classification or reliability == "high":
            return result

        confirmation = (
            "The image result is not high reliability, so it is not a confirmed diagnosis. "
            "Photograph several affected and healthy leaves, then ask a local KVK or "
            "agricultural extension officer to confirm symptoms before treatment."
        )
        existing_assessment = str(result.get("overall_assessment", "")).strip()
        result["overall_assessment"] = (
            existing_assessment + " " + confirmation
            if existing_assessment else confirmation
        )
        result["disease_explanation"] = confirmation
        result["possible_causes"] = (
            "No cause was assigned because the image match needs confirmation."
        )
        result["treatment_recommendation"] = (
            "Do not apply disease-specific pesticides or remove plants based only on this image. "
            "Confirm the symptoms locally first."
        )
        result["prevention"] = (
            "Continue routine scouting, keep tools clean, and avoid moving between wet plants."
        )
        result["monitoring_advice"] = (
            "Check nearby plants over the next few days and record whether symptoms are spreading."
        )
        return result


    # =========================================================
    # DETERMINISTIC EXPERT FALLBACK ENGINE
    # =========================================================

    def _generate_expert_fallback(
        self,
        sensors,
        crop,
        crop_conf,
        disease,
        disease_conf,
        health_status,
        disease_reliability,
        rag_context
    ):

        # -------------------------------------------------
        # Sensor values
        # -------------------------------------------------

        sm = sensors.get(
            "soil_moisture",
            "Not Available"
        )

        temp = sensors.get(
            "temperature",
            25.0
        )

        hum = sensors.get(
            "humidity",
            70.0
        )

        ph = sensors.get(
            "ph",
            6.5
        )

        n = sensors.get(
            "nitrogen",
            50
        )

        p = sensors.get(
            "phosphorus",
            50
        )

        k = sensors.get(
            "potassium",
            50
        )

        rain = sensors.get(
            "rainfall",
            100
        )


        # =================================================
        # DISEASE STATUS
        # =================================================

        is_diseased = (

            str(health_status).lower()
            == "diseased"

            and str(disease_reliability).lower()
            == "high"

            and

            "healthy"
            not in str(disease).lower()
        )


        # =================================================
        # RISK ASSESSMENT
        # =================================================

        is_healthy = (
            str(health_status).lower() == "healthy"
            and str(disease_reliability).lower() == "high"
        )

        if is_diseased:

            disease_lower = str(
                disease
            ).lower()

            if (
                "blight" in disease_lower
                or "virus" in disease_lower
                or "greening" in disease_lower
            ):

                risk_level = "High"

            else:

                risk_level = "Medium"


        elif (
            ph < 5.5
            or ph > 7.8
            or (
                sm != "Not Available"
                and float(sm) < 25
            )
        ):

            risk_level = "Medium"


        else:

            risk_level = "Low"


        # =================================================
        # OVERALL ASSESSMENT
        # =================================================

        assessment = (

            f"The crop recommendation model's closest match is "
            f"'{crop}' ({crop_conf}% model score) for the submitted NPK "
            f"({n}:{p}:{k}), temperature {temp}°C, "
            f"humidity {hum}%, soil pH {ph}, and "
            f"rainfall {rain}mm. This score is a match to labeled examples, "
            "not a forecast of planting success."
        )


        if is_diseased:

            assessment += (

                f" Plant leaf analysis indicates "
                f"{str(health_status).upper()} condition: "
                f"'{disease}' detected with "
                f"{disease_conf}% confidence."
            )

        elif is_healthy:

            assessment += (

                f" Plant leaf analysis indicates "
                f"HEALTHY crop condition with no major "
                f"active foliar pathogen detected "
                f"({disease_conf}% confidence)."
            )

        else:
            assessment += (
                " The leaf-image result is not reliable enough to confirm a disease "
                "or a healthy condition; seek local confirmation before treatment."
            )


        # =================================================
        # DISEASE EXPLANATION & CAUSES
        # =================================================

        if is_diseased:

            disease_lower = str(
                disease
            ).lower()


            dis_exp = (
                f"'{disease}' affects crop vitality "
                "and photosynthetic capacity. "
            )


            if "scab" in disease_lower:

                dis_exp += (
                    "Causes velvety olive-brown "
                    "lesions on leaves and fruit."
                )

                causes = (
                    "Overwintering fungal spores in "
                    "fallen leaves triggered by cool "
                    "temperatures and persistent moisture."
                )

                treatment = (
                    "Apply copper or sulfur-based "
                    "preventative fungicide spray. "
                    "Prune dense branches to allow "
                    "sunlight and airflow."
                )

                prevention = (
                    "Rake and destroy fallen leaves. "
                    "Avoid overhead leaf wetting."
                )


            elif "blight" in disease_lower:

                dis_exp += (
                    "Characterized by dark concentric "
                    "leaf lesions or water-soaked "
                    "necrotic patches."
                )

                causes = (
                    "Fungal or oomycete spores spreading "
                    "rapidly during warm wet conditions "
                    "with alternating high humidity."
                )

                treatment = (
                    "Apply Chlorothalonil, Mancozeb, "
                    "or copper hydroxide spray according "
                    "to the product label. Remove "
                    "severely damaged lower leaves."
                )

                prevention = (
                    "Use drip irrigation at ground level. "
                    "Rotate crops every 2-3 years with "
                    "non-susceptible plant families."
                )


            elif "rust" in disease_lower:

                dis_exp += (
                    "Causes reddish-brown powdery "
                    "pustules on leaf undersides."
                )

                causes = (
                    "Windborne fungal uredispores "
                    "thriving in warm humid air."
                )

                treatment = (
                    "Apply an approved sulfur or "
                    "locally recommended systemic "
                    "fungicide according to the label."
                )

                prevention = (
                    "Plant resistant hybrids and "
                    "clear weed host plants around borders."
                )


            elif (
                "virus" in disease_lower
                or "curl" in disease_lower
            ):

                dis_exp += (
                    "Causes leaf curling, stunting, "
                    "and severe yellowing."
                )

                causes = (
                    "Transmitted by sap-sucking insect "
                    "vectors such as whiteflies or aphids."
                )

                treatment = (
                    "Suppress insect vectors using "
                    "locally approved control measures "
                    "according to product labels. "
                    "Remove confirmed infected plants "
                    "where appropriate."
                )

                prevention = (
                    "Install insect netting over seedlings "
                    "and use yellow sticky traps where "
                    "appropriate."
                )


            else:

                dis_exp += (
                    "Foliar pathogen activity observed "
                    "on leaf tissues."
                )

                causes = (
                    "High relative humidity combined "
                    "with temperature fluctuations "
                    "and susceptible host tissue."
                )

                treatment = (
                    "Use only a locally approved treatment "
                    "after confirming the diagnosis with "
                    "an agricultural extension specialist."
                )

                prevention = (
                    "Maintain field sanitation and "
                    "balanced plant nutrition."
                )


        elif is_healthy:

            dis_exp = (
                "No active plant disease detected. "
                "Leaves display healthy green pigmentation "
                "and no major active foliar pathogen "
                "pattern was identified."
            )

            causes = (
                "Favorable environmental conditions "
                "and effective crop hygiene."
            )

            treatment = (
                "No chemical disease treatment required. "
                "Continue the current maintenance schedule."
            )

            prevention = (
                "Inspect foliage weekly, ensure adequate "
                "plant spacing, and maintain balanced "
                "soil nutrients."
            )

        else:
            dis_exp = (
                "The leaf-image match is uncertain, so no disease or healthy status "
                "is confirmed."
            )
            causes = "No cause was assigned because the image result needs confirmation."
            treatment = (
                "Do not apply disease-specific pesticides or remove plants based only "
                "on this image. Ask a local agricultural officer to confirm symptoms."
            )
            prevention = (
                "Continue routine scouting and keep tools clean after handling affected plants."
            )


        # =================================================
        # SOIL MANAGEMENT
        # =================================================

        if ph < 5.5:

            soil_adv = (

                f"Soil pH is acidic ({ph}). "
                "Apply agricultural lime according "
                "to soil-test recommendations to "
                "gradually raise pH toward the crop's "
                "appropriate range."
            )


        elif ph > 7.5:

            soil_adv = (

                f"Soil pH is alkaline ({ph}). "
                "Use soil-test-based amendments and "
                "organic matter management to improve "
                "nutrient availability."
            )


        else:

            soil_adv = (

                f"Soil pH is within the target range "
                f"for many crops ({ph}). Organic compost "
                "addition can help maintain soil health."
            )


        # =================================================
        # FERTILIZER ADVICE
        # =================================================

        fert_adv = (
            f"The sensor reported N:P:K values of {n}:{p}:{k}. "
            "Sensor units and calibration are unknown, so these readings cannot show "
            "whether nutrients are deficient or excessive. Use a soil test and "
            "crop- and district-specific guidance before choosing fertilizer or a dose."
        )


        # =================================================
        # IRRIGATION ADVICE
        # =================================================

        if sm != "Not Available":

            try:

                sm_val = float(sm)


                if sm_val < 30:

                    irrig_adv = (

                        f"Soil moisture is low ({sm_val}%). "
                        "Provide irrigation appropriate for "
                        "the crop and soil conditions."
                    )


                elif sm_val > 75:

                    irrig_adv = (

                        f"Soil moisture is high ({sm_val}%). "
                        "Pause or reduce irrigation as "
                        "appropriate and monitor drainage."
                    )


                else:

                    irrig_adv = (

                        f"Soil moisture is within a "
                        f"moderate range ({sm_val}%). "
                        "Continue monitoring and adjust "
                        "irrigation according to crop demand."
                    )


            except (
                ValueError,
                TypeError
            ):

                irrig_adv = (
                    "Soil moisture reading is unavailable. "
                    "Monitor soil moisture before irrigation."
                )


        else:

            irrig_adv = (

                f"Soil moisture sensor reading is "
                f"not available. Based on ambient humidity "
                f"({hum}%) and rainfall ({rain}mm), "
                "monitor soil moisture and irrigate "
                "according to crop demand."
            )


        # =================================================
        # MONITORING
        # =================================================

        monitoring = (

            "Check soil moisture regularly, inspect "
            "leaf surfaces and undersides for disease "
            "symptoms or insect activity, and log sensor "
            "readings in AGriSense."
        )


        # =================================================
        # FINAL FALLBACK RESULT
        # =================================================

        return {

            "overall_assessment":
                assessment,

            "disease_explanation":
                dis_exp,

            "possible_causes":
                causes,

            "treatment_recommendation":
                treatment,

            "fertilizer_recommendation":
                fert_adv,

            "irrigation_recommendation":
                irrig_adv,

            "soil_management":
                soil_adv,

            "prevention":
                prevention,

            "risk_level":
                risk_level,

            "monitoring_advice":
                monitoring
        }


# =========================================================
# DISEASE MANAGEMENT
# =========================================================

def build_disease_management(
    disease_pred,
    rag_context
):

    """
    Build actionable disease guidance using
    retrieved agricultural knowledge.
    """

    crop = disease_pred.get(
        "crop",
        "the crop"
    )

    disease = disease_pred.get(
        "disease",
        "Unknown disease"
    )

    health_status = disease_pred.get(
        "health_status",
        "Unknown"
    )

    confidence = disease_pred.get(
        "confidence",
        0
    )

    disease_key = disease.lower()

    context_text = " ".join(
        item.get("text", "")
        for item in rag_context
    ).lower()


    # =================================================
    # HEALTHY PLANT
    # =================================================

    if (
        health_status.lower() == "healthy"
        or "healthy" in disease_key
    ):

        return {

            "title":
                "No active disease pattern detected",

            "status":
                "Preventive care",

            "summary":
                (
                    f"The image model classified this "
                    f"{crop} leaf as healthy "
                    f"({confidence}% confidence). "
                    "No disease treatment is recommended "
                    "from this image."
                ),

            "treatment": [

                "Do not apply fungicide or insecticide "
                "solely from this result.",

                "Continue normal crop care and remove "
                "only clearly damaged or infected material."
            ],

            "prevention": [

                "Inspect both sides of leaves weekly.",

                "Keep plant spacing and irrigation "
                "balanced so the canopy dries quickly."
            ],

            "monitoring":
                (
                    "Recheck the crop if yellowing, "
                    "concentric spots, powdery growth, "
                    "curling, or rapid defoliation appears."
                ),

            "grounding":
                (
                    "The management guidance is "
                    "preventive because the image "
                    "was classified as healthy."
                ),

            "sources":
                rag_context[:1]
        }


    # =================================================
    # DISEASE-SPECIFIC MANAGEMENT
    # =================================================

    if "scab" in disease_key:

        treatment = [

            "Use a locally approved copper- or "
            "sulfur-based preventative fungicide "
            "according to its label.",

            "Prune crowded branches to improve "
            "sunlight and canopy airflow."
        ]

        prevention = [

            "Rake and destroy fallen leaves "
            "and fruit debris.",

            "Avoid prolonged leaf wetness and "
            "monitor new growth during cool, "
            "humid weather."
        ]


    elif "early blight" in disease_key:

        treatment = [

            "Remove severely affected lower leaves "
            "and dispose of them away from the field.",

            "Use a locally approved fungicide product "
            "only according to the product label."
        ]

        prevention = [

            "Use drip irrigation at soil level "
            "instead of overhead watering.",

            "Rotate away from susceptible crop "
            "families where practical."
        ]


    elif "late blight" in disease_key:

        treatment = [

            "Isolate and remove heavily infected "
            "foliage promptly.",

            "Seek local agricultural extension guidance "
            "for an approved treatment and follow "
            "the product label."
        ]

        prevention = [

            "Use certified disease-free planting "
            "material where available.",

            "Avoid working through wet foliage and "
            "improve drainage during wet periods."
        ]


    elif (
        "yellow leaf curl" in disease_key
        or "virus" in disease_key
    ):

        treatment = [

            "Remove confirmed infected plants "
            "where recommended to reduce disease spread.",

            "Manage insect vectors using locally "
            "approved products and label directions."
        ]

        prevention = [

            "Protect young nursery plants "
            "with suitable insect netting.",

            "Remove volunteer plants and weeds "
            "that can shelter insect vectors."
        ]


    elif (
        "greening" in disease_key
        or "huanglongbing" in disease_key
    ):

        treatment = [

            "Remove and safely dispose of confirmed "
            "infected trees according to local "
            "plant-health rules.",

            "Manage psyllid vectors with locally "
            "approved products and label directions."
        ]

        prevention = [

            "Inspect new flush for psyllids "
            "and blotchy mottling.",

            "Use balanced micronutrient management "
            "based on local agronomic advice or testing."
        ]


    elif "powdery mildew" in disease_key:

        treatment = [

            "Apply an approved treatment according "
            "to the product label.",

            "Remove heavily affected leaves and "
            "shoots where appropriate."
        ]

        prevention = [

            "Thin dense growth to improve ventilation.",

            "Avoid excessive nitrogen that can "
            "produce a crowded canopy."
        ]


    elif "bacterial spot" in disease_key:

        treatment = [

            "Remove badly affected leaves and fruit "
            "and avoid handling plants while wet.",

            "Use an approved copper-based program "
            "according to local label directions."
        ]

        prevention = [

            "Use clean or appropriately treated seed "
            "where recommended.",

            "Reduce splash dispersal with drip irrigation, "
            "mulch, and good field sanitation."
        ]


    else:

        treatment = [

            "Remove severely affected leaves and "
            "dispose of them outside the growing area.",

            "Use only a locally approved treatment "
            "after confirming the diagnosis with "
            "an agricultural extension specialist."
        ]

        prevention = [

            "Keep foliage dry, improve airflow, "
            "and avoid moving through wet plants.",

            "Sanitize tools and monitor nearby "
            "plants for the same symptoms."
        ]


    # =================================================
    # FIND RAG GROUNDING
    # =================================================

    matched_grounding = []


    for item in rag_context:

        text = item.get(
            "text",
            ""
        )

        search_terms = (
            disease_key.split()
            + [
                "treatment",
                "prevention"
            ]
        )


        if any(
            term in text.lower()
            for term in search_terms
        ):

            matched_grounding.append(
                item
            )


    if not matched_grounding:

        matched_grounding = (
            rag_context[:2]
        )


    # =================================================
    # GROUNDING MESSAGE
    # =================================================

    grounding = (

        f"This plan is grounded in the retrieved "
        f"disease-management guide for {disease}. "
        "Chemical names are references from the guide; "
        "always follow the local product label, "
        "pre-harvest interval, and protective-equipment "
        "requirements."
    )


    if (
        context_text
        and matched_grounding
    ):

        grounding += (

            " Retrieved source: "
            + matched_grounding[0].get(
                "source_doc",
                "agricultural guide"
            )
            + "."
        )


    # =================================================
    # RETURN MANAGEMENT
    # =================================================

    return {

        "title":
            f"{disease} management plan",

        "status":
            "Action recommended",

        "summary":
            (
                f"The image model detected "
                f"{disease} on {crop} with "
                f"{confidence}% confidence. "
                "Start sanitation and canopy-management "
                "steps now, then confirm the diagnosis "
                "locally before chemical treatment."
            ),

        "treatment":
            treatment,

        "prevention":
            prevention,

        "monitoring":
            (
                "Check new leaves and the underside "
                "of foliage every 2-3 days. Photograph "
                "changes and escalate if symptoms spread "
                "rapidly, affect fruit, or appear across "
                "multiple plants."
            ),

        "grounding":
            grounding,

        "sources":
            matched_grounding[:1]
    }


# =========================================================
# GLOBAL LLM INSTANCE
# =========================================================

_llm_service_instance = None


# =========================================================
# PUBLIC ADVISOR FUNCTION
# =========================================================

def get_agricultural_advice(
    sensor_data,
    crop_pred,
    disease_pred,
    rag_context
):

    global _llm_service_instance


    if _llm_service_instance is None:

        _llm_service_instance = (
            LLMService()
        )


    return (
        _llm_service_instance
        .generate_recommendation(
            sensor_data,
            crop_pred,
            disease_pred,
            rag_context
        )
    )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Testing LLM Agricultural Advisor Service..."
    )

    print("=" * 60)


    s_data = {

        "soil_moisture": 42,

        "temperature": 28.5,

        "humidity": 70,

        "ph": 6.5,

        "nitrogen": 50,

        "phosphorus": 40,

        "potassium": 40,

        "rainfall": 100
    }


    c_p = {

        "recommended_crop":
            "rice",

        "confidence":
            98.5
    }


    d_p = {

        "crop":
            "Tomato",

        "disease":
            "Early blight",

        "health_status":
            "Diseased",

        "confidence":
            89.2
    }


    r_c = [

        {

            "source_doc":
                "disease_management_guide.txt",

            "text":
                "Early Blight treatment..."
        }

    ]


    adv = get_agricultural_advice(

        s_data,

        c_p,

        d_p,

        r_c
    )


    print(
        "\nGenerated Advice:"
    )

    print(
        json.dumps(
            adv,
            indent=2
        )
    )
