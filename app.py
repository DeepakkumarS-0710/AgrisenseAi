import os
import sys
import json
import math
import uuid
from functools import wraps
import werkzeug.utils

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash
)


# ============================================================
# PROJECT ROOT
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)


# ============================================================
# DATABASE
# ============================================================

from database.db import (
    init_db,
    create_user,
    get_user_by_email,
    get_user_by_id,
    verify_user_password,
    save_sensor_reading,
    get_latest_sensor_reading,
    get_sensor_history,
    get_sensor_readings_since,
    save_analysis_result,
    get_analysis_history
)


# ============================================================
# SERVICES
# ============================================================

from services.crop_service import predict_crop
from services.tamil_nadu_crop_catalog import get_tamil_nadu_crop_catalog
from services.disease_service import predict_disease
from services.monitoring_service import analyze_soil_week
from services.analysis_service import run_multimodal_analysis
from rag.retriever import retrieve_knowledge
from services.llm_service import build_disease_management


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "agrisense_secret_key_super_secure_2026"
)

app.config["UPLOAD_FOLDER"] = os.path.join(
    "static",
    "uploads"
)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

init_db()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("login_page")
            )

        return f(*args, **kwargs)

    return decorated_function


# ============================================================
# USER CONTEXT
# ============================================================

@app.context_processor
def inject_user():

    user = None

    if "user_id" in session:

        user = get_user_by_id(
            session["user_id"]
        )

    return dict(
        current_user=user
    )


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route("/")
def index():

    if "user_id" in session:

        return redirect(
            url_for("dashboard_page")
        )

    return redirect(
        url_for("login_page")
    )


# ------------------------------------------------------------
# LOGIN
# ------------------------------------------------------------

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login_page():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()


        if not email or not password:

            flash(
                "Please enter both email and password.",
                "error"
            )

            return render_template(
                "login.html"
            )


        user = get_user_by_email(
            email
        )


        if user and verify_user_password(
            user,
            password
        ):

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            flash(
                "Login successful!",
                "success"
            )

            return redirect(
                url_for("dashboard_page")
            )

        else:

            flash(
                "Invalid email or password. Please try again.",
                "error"
            )


    return render_template(
        "login.html"
    )


# ------------------------------------------------------------
# SIGNUP
# ------------------------------------------------------------

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup_page():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        confirm_password = request.form.get(
            "confirm_password",
            ""
        ).strip()


        if not name or not email or not password:

            flash(
                "All fields are required.",
                "error"
            )

            return render_template(
                "signup.html"
            )


        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template(
                "signup.html"
            )


        if len(password) < 6:

            flash(
                "Password must be at least 6 characters long.",
                "error"
            )

            return render_template(
                "signup.html"
            )


        try:

            user_id = create_user(
                name,
                email,
                password
            )

            flash(
                "Account created successfully! Please log in.",
                "success"
            )

            return redirect(
                url_for("login_page")
            )


        except ValueError as err:

            flash(
                str(err),
                "error"
            )


        except Exception as e:

            flash(
                f"Registration error: {str(e)}",
                "error"
            )


    return render_template(
        "signup.html"
    )


# ------------------------------------------------------------
# LOGOUT
# ------------------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(
        url_for("login_page")
    )


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard_page():

    latest_sensors = get_latest_sensor_reading()

    recent_analyses = get_analysis_history(
        user_id=session.get("user_id"),
        limit=5
    )

    return render_template(
        "dashboard.html",
        latest_sensors=latest_sensors,
        recent_analyses=recent_analyses
    )


@app.route("/soil-monitoring")
@login_required
def soil_monitoring_page():
    return render_template("soil_monitoring.html")


@app.route("/crop-analysis")
@login_required
def crop_analysis_page():

    latest_sensors = get_latest_sensor_reading()

    return render_template(
        "crop_analysis.html",
        latest_sensors=latest_sensors,
        tamil_nadu_crops=get_tamil_nadu_crop_catalog()
    )


@app.route("/disease-detection")
@login_required
def disease_detection_page():

    return render_template(
        "disease_detection.html"
    )


@app.route("/advisor")
@login_required
def advisor_page():

    latest_sensors = get_latest_sensor_reading()

    return render_template(
        "advisor.html",
        latest_sensors=latest_sensors
    )


@app.route("/history")
@login_required
def history_page():

    user_id = session.get(
        "user_id"
    )

    history_items = get_analysis_history(
        user_id=user_id,
        limit=50
    )

    return render_template(
        "history.html",
        history=history_items
    )


@app.route("/about")
def about_page():

    return render_template(
        "about.html"
    )


# ============================================================
# FULL MULTIMODAL ANALYSIS
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
@login_required
def analyze():

    try:

        # ----------------------------------------------------
        # GET INPUT DATA
        # ----------------------------------------------------

        data = (
            request.get_json()
            if request.is_json
            else request.form.to_dict()
        )


        soil_moisture = data.get(
            "soil_moisture"
        )


        # ----------------------------------------------------
        # CONVERT NUMERICAL VALUES
        # ----------------------------------------------------

        try:

            temperature = float(
                data.get(
                    "temperature",
                    25.0
                )
            )

            humidity = float(
                data.get(
                    "humidity",
                    70.0
                )
            )

            ph = float(
                data.get(
                    "ph",
                    6.5
                )
            )

            nitrogen = float(
                data.get(
                    "nitrogen",
                    50.0
                )
            )

            phosphorus = float(
                data.get(
                    "phosphorus",
                    50.0
                )
            )

            potassium = float(
                data.get(
                    "potassium",
                    50.0
                )
            )

            rainfall = float(
                data.get(
                    "rainfall",
                    100.0
                )
            )


        except (
            ValueError,
            TypeError
        ) as val_err:

            return jsonify({
                "status": "error",
                "message":
                    f"Invalid numerical sensor value: {val_err}"
            }), 400


        # ====================================================
        # VALIDATE SENSOR VALUES
        # ====================================================

        # Rainfall changed from 1000 to 5000 mm
        # so high-rainfall crops can be tested.

        if not (
            0 <= temperature <= 60
            and 0 <= humidity <= 100
            and 0 <= ph <= 14
            and 0 <= nitrogen <= 300
            and 0 <= phosphorus <= 300
            and 0 <= potassium <= 300
            and 0 <= rainfall <= 5000
        ):

            return jsonify({
                "status": "error",
                "message":
                    "Sensor values are out of reasonable biological ranges."
            }), 400


        # ====================================================
        # LEAF IMAGE UPLOAD
        # ====================================================

        image_path = None
        crop_hint = str(data.get("crop_type", "")).strip()


        if "image" in request.files:

            file = request.files["image"]


            if file and file.filename != "":

                if not crop_hint:
                    return jsonify({
                        "status": "error",
                        "message": "Choose the crop before submitting a leaf photo so the model checks the right disease classes."
                    }), 400

                if not allowed_file(
                    file.filename
                ):

                    return jsonify({
                        "status": "error",
                        "message":
                            "Unsupported image format. "
                            "Allowed formats: PNG, JPG, JPEG, WEBP."
                    }), 400


                file_extension = os.path.splitext(
                    werkzeug.utils.secure_filename(file.filename)
                )[1].lower()
                filename = uuid.uuid4().hex + file_extension


                save_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )


                file.save(
                    save_path
                )


                image_path = save_path


        # ====================================================
        # SAVE SENSOR DATA
        # ====================================================

        save_sensor_reading({

            "soil_moisture":
                soil_moisture
                if soil_moisture is not None
                else "Not Available",

            "temperature":
                temperature,

            "humidity":
                humidity,

            "ph":
                ph,

            "nitrogen":
                nitrogen,

            "phosphorus":
                phosphorus,

            "potassium":
                potassium,

            "rainfall":
                rainfall
        })


        # ====================================================
        # RUN MULTIMODAL ANALYSIS
        # ====================================================

        analysis_result = run_multimodal_analysis(

            soil_moisture=soil_moisture,

            temperature=temperature,

            humidity=humidity,

            ph=ph,

            nitrogen=nitrogen,

            phosphorus=phosphorus,

            potassium=potassium,

            rainfall=rainfall,

            image_path=image_path,

            crop_hint=crop_hint or None
        )


        # ====================================================
        # SAVE ANALYSIS
        # ====================================================

        user_id = session.get(
            "user_id"
        )


        analysis_id = save_analysis_result(

            analysis_result,

            user_id=user_id,

            image_path=image_path
        )


        analysis_result["analysis_id"] = analysis_id


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "status": "success",

            "data": analysis_result
        })


    except Exception as e:

        print(
            f"Error during /analyze: {e}"
        )

        return jsonify({

            "status": "error",

            "message":
                f"Server Analysis Failure: {str(e)}"

        }), 500


# ============================================================
# CROP PREDICTION API
# ============================================================

@app.route(
    "/predict-crop",
    methods=["POST"]
)
@login_required
def api_predict_crop():

    try:

        # ----------------------------------------------------
        # GET DATA
        # ----------------------------------------------------

        data = (
            request.get_json()
            if request.is_json
            else request.form.to_dict()
        )

        ranges = {
            "nitrogen": (0, 300),
            "phosphorus": (0, 300),
            "potassium": (0, 300),
            "temperature": (0, 60),
            "humidity": (0, 100),
            "ph": (0, 14),
            "rainfall": (0, 5000),
        }
        values = {}
        for key, (minimum, maximum) in ranges.items():
            try:
                value = float(data.get(key, ""))
            except (AttributeError, TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": f"Enter a valid {key.replace('_', ' ')} value."
                }), 400
            if not math.isfinite(value) or not minimum <= value <= maximum:
                return jsonify({
                    "status": "error",
                    "message": f"{key.replace('_', ' ').capitalize()} must be between {minimum} and {maximum}."
                }), 400
            values[key] = value


        # ----------------------------------------------------
        # PREDICT CROP
        # ----------------------------------------------------

        res = predict_crop(
            N=values["nitrogen"],
            P=values["phosphorus"],
            K=values["potassium"],
            temperature=values["temperature"],
            humidity=values["humidity"],
            ph=values["ph"],
            rainfall=values["rainfall"],

            planned_crop=data.get(
                "planned_crop"
            )
        )


        # ----------------------------------------------------
        # RETURN CLEAR MODEL RESULT
        # ----------------------------------------------------

        return jsonify({

            "status": "success",

            "data": res

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 400


# ============================================================
# DISEASE DETECTION API
# ============================================================

@app.route(
    "/detect-disease",
    methods=["POST"]
)
@login_required
def api_detect_disease():

    try:

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "status": "error",

                "message":
                    "No crop leaf image uploaded."

            }), 400


        file = request.files["image"]


        if file.filename == "":

            return jsonify({

                "status": "error",

                "message":
                    "No image selected."

            }), 400


        crop_type = request.form.get("crop_type", "").strip()
        if not crop_type:
            return jsonify({
                "status": "error",
                "message": "Choose the crop before uploading the leaf image."
            }), 400


        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "status": "error",

                "message":
                    "Unsupported image format."

            }), 400


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        file_extension = os.path.splitext(
            werkzeug.utils.secure_filename(file.filename)
        )[1].lower()
        filename = uuid.uuid4().hex + file_extension


        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        file.save(
            save_path
        )


        # ----------------------------------------------------
        # PREDICT DISEASE
        # ----------------------------------------------------

        res = predict_disease(
            save_path,
            crop_hint=crop_type
        )


        # ----------------------------------------------------
        # RAG KNOWLEDGE
        # ----------------------------------------------------

        rag_query = (
            f"{res.get('crop', '')} "
            f"{res.get('disease', '')} "
            f"symptoms causes treatment prevention "
            f"disease management"
        )


        rag_sources = retrieve_knowledge(
            rag_query,
            top_k=5
        )


        res["rag_sources"] = rag_sources


        res["management"] = build_disease_management(
            res,
            rag_sources
        )

        grounded_sources = (res["management"] or {}).get("sources", [])
        if res.get("health_status") in {"Unsupported crop", "Invalid Image"}:
            grounded_sources = []

        if res.get("reliability") != "High":
            res["management"] = {
                "title": "Confirm the image result first",
                "status": "Needs confirmation",
                "summary": res.get("interpretation", "The image result is uncertain."),
                "treatment": [
                    "Photograph several affected leaves and a healthy leaf in daylight.",
                    "Ask your local KVK or agricultural extension officer to confirm the symptoms before treatment."
                ],
                "prevention": [
                    "Avoid moving between wet plants and clean tools after handling affected leaves.",
                    "Do not apply pesticides or remove whole plants based only on this image score."
                ],
                "monitoring": "Check nearby plants over the next few days and note whether symptoms are spreading.",
                "grounding": "The image match needs confirmation. Sources below relate to the model's tentative label; disease-specific treatment is held until the match is stronger.",
                "sources": grounded_sources
            }


        return jsonify({

            "status": "success",

            "data": res

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ============================================================
# IOT SENSOR API
# ============================================================

@app.route(
    "/api/sensors",
    methods=["POST"]
)
def post_iot_sensors():

    try:

        raw_data = request.get_json(
            force=True
        )


        if not raw_data:

            return jsonify({

                "status": "error",

                "message":
                    "Invalid JSON body"

            }), 400

        if not isinstance(raw_data, dict):
            return jsonify({
                "status": "error",
                "message": "Sensor readings must be sent as a JSON object."
            }), 400

        aliases = {
            "n": "nitrogen",
            "p": "phosphorus",
            "k": "potassium",
            "ph": "ph",
            "soilmoisture": "soil_moisture",
            "air_temperature": "temperature",
            "air_humidity": "humidity",
            "rain": "rainfall",
        }
        data = {}
        for key, value in raw_data.items():
            normalized = str(key).strip().lower()
            data[aliases.get(normalized, normalized)] = value


        sensor_bounds = {
            "soil_moisture": (0, 100),
            "temperature": (-10, 70),
            "humidity": (0, 100),
            "ph": (0, 14),
            "nitrogen": (0, 1000),
            "phosphorus": (0, 1000),
            "potassium": (0, 1000),
            "rainfall": (0, 5000),
        }
        received = 0
        for key, (minimum, maximum) in sensor_bounds.items():
            value = data.get(key)
            if value is None or str(value).strip() == "":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": f"{key.replace('_', ' ').capitalize()} must be a number."
                }), 400
            if not minimum <= numeric_value <= maximum:
                return jsonify({
                    "status": "error",
                    "message": f"{key.replace('_', ' ').capitalize()} is outside the supported range."
                }), 400
            received += 1

        if received == 0:
            return jsonify({
                "status": "error",
                "message": "Include at least one supported sensor value."
            }), 400


        reading_id = save_sensor_reading(
            data
        )


        return jsonify({

            "status": "success",

            "message":
                "Sensor reading saved",

            "reading_id":
                reading_id

        }), 201


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 400


# ============================================================
# GET LATEST SENSOR DATA
# ============================================================

@app.route(
    "/api/sensors/latest",
    methods=["GET"]
)
def get_latest_sensors_api():

    reading = get_latest_sensor_reading()


    if reading:

        return jsonify({

            "status": "success",

            "data": reading

        })


    return jsonify({

        "status": "success",

        "data": None,

        "message":
            "No sensor readings stored yet."

    })


# ============================================================
# GET SENSOR HISTORY
# ============================================================

@app.route(
    "/api/sensors/history",
    methods=["GET"]
)
def get_sensors_history_api():

    history = get_sensor_history()


    return jsonify({

        "status": "success",

        "data": history

    })


@app.route("/api/sensors/week", methods=["GET"])
@login_required
def get_weekly_sensor_summary_api():
    readings = get_sensor_readings_since(days=7, limit=10000)
    return jsonify({
        "status": "success",
        "data": analyze_soil_week(readings),
    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "Starting AgriSense Flask Multi-Page Server "
        "on http://127.0.0.1:5000"
    )


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )
