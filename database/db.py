import os
import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join("database", "agrisense.db")

def get_db_connection():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_file = os.path.join(base_dir, DB_PATH)
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table 1: users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table 2: sensor_readings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        soil_moisture TEXT,
        temperature REAL,
        humidity REAL,
        ph REAL,
        nitrogen REAL,
        phosphorus REAL,
        potassium REAL,
        rainfall REAL
    )
    """)
    
    # Table 3: analysis_results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NULLABLE,
        image_path TEXT NULLABLE,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        recommended_crop TEXT,
        crop_confidence REAL,
        disease TEXT,
        disease_confidence REAL,
        health_status TEXT,
        risk_level TEXT,
        recommendation TEXT,
        rag_sources TEXT,
        sensor_data TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(analysis_results)").fetchall()
    }
    missing_columns = {
        "user_id": "INTEGER",
        "image_path": "TEXT",
    }
    for column_name, column_type in missing_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE analysis_results ADD COLUMN {column_name} {column_type}"
            )
    
    conn.commit()
    conn.close()
    print(f"Database schema initialized at {DB_PATH}")

# ==========================================
# USER AUTHENTICATION HELPERS
# ==========================================

def create_user(name, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    email_clean = email.strip().lower()
    # Check if email already exists
    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email_clean,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("An account with this email address already exists.")
        
    pwd_hash = generate_password_hash(password)
    cursor.execute("""
    INSERT INTO users (name, email, password_hash)
    VALUES (?, ?, ?)
    """, (name.strip(), email_clean, pwd_hash))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id

def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def verify_user_password(user, password):
    if not user or 'password_hash' not in user:
        return False
    return check_password_hash(user['password_hash'], password)

# ==========================================
# SENSOR & ANALYSIS HELPERS
# ==========================================

def save_sensor_reading(data):
    conn = get_db_connection()
    cursor = conn.cursor()

    def optional_float(key):
        value = data.get(key)
        if value is None or str(value).strip().lower() in {
            "", "none", "null", "not available", "n/a"
        }:
            return None
        return float(value)

    sm_value = optional_float("soil_moisture")

    cursor.execute("""
    INSERT INTO sensor_readings (soil_moisture, temperature, humidity, ph, nitrogen, phosphorus, potassium, rainfall)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(sm_value) if sm_value is not None else None,
        optional_float("temperature"),
        optional_float("humidity"),
        optional_float("ph"),
        optional_float("nitrogen"),
        optional_float("phosphorus"),
        optional_float("potassium"),
        optional_float("rainfall")
    ))
    reading_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return reading_id

def get_latest_sensor_reading():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_sensor_history(limit=20):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sensor_readings_since(days=7, limit=10000):
    days = max(1, min(int(days), 30))
    limit = max(1, min(int(limit), 20000))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM (
            SELECT * FROM sensor_readings
            WHERE datetime(timestamp) >= datetime('now', ?)
            ORDER BY datetime(timestamp) DESC, id DESC
            LIMIT ?
        )
        ORDER BY datetime(timestamp) ASC, id ASC
        """,
        (f"-{days} days", limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_analysis_result(analysis, user_id=None, image_path=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    crop_info = analysis.get("crop_recommendation", {})
    disease_info = analysis.get("disease_detection", {})
    advisor = analysis.get("advisor", {})
    rag_sources = analysis.get("rag_sources", [])
    sensor_data = analysis.get("sensor_data", {})
    
    cursor.execute("""
    INSERT INTO analysis_results (
        user_id, image_path, recommended_crop, crop_confidence, disease, disease_confidence,
        health_status, risk_level, recommendation, rag_sources, sensor_data
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        image_path,
        crop_info.get("recommended_crop", "Unknown"),
        float(crop_info.get("confidence", 0.0)),
        disease_info.get("disease", "Unknown"),
        float(disease_info.get("confidence", 0.0)),
        disease_info.get("health_status", "Unknown"),
        advisor.get("risk_level", "Low"),
        json.dumps(advisor),
        json.dumps(rag_sources),
        json.dumps(sensor_data)
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

def get_analysis_history(user_id=None, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("SELECT * FROM analysis_results WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
    else:
        cursor.execute("SELECT * FROM analysis_results ORDER BY id DESC LIMIT ?", (limit,))
        
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["recommendation"] = json.loads(d["recommendation"])
        except Exception:
            pass
        try:
            d["rag_sources"] = json.loads(d["rag_sources"])
        except Exception:
            pass
        try:
            d["sensor_data"] = json.loads(d["sensor_data"])
        except Exception:
            pass
        results.append(d)
    return results

if __name__ == "__main__":
    init_db()
