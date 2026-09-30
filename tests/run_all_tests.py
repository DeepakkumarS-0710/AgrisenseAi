import os
import sys
import json
import unittest
import glob
from PIL import Image

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import app
from database.db import init_db
from services.crop_service import predict_crop
from services.disease_service import predict_disease
from rag.retriever import retrieve_knowledge
from services.analysis_service import run_multimodal_analysis

class TestAGriSenseSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_1_crop_prediction(self):
        """Test ML Crop Recommendation Model"""
        res = predict_crop(N=90, P=42, K=43, temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.9)
        self.assertIn("recommended_crop", res)
        self.assertIn("confidence", res)
        self.assertIsInstance(res["confidence"], float)
        self.assertEqual(res["recommended_crop"].lower(), "rice")
        print("[OK] Test 1 Passed: Crop Recommendation Model prediction")

    def test_2_disease_prediction(self):
        """Test Deep Learning Disease Detection Model with actual PlantVillage image"""
        sample_imgs = glob.glob(r"F:\adithyan\agrisense\data\plantvillage\**\*.jpg", recursive=True)
        self.assertTrue(len(sample_imgs) > 0, "PlantVillage sample images found")
        
        img_path = sample_imgs[0]
        res = predict_disease(img_path)
        self.assertIn("crop", res)
        self.assertIn("disease", res)
        self.assertIn("health_status", res)
        self.assertIn("confidence", res)
        print(f"[OK] Test 2 Passed: Disease Detection Model on {os.path.basename(img_path)} -> {res['crop']}, {res['disease']}")

    def test_3_rag_retrieval(self):
        """Test RAG FAISS Vector Retriever"""
        results = retrieve_knowledge("Apple scab symptoms and treatment", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("source_doc", results[0])
        self.assertIn("text", results[0])
        print("[OK] Test 3 Passed: RAG Retrieval from FAISS vector store")

    def test_4_combined_multimodal_analysis(self):
        """Test End-to-End Multimodal Analysis Pipeline"""
        res = run_multimodal_analysis(
            soil_moisture=45.0,
            temperature=27.0,
            humidity=75.0,
            ph=6.5,
            nitrogen=60,
            phosphorus=40,
            potassium=50,
            rainfall=120.0
        )
        self.assertIn("sensor_data", res)
        self.assertIn("crop_recommendation", res)
        self.assertIn("disease_detection", res)
        self.assertIn("rag_sources", res)
        self.assertIn("advisor", res)
        
        # Verify soil moisture is NOT replaced with humidity
        self.assertEqual(res["sensor_data"]["soil_moisture"], 45.0)
        print("[OK] Test 4 Passed: Combined Multimodal Analysis Pipeline")

    def test_5_iot_sensor_api(self):
        """Test IoT Sensor API Endpoints"""
        payload = {
            "soil_moisture": 42,
            "temperature": 29.4,
            "humidity": 72,
            "ph": 6.4,
            "nitrogen": 48,
            "phosphorus": 31,
            "potassium": 40,
            "rainfall": 12
        }
        # POST /api/sensors
        response = self.client.post('/api/sensors', json=payload)
        self.assertEqual(response.status_code, 201)
        res_json = response.get_json()
        self.assertEqual(res_json["status"], "success")

        # GET /api/sensors/latest
        response_latest = self.client.get('/api/sensors/latest')
        self.assertEqual(response_latest.status_code, 200)
        latest_json = response_latest.get_json()
        self.assertEqual(latest_json["data"]["temperature"], 29.4)
        print("[OK] Test 5 Passed: IoT REST API Endpoints (POST & GET)")

    def test_6_web_routes(self):
        """Test Flask Dashboard & History Page routes"""
        res_sensors = self.client.get('/sensors')
        self.assertEqual(res_sensors.status_code, 404)

        res_dash = self.client.get('/dashboard')
        self.assertEqual(res_dash.status_code, 302)
        self.assertIn('/login', res_dash.location)

        res_hist = self.client.get('/history')
        self.assertEqual(res_hist.status_code, 302)
        self.assertIn('/login', res_hist.location)
        print("[OK] Test 6 Passed: Flask Web Routes GET /dashboard and GET /history")

if __name__ == '__main__':
    unittest.main()
