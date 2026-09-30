import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageStat, ImageFilter
import numpy as np

class DiseaseService:
    def __init__(self, model_dir="models/disease_model"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_mapping = None
        self.transform = transforms.Compose([
            # Match the preprocessing used by the script that produced the deployed weights.
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_model()

    def _load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(self.model_dir, "best_model.pth")
        mapping_path = os.path.join(self.model_dir, "class_names.json")

        if not os.path.exists(model_path):
            alt_dir = os.path.join(base_dir, self.model_dir)
            model_path = os.path.join(alt_dir, "best_model.pth")
            mapping_path = os.path.join(alt_dir, "class_names.json")

        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, self.model_dir, "disease_model.pth")

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Class names mapping file not found at {mapping_path}")

        with open(mapping_path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)
            self.class_mapping = {int(k): v for k, v in raw_mapping.items()}

        num_classes = len(self.class_mapping)
        self.model = models.mobilenet_v3_small(weights=None)
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model weights missing at {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def is_valid_leaf_image(self, image):
        """
        Checks basic visual characteristics to ensure image is a plant leaf or agricultural crop.
        Returns (is_valid: bool, reason: str)
        """
        try:
            stat = ImageStat.Stat(image)
            r_std, g_std, b_std = stat.stddev[:3]
            
            # Check for pure solid color or blank white/black image
            if (r_std + g_std + b_std) < 15.0:
                return False, "Image has low visual contrast or is a blank solid color."
                
            # Convert to HSV or check RGB ratio
            rgb = np.array(image.resize((100, 100)))
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            
            # Check presence of plant foliage hues (green, yellow-green, brown leaf lesions)
            greenish_pixels = np.sum((g > r * 0.8) & (g > b * 0.8) & (g > 30))
            brownish_pixels = np.sum((r > b) & (g > b * 0.7) & (r > 40) & (r < 220))
            total_px = 100 * 100
            foliage_ratio = (greenish_pixels + brownish_pixels) / total_px
            
            if foliage_ratio < 0.12:
                return False, "Unable to identify a plant leaf. Please upload a clear crop leaf image."
                
            return True, "Valid leaf image"
        except Exception:
            # Fallback to True if stat check fails
            return True, "Valid leaf image"

    def parse_class_label(self, raw_label):
        if "___" in raw_label:
            parts = raw_label.split("___")
            raw_crop = parts[0].replace("_", " ").strip()
            raw_disease = parts[1].replace("_", " ").strip()
        else:
            raw_crop = "Unknown Crop"
            raw_disease = raw_label.replace("_", " ").strip()

        if raw_disease.lower() == "healthy":
            health_status = "Healthy"
            disease = "None (Healthy)"
        else:
            health_status = "Diseased"
            disease = raw_disease

        return raw_crop, disease, health_status

    def predict_disease(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")

        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image file: {e}")

        # Quality / Leaf Check
        is_leaf, reason = self.is_valid_leaf_image(raw_image)
        if not is_leaf:
            return {
                "crop": "Unidentified",
                "disease": "No Plant Leaf Detected",
                "health_status": "Invalid Image",
                "confidence": 0.0,
                "reliability": "Invalid Image",
                "message": reason,
                "top5_predictions": []
            }

        img_tensor = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            topk_probs, topk_indices = torch.topk(probabilities, k=min(5, len(self.class_mapping)))

        topk_probs = [float(p) * 100.0 for p in topk_probs.cpu().numpy()]
        topk_indices = [int(i) for i in topk_indices.cpu().numpy()]

        top1_conf = topk_probs[0]
        top1_raw_class = self.class_mapping.get(topk_indices[0], "Unknown___healthy")
        crop, disease, health_status = self.parse_class_label(top1_raw_class)

        # Confidence reliability rating
        if top1_conf >= 80.0:
            reliability = "High"
        elif top1_conf >= 60.0:
            reliability = "Moderate"
        else:
            reliability = "Low"

        # Format Top 5 Predictions
        top5_list = []
        for idx, prob in zip(topk_indices, topk_probs):
            raw_cls = self.class_mapping.get(idx, "Unknown")
            c, d, h = self.parse_class_label(raw_cls)
            top5_list.append({
                "raw_class": raw_cls,
                "crop": c,
                "disease": d,
                "confidence": round(prob, 2)
            })

        return {
            "crop": crop,
            "disease": disease,
            "health_status": health_status,
            "confidence": round(top1_conf, 2),
            "reliability": reliability,
            "raw_class": top1_raw_class,
            "top5_predictions": top5_list
        }

_disease_service_instance = None

def predict_disease(image_path):
    global _disease_service_instance
    if _disease_service_instance is None:
        _disease_service_instance = DiseaseService()
    return _disease_service_instance.predict_disease(image_path)

if __name__ == "__main__":
    print("Testing Disease Service Top-5 predictions...")
    ds = DiseaseService()
    sample = r"F:\adithyan\agrisense\data\plantvillage\PlantVillage\val\Apple___Apple_scab\01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003.JPG"
    if os.path.exists(sample):
        res = ds.predict_disease(sample)
        print("Result Top-1:", res["crop"], "-", res["disease"], res["confidence"], "%")
        print("Top 5 predictions:", json.dumps(res["top5_predictions"], indent=2))
