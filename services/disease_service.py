import os
import json
import re
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageStat, ImageOps
import numpy as np

class DiseaseService:
    def __init__(self, model_dir="models/disease_model"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = (
            model_dir if os.path.isabs(model_dir)
            else os.path.join(base_dir, model_dir)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_mapping = None
        self.crop_class_indices = {}
        self.crop_disease_indices = set()
        self.temperature_scale = 1.0
        self.architecture = "MobileNetV3-Small"
        self.field_test_summary = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "best_model.pth")
        mapping_path = os.path.join(self.model_dir, "class_names.json")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, "disease_model.pth")
        if not os.path.exists(mapping_path):
            mapping_path = os.path.join(self.model_dir, "class_mapping.json")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Class names mapping file not found at {mapping_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Disease model weights are missing. Train the model before enabling predictions."
            )

        with open(mapping_path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)
            self.class_mapping = {int(k): v for k, v in raw_mapping.items()}
        for class_index, raw_class in self.class_mapping.items():
            crop_name, _, health_status = self.parse_class_label(raw_class)
            crop_key = self._crop_key(crop_name)
            self.crop_class_indices.setdefault(crop_key, []).append(class_index)
            if health_status != "Healthy":
                self.crop_disease_indices.add(crop_key)

        config_path = os.path.join(self.model_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.temperature_scale = max(
                0.5, float(config.get("temperature_scale", 1.0))
            )
            self.architecture = config.get("architecture", self.architecture)
            self.field_test_summary = config.get("field_test") or None

        num_classes = len(self.class_mapping)
        self.model = models.mobilenet_v3_small(weights=None)
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)

        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)

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

            if min(image.size) < 96:
                return False, "Image resolution is too small. Upload a sharper photo taken closer to the leaf."
            if (r_std + g_std + b_std) < 12.0:
                return False, "Image has very little detail. Upload a clear, well-lit leaf photo."
                
            rgb = np.array(image.resize((100, 100)))
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            
            # Keep this check permissive so yellow leaves and brown lesions remain valid.
            greenish_pixels = np.sum((g > r * 0.78) & (g > b * 0.78) & (g > 30))
            brownish_pixels = np.sum((r > b) & (g > b * 0.68) & (r > 35) & (r < 235))
            total_px = 100 * 100
            foliage_ratio = (greenish_pixels + brownish_pixels) / total_px
            
            if foliage_ratio < 0.06:
                return False, "This photo does not show enough leaf area. Retake it with the leaf filling more of the frame."
                
            return True, "Valid leaf image"
        except Exception:
            # Fallback to True if stat check fails
            return True, "Valid leaf image"

    def parse_class_label(self, raw_label):
        if "___" in raw_label:
            parts = raw_label.split("___")
            raw_crop = parts[0].replace("_", " ").replace(",", "").strip()
            raw_disease = re.sub(r"\s+", " ", parts[1].replace("_", " ")).strip()
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

    @staticmethod
    def _crop_key(value):
        key = re.sub(r"[^a-z0-9]", "", str(value).lower())
        aliases = {
            "maize": "cornmaize",
            "corn": "cornmaize",
            "grapes": "grape",
            "pepper": "pepperbell",
            "bellpepper": "pepperbell",
            "capsicum": "pepperbell",
            "soyabean": "soybean",
            "soya": "soybean",
        }
        return aliases.get(key, key)

    def unsupported_crop_result(self, crop_hint):
        crop_name = str(crop_hint or "This crop").replace("_", " ").strip().title()
        return {
            "crop": crop_name,
            "disease": "Disease classes are not available for this crop",
            "health_status": "Unsupported crop",
            "confidence": 0.0,
            "reliability": "Unavailable",
            "interpretation": (
                f"AGriSense does not have a trained disease class for {crop_name} yet. "
                "Do not use another crop's result; ask your local KVK or agricultural extension officer to confirm symptoms."
            ),
            "confidence_note": "No prediction was made because this crop is outside the model's supported labels.",
            "top5_predictions": []
        }

    def predict_disease(self, image_path, crop_hint=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")

        try:
            with Image.open(image_path) as opened:
                raw_image = ImageOps.exif_transpose(opened).convert("RGB")
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

        hinted_key = self._crop_key(crop_hint) if crop_hint else None
        if hinted_key and (
            hinted_key not in self.crop_class_indices
            or hinted_key not in self.crop_disease_indices
        ):
            return self.unsupported_crop_result(crop_hint)

        img_tensor = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(img_tensor)
            if hinted_key:
                class_indices = self.crop_class_indices[hinted_key]
                masked_outputs = torch.full_like(outputs, float("-inf"))
                masked_outputs[:, class_indices] = outputs[:, class_indices]
                outputs = masked_outputs
            probabilities = torch.softmax(
                outputs / self.temperature_scale, dim=1
            )[0]
            candidate_count = len(class_indices) if hinted_key else len(self.class_mapping)
            topk_probs, topk_indices = torch.topk(
                probabilities,
                k=min(5, candidate_count)
            )

        topk_probs = [float(p) * 100.0 for p in topk_probs.cpu().numpy()]
        topk_indices = [int(i) for i in topk_indices.cpu().numpy()]

        top1_conf = topk_probs[0]
        top1_raw_class = self.class_mapping.get(topk_indices[0], "Unknown___healthy")
        crop, disease, health_status = self.parse_class_label(top1_raw_class)
        margin = top1_conf - (topk_probs[1] if len(topk_probs) > 1 else 0.0)

        if top1_conf >= 75.0 and margin >= 25.0:
            reliability = "High"
        elif top1_conf >= 50.0 and margin >= 10.0:
            reliability = "Moderate"
        else:
            reliability = "Low"

        field_accuracy = None
        field_note = ""
        if self.field_test_summary:
            field_accuracy = self.field_test_summary.get("accuracy")
            field_images = self.field_test_summary.get("images")
            if field_accuracy is not None:
                field_note = (
                    f" Held-out field-photo accuracy was {float(field_accuracy):.1f}% "
                    f"on {int(field_images or 0)} mapped images; farm photos can differ."
                )
                if float(field_accuracy) < 40.0:
                    reliability = "Low"
                elif float(field_accuracy) < 60.0 and reliability == "High":
                    reliability = "Moderate"

        if reliability == "Low":
            interpretation = (
                "The model is unsure. Retake the photo in daylight and compare the alternatives; "
                "ask a local agricultural officer to confirm before treating the crop."
                + field_note
            )
        elif health_status == "Healthy":
            interpretation = (
                f"The photo most closely matches the model's healthy {crop} examples. "
                "Continue normal scouting because the model covers only its listed classes."
                + field_note
            )
        else:
            interpretation = (
                f"The photo most closely matches the model's {disease} pattern on {crop}. "
                "Check the leaf and nearby plants before choosing a treatment."
                + field_note
            )

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
            "interpretation": interpretation,
            "top_two_margin": round(margin, 2),
            "confidence_note": (
                "A calibrated model score, not a confirmed field diagnosis." + field_note
            ),
            "model_name": self.architecture,
            "top5_predictions": top5_list
        }

_disease_service_instance = None

def predict_disease(image_path, crop_hint=None):
    global _disease_service_instance
    if _disease_service_instance is None:
        _disease_service_instance = DiseaseService()
    return _disease_service_instance.predict_disease(image_path, crop_hint=crop_hint)

if __name__ == "__main__":
    print("Testing Disease Service Top-5 predictions...")
    ds = DiseaseService()
    sample = r"F:\adithyan\agrisense\data\plantvillage\PlantVillage\val\Apple___Apple_scab\01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003.JPG"
    if os.path.exists(sample):
        res = ds.predict_disease(sample)
        print("Result Top-1:", res["crop"], "-", res["disease"], res["confidence"], "%")
        print("Top 5 predictions:", json.dumps(res["top5_predictions"], indent=2))
