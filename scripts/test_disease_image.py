import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def predict_single_image(image_path, show=False):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        sys.exit(1)

    model_dir = os.path.join(BASE_DIR, "models", "disease_model")
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "disease_model.pth")
    
    class_names_path = os.path.join(model_dir, "class_names.json")
    
    if not os.path.exists(class_names_path):
        print(f"Error: Class mapping file not found at {class_names_path}")
        sys.exit(1)

    with open(class_names_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
        idx_to_class = {int(k): v for k, v in raw_mapping.items()}

    num_classes = len(idx_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MobileNetV3 model
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing matching validation transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        raw_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    tensor_img = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, k=min(5, num_classes))

    top5_probs = [float(p) * 100.0 for p in top5_probs.cpu().numpy()]
    top5_indices = [int(i) for i in top5_indices.cpu().numpy()]

    top1_class = idx_to_class[top5_indices[0]]
    top1_conf = top5_probs[0]

    if "___" in top1_class:
        crop_part, disease_part = top1_class.split("___")
        crop = crop_part.replace("_", " ").strip()
        disease_clean = disease_part.replace("_", " ").strip()
    else:
        crop = "Unknown Crop"
        disease_clean = top1_class.replace("_", " ").strip()

    if disease_clean.lower() == "healthy":
        health_status = "Healthy"
        disease = "None (Healthy)"
    else:
        health_status = "Diseased"
        disease = disease_clean

    print("==================================================")
    print("AGRISENSE PLANT DISEASE INFERENCE REPORT")
    print("==================================================")
    print(f"Image File:       {os.path.basename(image_path)}")
    print(f"Predicted Crop:   {crop}")
    print(f"Predicted Disease:{disease}")
    print(f"Health Status:    {health_status}")
    print(f"Top-1 Confidence: {top1_conf:.2f}%")
    print("\nTop 5 Predictions:")
    
    top5_list = []
    for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), 1):
        c_name = idx_to_class[idx]
        print(f"  {rank}. {c_name} – {prob:.1f}%")
        top5_list.append({"class_name": c_name, "confidence": round(prob, 2)})
    print("==================================================")

    if show:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(raw_img)
            plt.title(f"{crop} - {disease} ({top1_conf:.1f}%)")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

    return {
        "crop": crop,
        "disease": disease,
        "health_status": health_status,
        "confidence": round(top1_conf, 2),
        "raw_class": top1_class,
        "top5_predictions": top5_list
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Plant Disease Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to crop leaf image")
    parser.add_argument("--show", action="store_true", help="Display image using matplotlib")
    args = parser.parse_args()

    predict_single_image(args.image, args.show)
