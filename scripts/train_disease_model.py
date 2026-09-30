import os
import sys
import json
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set random seeds for strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PlantDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_and_split_dataset(pv_dir, max_train_per_class=150, max_val_per_class=35, max_test_per_class=35):
    img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # 1. Map class name -> set of unique file paths (prevent duplicates)
    class_to_files = {}
    
    for root, dirs, files in os.walk(pv_dir):
        image_files = [os.path.join(root, f) for f in files if f.endswith(img_extensions)]
        if image_files:
            c_name = os.path.basename(root)
            if c_name not in class_to_files:
                class_to_files[c_name] = set()
            class_to_files[c_name].update(image_files)
            
    class_names = sorted(list(class_to_files.keys()))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {str(i): c for i, c in enumerate(class_names)}
    
    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    print(f"Discovered {len(class_names)} unique disease/healthy classes.", flush=True)
    
    for c_name in class_names:
        c_idx = class_to_idx[c_name]
        files = sorted(list(class_to_files[c_name]))
        
        # Shuffle reproducibly
        random.Random(SEED).shuffle(files)
        
        # Split 70% train, 15% val, 15% test BEFORE augmentation
        tr, test_val = train_test_split(files, test_size=0.30, random_state=SEED)
        va, te = train_test_split(test_val, test_size=0.50, random_state=SEED)
        
        # Apply cap if needed for balanced CPU training speed
        if len(tr) > max_train_per_class:
            tr = tr[:max_train_per_class]
        if len(va) > max_val_per_class:
            va = va[:max_val_per_class]
        if len(te) > max_test_per_class:
            te = te[:max_test_per_class]
            
        train_paths.extend(tr)
        train_labels.extend([c_idx] * len(tr))
        
        val_paths.extend(va)
        val_labels.extend([c_idx] * len(va))
        
        test_paths.extend(te)
        test_labels.extend([c_idx] * len(te))
        
    print(f"Dataset split completed without data leakage:")
    print(f"  - Train samples: {len(train_paths)}")
    print(f"  - Val samples:   {len(val_paths)}")
    print(f"  - Test samples:  {len(test_paths)}")
    
    return class_names, idx_to_class, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def train_and_evaluate():
    pv_dir = r"F:\adithyan\agrisense\data\plantvillage"
    output_dir = os.path.join("models", "disease_model")
    reports_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print(f"Loading PlantVillage dataset from: {pv_dir}", flush=True)
    class_names, idx_to_class, train_data, val_data, test_data = load_and_split_dataset(
        pv_dir, max_train_per_class=160, max_val_per_class=35, max_test_per_class=35
    )
    
    # Save class_names.json mapping index -> class name
    class_names_path = os.path.join(output_dir, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Saved class mapping to {class_names_path}", flush=True)
    
    # Standard ImageNet Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # DataLoaders
    train_ds = PlantDataset(train_data[0], train_data[1], transform=train_transform)
    val_ds = PlantDataset(val_data[0], val_data[1], transform=val_test_transform)
    test_ds = PlantDataset(test_data[0], test_data[1], transform=val_test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)
    
    # Pre-trained MobileNetV3 Small transfer learning
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Unfreeze upper layers for fine-tuning
    for param in model.features.parameters():
        param.requires_grad = True
        
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    epochs = 5
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pth")
    
    print("\nStarting Training & Fine-Tuning...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved best model checkpoint to {best_model_path}", flush=True)

    # Load best checkpoint for final testing
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    print("\nEvaluating Best Model on Held-out Test Set...", flush=True)
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)
    
    print(f"\n==========================================")
    print(f"FINAL TEST METRICS:")
    print(f"  - Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"  - Macro Precision:   {macro_prec:.4f}")
    print(f"  - Macro Recall:      {macro_rec:.4f}")
    print(f"  - Macro F1-score:    {macro_f1:.4f}")
    print(f"==========================================\n")
    
    # Save model_config.json
    config_data = {
        "architecture": "MobileNetV3-Small Transfer Learning",
        "num_classes": len(class_names),
        "image_size": [224, 224],
        "normalization": {"mean": mean, "std": std},
        "class_mapping": idx_to_class,
        "test_accuracy": round(test_acc * 100, 2),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4)
    }
    with open(os.path.join(output_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
        
    # Save disease_test_results.json
    with open(os.path.join(reports_dir, "disease_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Save disease_classification_report.csv
    report_dict = classification_report(test_targets, test_preds, target_names=class_names, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(report_dict).T
    df_rep.to_csv(os.path.join(reports_dir, "disease_classification_report.csv"))
    print(f"Saved classification report to reports/disease_classification_report.csv")

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('PlantVillage Disease Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [c.split('___')[-1] for c in class_names], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [c.split('___')[-1] for c in class_names], fontsize=8)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(reports_dir, "disease_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to reports/disease_confusion_matrix.png")

if __name__ == "__main__":
    train_and_evaluate()
