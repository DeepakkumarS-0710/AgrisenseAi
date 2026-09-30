import os
import json
import random
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
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

def get_dataset_splits(pv_dir, max_samples_per_class=60):
    img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # Map class name -> list of all image paths
    class_to_files = {}
    
    for root, dirs, files in os.walk(pv_dir):
        image_files = [os.path.join(root, f) for f in files if f.endswith(img_extensions)]
        if image_files:
            c_name = os.path.basename(root)
            if c_name not in class_to_files:
                class_to_files[c_name] = []
            class_to_files[c_name].extend(image_files)
            
    class_names = sorted(list(class_to_files.keys()))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    print(f"Discovered {len(class_names)} unique disease/healthy classes.", flush=True)
    
    for c_name in class_names:
        c_idx = class_to_idx[c_name]
        files = class_to_files[c_name]
        
        # Shuffle reproducibly
        random.Random(SEED).shuffle(files)
        
        # Sample for fast, balanced training
        if len(files) > max_samples_per_class:
            files = files[:max_samples_per_class]
            
        # 70% train, 15% val, 15% test split
        tr, test_val = train_test_split(files, test_size=0.30, random_state=SEED)
        va, te = train_test_split(test_val, test_size=0.50, random_state=SEED)
        
        train_paths.extend(tr)
        train_labels.extend([c_idx] * len(tr))
        
        val_paths.extend(va)
        val_labels.extend([c_idx] * len(va))
        
        test_paths.extend(te)
        test_labels.extend([c_idx] * len(te))
        
    print(f"Dataset split completed:", flush=True)
    print(f"  - Train samples: {len(train_paths)}", flush=True)
    print(f"  - Val samples:   {len(val_paths)}", flush=True)
    print(f"  - Test samples:  {len(test_paths)}", flush=True)
    
    return class_names, class_to_idx, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def train_disease_model(data_dir, max_samples_per_class=60, epochs=3):
    pv_dir = os.path.join(data_dir, "plantvillage")
    if not os.path.isdir(pv_dir):
        raise FileNotFoundError(
            f"PlantVillage dataset not found at {pv_dir}. "
            "Pass the dataset root with --data-dir or AGRISENSE_DATA_DIR."
        )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "models", "disease_model")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading PlantVillage dataset from: {pv_dir}", flush=True)
    class_names, class_to_idx, train_data, val_data, test_data = get_dataset_splits(
        pv_dir, max_samples_per_class=max_samples_per_class
    )
    
    # Save class mapping
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    for mapping_name in ("class_mapping.json", "class_names.json"):
        mapping_path = os.path.join(output_dir, mapping_name)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_class, f, indent=2)
        print(f"Saved class mapping to {mapping_path}", flush=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    # Model Setup: MobileNetV3 Transfer Learning
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Freeze initial feature layers for fast transfer learning
    for param in model.features.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting Training Epochs...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for b_idx, (images, labels) in enumerate(train_loader):
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
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}", flush=True)

    # Evaluate on Test set
    print("\nEvaluating on Test Set...", flush=True)
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%", flush=True)
    
    # Save Model Weights
    for model_name in ("disease_model.pth", "best_model.pth"):
        model_pth_path = os.path.join(output_dir, model_name)
        torch.save(model.state_dict(), model_pth_path)
        print(f"Saved trained disease model weights to {model_pth_path}", flush=True)
    
    # Generate and Save Accuracy & Loss Graphs
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    graph_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Saved training metrics graph to {graph_path}", flush=True)
    
    # Classification report & Confusion matrix
    rep = classification_report(test_targets, test_preds, target_names=class_names, output_dict=True, zero_division=0)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
        
    cm = confusion_matrix(test_targets, test_preds).tolist()
    with open(os.path.join(output_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm, f, indent=2)
        
    print(f"Phase 3 Disease Model Training complete!", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PlantVillage disease model")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("AGRISENSE_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")),
        help="Dataset root containing plantvillage/",
    )
    parser.add_argument("--max-samples-per-class", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_disease_model(os.path.abspath(args.data_dir), args.max_samples_per_class, args.epochs)
