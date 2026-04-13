import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from tqdm import tqdm

from data.dataset import PlantDiseaseDataset, get_train_transforms, get_valid_transforms
from models.model import PlantClassifier

# Configuration
TRAIN_DIR = "data/raw/train" 
VALID_DIR = "data/raw/valid"
BATCH_SIZE = 32
EPOCHS = 5
MAX_LR = 0.01
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load Training Data (WITH Augmentation)
    train_dataset = PlantDiseaseDataset(data_dir=TRAIN_DIR, transform=get_train_transforms())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Load Validation Data (NO Augmentation, just resize/normalize)
    valid_dataset = PlantDiseaseDataset(data_dir=VALID_DIR, transform=get_valid_transforms())
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize Model
    model = PlantClassifier(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    # OneCycleLR Scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        epochs=EPOCHS, 
        steps_per_epoch=len(train_loader)
    )

    mlflow.set_experiment("Plant-Disease-Classification")
    
    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_lr": MAX_LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": "OneCycleLR"
        })

        print("Starting training and validation loop...")
        
        for epoch in range(EPOCHS):
            model.train() # Set model to training mode
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                sched.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'loss': loss.item()})

            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100 * train_correct / train_total

            model.eval() # Set model to evaluation mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Disable gradient calculation for validation
            with torch.no_grad():
                val_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / len(valid_loader)
            epoch_val_acc = 100 * val_correct / val_total

            print(f"Epoch {epoch+1} Results:")
            print(f"  Train -> Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
            print(f"  Valid -> Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")
            
            # Log to MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)


        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="models"
        )
        print("Training complete! Model securely logged.")

if __name__ == "__main__":
    train()