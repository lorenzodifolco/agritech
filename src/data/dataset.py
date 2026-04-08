import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom PyTorch Dataset for the Plant Disease images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get all class folders (the 38 diseases/healthy classes)
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and their corresponding labels
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using OpenCV (Albumentations expects OpenCV format: BGR)
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        # Apply Albumentations transformations if any
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# --- DEFINE THE ALBUMENTATIONS PIPELINE ---
def get_train_transforms():
    """
    Data augmentation pipeline for training data.
    """
    return A.Compose([
        A.Resize(224, 224), # Standard size for models like ResNet
        A.HorizontalFlip(p=0.5), # 50% chance to flip horizontally
        A.RandomBrightnessContrast(p=0.2), # Simulate different sunlight
        A.Rotate(limit=30, p=0.5), # Simulate different camera angles
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet standards
        ToTensorV2() # Convert to PyTorch Tensor
    ])

def get_valid_transforms():
    """
    Transforms for validation data (NO augmentation, just resize and normalize).
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])