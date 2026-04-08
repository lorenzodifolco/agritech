import torch
from src.data.dataset import PlantDiseaseDataset, get_train_transforms

# Make sure this path points to where your extracted Kaggle train folder is!
TRAIN_DIR = "data/raw/train" 

if __name__ == "__main__":
    # Initialize the dataset with our augmentation pipeline
    train_dataset = PlantDiseaseDataset(data_dir=TRAIN_DIR, transform=get_train_transforms())
    
    print(f"Total images loaded: {len(train_dataset)}")
    print(f"Total classes found: {len(train_dataset.classes)}")
    
    # Fetch the very first image
    image_tensor, label_idx = train_dataset[0]
    
    print(f"Successfully loaded an image!")
    print(f"Image Tensor Shape: {image_tensor.shape}") # Should be [3, 224, 224]
    print(f"Label Index: {label_idx} (Class: {train_dataset.classes[label_idx]})")