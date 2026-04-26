import torch
from src.models.model import PlantClassifier

# Create dummy model with random weights
model = PlantClassifier(num_classes=38)

# Save it as model.pth
torch.save(model.state_dict(), "model.pth")
print("Dummy model saved as model.pth")