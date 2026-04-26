import torch.nn as nn
from torchvision import models

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=38, pretrained=True):
        super(PlantClassifier, self).__init__()

        # pretrained=True loads ImageNet weights for transfer learning (training only).
        # Pass pretrained=False at inference time — weights come from model.pth instead.
        self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Unfreeze all layers for fine-tuning (you can choose to freeze some layers if you want)
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # Replace the final fully connected layer to output our 38 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), # Prevent overfitting
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Make sure the new fully connected layer IS trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)