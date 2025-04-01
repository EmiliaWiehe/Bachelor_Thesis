import torch
import torch.nn as nn
from torchvision import models

def build_resnet_model(pretrained=False, num_classes=4):
    """
    Build a ResNet50 model for multi-label classification.

    Args:
    - pretrained (bool): Whether to load ImageNet pre-trained weights.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - model (nn.Module): PyTorch model with the ResNet50 backbone.
    """
    # Handle correct loading of pretrained weights
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Replace avg pooling with Adaptive Pooling (to prevent shape mismatch)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # Modify the fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.BatchNorm1d(128),  # Added BatchNorm
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)  # Output layer remains unchanged
    )

    return model