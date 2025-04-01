import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

def build_vit_model(num_classes=4, pretrained=True, multi_label=False):
    """
    Builds a Vision Transformer (ViT) model for classification using Hugging Face pre-trained weights.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pre-trained weights from Hugging Face.
        multi_label (bool): Whether the task is multi-label classification.

    Returns:
        model (torch.nn.Module): ViT model with a custom classification head.
    """
    # Load ViT configuration and model
    model_name = "google/vit-base-patch16-224-in21k" 
    config = ViTConfig.from_pretrained(model_name)
    config.num_labels = num_classes  # Adjust output layer for classification

    # Load ViT model (with or without pretrained weights)
    vit = ViTModel.from_pretrained(model_name) if pretrained else ViTModel(config)

    # Define custom classification head
    class ViTClassifier(nn.Module):
        def __init__(self, vit, num_classes, multi_label):
            super(ViTClassifier, self).__init__()
            self.vit = vit
            self.multi_label = multi_label
            self.classifier = nn.Sequential(
                nn.Linear(vit.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            outputs = self.vit(pixel_values=x)  # Extract features from ViT
            pooled_output = outputs.last_hidden_state[:, 0]  # CLS token output
            logits = self.classifier(pooled_output)
            
            if self.multi_label:
                return torch.sigmoid(logits)  # Sigmoid activation for multi-label classification
            return logits  # Raw logits for CrossEntropyLoss
    
    return ViTClassifier(vit, num_classes, multi_label)

