import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

class ResNet50TaggingNN(nn.Module):
    """Heavy but highly accurate architecture"""
    def __init__(self, num_classes=25, use_pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # ResNet's final layer is called 'fc'
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class MobileNetV3TaggingNN(nn.Module):
    """Very fast, lightweight architecture - ideal for moving robots"""
    def __init__(self, num_classes=25, use_pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)
        
        # MobileNetV3's final layer is inside 'classifier'
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB0TaggingNN(nn.Module):
    """Great balance between speed and accuracy"""
    def __init__(self, num_classes=25, use_pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # EfficientNet's final layer is inside 'classifier'
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Quick test
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224) # Batch of 2 images
    
    # You can instantiate any of the models here
    model = EfficientNetB0TaggingNN(num_classes=25)
    
    output = model(dummy_input)
    print(f"Model output shape: {output.shape} (Should be [2, 25])")



class CustomTaggingNN(nn.Module):
    """
    A custom CNN built entirely from scratch for multi-label tagging.
    This doesn't use any pre-trained weights, so it will take longer to train.
    """
    def __init__(self, num_classes=25):
        super().__init__()
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 112 x 112
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 56 x 56
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 28 x 28
            
            # Block 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 256 x 14 x 14
            
            # Global Average Pooling (flattens the 14x14 spatial dimensions into 1x1)
            nn.AdaptiveAvgPool2d((1, 1)) # Output: 256 x 1 x 1
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout helps prevent overfitting since we are training from scratch
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # We DO NOT apply Sigmoid here. BCEWithLogitsLoss does it for us in train.py!
        return x
