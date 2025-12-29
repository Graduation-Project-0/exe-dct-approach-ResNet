import torch
import torch.nn as nn
from torchvision import models


class ResNetMalwareDetector(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=2, pretrained=True, freeze_backbone=False):
        super(ResNetMalwareDetector, self).__init__()
        
        if model_name == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            num_features = 512
        elif model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose 'resnet18' or 'resnet50'")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer for binary classification (2 classes: Benign & Malware)
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing ResNet-18...")
    model18 = ResNetMalwareDetector('resnet18', num_classes=2, pretrained=True)
    print(f"Total parameters: {count_parameters(model18):,}")
    
    print("\nTesting ResNet-50...")
    model50 = ResNetMalwareDetector('resnet50', num_classes=2, pretrained=True)
    print(f"Total parameters: {count_parameters(model50):,}")
    
    x = torch.randn(2, 3, 256, 256)
    y18 = model18(x)
    y50 = model50(x)
    print(f"\nInput shape: {x.shape}")
    print(f"ResNet-18 output: {y18.shape} (expected: [2, 2] for 2 classes)")
    print(f"ResNet-50 output: {y50.shape} (expected: [2, 2] for 2 classes)")
