import torch
import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cnn_models import C3C2D_TwoChannel
from models.resnet_models import ResNetMalwareDetector
from utils.image_generation import create_two_channel_image, create_three_channel_image

def detect_model_type(checkpoint_path):
    """Detect model type from checkpoint filename"""
    basename = os.path.basename(checkpoint_path).lower()
    if 'resnet' in basename:
        return 'resnet'
    elif '3c2d' in basename or 'pipeline2' in basename:
        return '3c2d'
    else:
        # Try to load and inspect
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Check for ResNet-specific layers
        if any('layer1' in k or 'layer2' in k for k in state_dict.keys()):
            return 'resnet'
        else:
            return '3c2d'

def load_model(checkpoint_path, device, model_type=None):
    """Load model from checkpoint"""
    if model_type is None:
        model_type = detect_model_type(checkpoint_path)
    
    print(f"Loading {model_type.upper()} model from {checkpoint_path}...")
    
    if model_type == '3c2d':
        model = C3C2D_TwoChannel()
    elif model_type == 'resnet':
        # Try to detect ResNet variant from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect variant by checking fc layer size
        fc_key = 'backbone.fc.weight' if 'backbone.fc.weight' in state_dict else 'fc.weight'
        if fc_key in state_dict:
            fc_in_features = state_dict[fc_key].shape[1]
            variant = 'resnet50' if fc_in_features == 2048 else 'resnet18'
        else:
            variant = 'resnet18'  # default
        
        print(f"Detected ResNet variant: {variant}")
        model = ResNetMalwareDetector(model_name=variant, num_classes=2, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Remove '_orig_mod.' prefix from torch.compile
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, model_type

def preprocess_file(file_path, model_type):
    """Preprocess file based on model type"""
    if model_type == '3c2d':
        # 2-way XOR: single channel
        image = create_two_channel_image(file_path)
        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    elif model_type == 'resnet':
        # 3-channel: separate channels
        image = create_three_channel_image(file_path)
        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        tensor = torch.from_numpy(image).float().unsqueeze(0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return tensor

def predict(model, tensor, device):
    """Run inference and return prediction"""
    tensor = tensor.to(device)
    
    with torch.no_grad():
        # H100 optimization: Use BFloat16 for inference
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                           dtype=torch.bfloat16,
                           enabled=torch.cuda.is_available()):
            output = model(tensor)
        
        # For 2-class output: output shape is (1, 2)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        
        # Class 0 = Benign, Class 1 = Malware
        benign_prob = probs[0, 0].item()
        malware_prob = probs[0, 1].item()
    
    return predicted_class, benign_prob, malware_prob

def main():
    parser = argparse.ArgumentParser(description='Malware Detection Prediction')
    parser.add_argument('--file', type=str, default='data/benign/ab.exe',
                       help='Path to executable file to analyze')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/resnet_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['3c2d', 'resnet'], default=None,
                       help='Model type (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model
    model, model_type = load_model(args.checkpoint, device, args.model_type)
    
    # File info
    file_size = os.path.getsize(args.file)
    print(f"File: {args.file}")
    print(f"Size: {file_size} bytes ({file_size/1024:.2f} KB)")
    print()
    
    # Preprocess
    print(f"Processing with {model_type.upper()} model...")
    try:
        tensor = preprocess_file(args.file, model_type)
        print(f"Input tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return
    
    # Predict
    print("Running inference...")
    predicted_class, benign_prob, malware_prob = predict(model, tensor, device)
    
    label = "MALWARE" if predicted_class == 1 else "BENIGN"
    confidence = malware_prob if label == "MALWARE" else benign_prob
    
    print("\n" + "="*50)
    print(f"File:        {os.path.basename(args.file)}")
    print(f"Prediction:  {label}")
    print(f"Confidence:  {confidence*100:.2f}%")
    print(f"Benign Prob: {benign_prob:.6f}")
    print(f"Malware Prob: {malware_prob:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
