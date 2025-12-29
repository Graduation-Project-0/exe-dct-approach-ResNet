import torch
import numpy as np
import sys
import os

INPUT_FILE_PATH = r'test_data/f81f710f5968fea399551a1fb7a13fad48b005f3c9ba2ea419d14b597401838c.exe'
CHECKPOINT_PATH = 'checkpoints/resnet_best.pth'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.resnet_models import ResNetMalwareDetector
from utils.image_generation import create_three_channel_image

def load_resnet_model(checkpoint_path, device):
    print(f"Loading ResNet model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    fc_key = 'backbone.fc.weight' if 'backbone.fc.weight' in state_dict else 'fc.weight'
    is_resnet50 = False
    
    if fc_key in state_dict:
        fc_in_features = state_dict[fc_key].shape[1]
        if fc_in_features == 2048:
            is_resnet50 = True
    
    if 'backbone.layer1.2.conv1.weight' in state_dict or 'layer1.2.conv1.weight' in state_dict:
        is_resnet50 = True
        
    variant = 'resnet50' if is_resnet50 else 'resnet18'
    
    print(f"Detected ResNet variant: {variant}")
    model = ResNetMalwareDetector(model_name=variant, num_classes=2, pretrained=False)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def preprocess_resnet(file_path):
    image = create_three_channel_image(file_path)
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    return tensor

def predict_resnet(model, tensor, device):
    tensor = tensor.to(device)
    
    with torch.no_grad():
        dtype = torch.float32
        if device.type == 'cuda':
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == 'cuda')):
            output = model(tensor)
        
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        
        benign_prob = probs[0, 0].item()
        malware_prob = probs[0, 1].item()
    
    return predicted_class, benign_prob, malware_prob

def main():
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"ERROR: Input file not found: {INPUT_FILE_PATH}")
        print("Please edit 'INPUT_FILE_PATH' at the top of this script.")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Please edit 'CHECKPOINT_PATH' at the top of this script.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        model = load_resnet_model(CHECKPOINT_PATH, device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    print(f"\nProcessing file: {INPUT_FILE_PATH}")
    try:
        tensor = preprocess_resnet(INPUT_FILE_PATH)
        print(f"Input tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"ERROR: Preprocessing failed: {e}")
        return

    print("Running inference...")
    try:
        predicted_class, benign_prob, malware_prob = predict_resnet(model, tensor, device)
        
        label = "MALWARE" if predicted_class == 1 else "BENIGN"
        confidence = malware_prob if label == "MALWARE" else benign_prob
        
        print(f"RESULT:     {label}")
        print(f"CONFIDENCE: {confidence*100:.2f}%")
        print(f"FILE:       {os.path.basename(INPUT_FILE_PATH)}")
        
        print(f"\nDetails:")
        print(f"- Benign Probability:  {benign_prob:.6f}")
        print(f"- Malware Probability: {malware_prob:.6f}")
        
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")

if __name__ == "__main__":
    main()
