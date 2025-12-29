import torch
import numpy as np
import sys
import os

FILE_PATH = r"data/benign/ab.exe" 
CHECKPOINT_PATH = "checkpoints/pipeline2_best.pth"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cnn_models import C3C2D_TwoChannel
from utils.image_generation import create_two_channel_image

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found: {FILE_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = C3C2D_TwoChannel()
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    file_size = os.path.getsize(FILE_PATH)
    print(f"File Size: {file_size} bytes ({file_size/1024:.2f} KB)")

    print(f"Processing {FILE_PATH}...")
    try:
        image = create_two_channel_image(FILE_PATH)
        print(f"Generated Image Shape: {image.shape}")
        
        # Add dimensions: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        print(f"Input Tensor Shape: {tensor.shape}")
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return

    print("Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(tensor)
        
        # For 2-class output: output shape is (1, 2)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        
        # Class 0 = Benign, Class 1 = Malware
        benign_prob = probs[0, 0].item()
        malware_prob = probs[0, 1].item()

    label = "MALWARE" if predicted_class == 1 else "BENIGN"
    confidence = malware_prob if label == "MALWARE" else benign_prob

    print(f"File:        {os.path.basename(FILE_PATH)}")
    print(f"Prediction:  {label}")
    print(f"Confidence:  {confidence*100:.2f}%")
    print(f"Benign Prob: {benign_prob:.6f}")
    print(f"Malware Prob: {malware_prob:.6f}")

if __name__ == "__main__":
    main()
