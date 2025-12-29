# Malware Detection Using DCT-Based Image Visualization

A deep learning approach for malware detection using frequency domain image visualization with support for both shallow CNN and ResNet architectures.

## Models

| Model     | Input Method | Input Shape   | Parameters | Accuracy |
| --------- | ------------ | ------------- | ---------- | -------- |
| 3C2D      | 2-way XOR    | (1, 256, 256) | ~17M       | ~98%     |
| ResNet-18 | 3-channel    | (3, 256, 256) | ~11M       | ~88%     |
| ResNet-50 | 3-channel    | (3, 256, 256) | ~23M       | ~89%     |

**XOR Methods**:

- **2-way XOR**: byteplot ⊕ bigram-DCT → single channel
- **3-channel**: [sparse bigram, DCT bigram, byteplot] → 3 separate channels

## Setup

```bash
pip install -r requirements.txt
```

## Data Structure

```
data/
├── malware/    # Malicious executables
└── benign/     # Clean executables
```

## Usage

### Training

Edit `main.py` to select model:

```python
# Choose model: '3c2d' or 'resnet'
MODEL_TYPE = 'resnet'  # or '3c2d'

# Optional: Override ResNet variant
# config['model']['resnet_variant'] = 'resnet50'  # or 'resnet18'
```

Run training:

```bash
python main.py
```

### Configuration Options

| Parameter         | Default    | Description            |
| ----------------- | ---------- | ---------------------- |
| `MODEL_TYPE`      | `resnet`   | Model architecture     |
| `resnet_variant`  | `resnet18` | ResNet-18 or ResNet-50 |
| `pretrained`      | `True`     | Use ImageNet weights   |
| `freeze_backbone` | `False`    | Freeze backbone layers |
| `DATA_DIR`        | `./data`   | Dataset directory      |
| `EPOCHS`          | `50`       | Training epochs        |
| `BATCH_SIZE`      | `1024`     | Batch size             |
| `LEARNING_RATE`   | `0.001`    | Learning rate          |
| `DEVICE`          | `auto`     | auto, cpu, cuda        |

## Output

- `checkpoints/{model_type}_best.pth` - Trained model
- `results/{model_type}_training_history.png` - Training curves
- `results/{model_type}_roc_curve.png` - ROC curve
- `results/{model_type}_confusion_matrix.png` - Confusion matrix

## Requirements

- Python 3.7+
- PyTorch >= 1.12.0
- torchvision
- NumPy, SciPy, scikit-learn, matplotlib
