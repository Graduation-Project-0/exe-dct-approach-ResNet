# Malware Detection Using ResNet and DCT bigram Visualization

A deep architecture for malware detection using frequency domain image visualization and ResNet models (18 and 50). **Optimized architecture for H100 GPU.**

## Approach

This project implements a 3-channel image visualization technique for executable files:

- **Channel 0**: Sparse bigram frequency image
- **Channel 1**: DCT-transformed bigram image
- **Channel 2**: Byteplot image (raw byte visualization)

These three channels are stacked into a single tensor for classification using ResNet architectures.

## H100 GPU Optimizations

This codebase is optimized for NVIDIA H100 80GB HBM3:

- ✅ **TF32 Precision**: Enabled for faster matrix multiplications
- ✅ **BFloat16 Mixed Precision**: Automatic mixed precision training and inference
- ✅ **torch.compile()**: JIT compilation for model optimization
- ✅ **Large Batch Sizes**: Default 1024 to utilize 80GB+ memory
- ✅ **Optimized Data Loading**: 16 workers, prefetching, persistent workers
- ✅ **cuDNN Benchmark**: Automatic convolution algorithm selection
- ✅ **Non-blocking Transfers**: Asynchronous CPU-GPU data transfers

### Check GPU Status

```bash
python utils/gpu_info.py
```

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

The training process is configured in `config.py`. By default, it uses ResNet-50.

Run training:

```bash
python main.py
```

### Prediction

Use the dedicated ResNet prediction script to analyze a single file. Edit the `INPUT_FILE_PATH` and `CHECKPOINT_PATH` variables at the top of the script:

```python
# resnet_predict.py
INPUT_FILE_PATH = 'data/benign/ab.exe'
CHECKPOINT_PATH = 'checkpoints/resnet_best.pth'
```

Then run:

```bash
python resnet_predict.py
```

## Configuration Options

| Parameter         | Default    | Description                 |
| ----------------- | ---------- | --------------------------- |
| `resnet_variant`  | `resnet50` | ResNet-18 or ResNet-50      |
| `pretrained`      | `True`     | Use ImageNet weights        |
| `freeze_backbone` | `False`    | Freeze backbone layers      |
| `DATA_DIR`        | `./data`   | Dataset directory           |
| `EPOCHS`          | `25`       | Training epochs             |
| `BATCH_SIZE`      | `1024`     | Batch size (H100 optimized) |
| `LEARNING_RATE`   | `0.001`    | Learning rate               |
| `num_workers`     | `16`       | Data loading workers (H100) |

## Output

- `checkpoints/resnet_best.pth` - Trained model
- `results/resnet_training_history.png` - Training curves
- `results/resnet_roc_curve.png` - ROC curve
- `results/resnet_confusion_matrix.png` - Confusion matrix
