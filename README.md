# Malware Detection Using DCT-Based Image Visualization

A deep learning approach for malware detection using frequency domain image visualization with support for both shallow CNN and ResNet architectures. **Optimized for NVIDIA H100 80GB HBM3 GPU.**

## Models

| Model     | Input Method | Input Shape   | Parameters | Accuracy |
| --------- | ------------ | ------------- | ---------- | -------- |
| 3C2D      | 2-way XOR    | (1, 256, 256) | ~17M       | ~98%     |
| ResNet-18 | 3-channel    | (3, 256, 256) | ~11M       | ~88%     |
| ResNet-50 | 3-channel    | (3, 256, 256) | ~23M       | ~89%     |

**XOR Methods**:

- **2-way XOR**: byteplot ⊕ bigram-DCT → single channel
- **3-channel**: [sparse bigram, DCT bigram, byteplot] → 3 separate channels

## H100 GPU Optimizations

This codebase is optimized for NVIDIA H100 80GB HBM3:

- ✅ **TF32 Precision**: Enabled for faster matrix multiplications
- ✅ **BFloat16 Mixed Precision**: Automatic mixed precision training
- ✅ **torch.compile()**: JIT compilation for model optimization
- ✅ **Large Batch Sizes**: Default 1024 (up from 128) to utilize 85GB memory
- ✅ **Optimized Data Loading**: 16 workers, prefetching, persistent workers
- ✅ **cuDNN Benchmark**: Automatic convolution algorithm selection
- ✅ **Non-blocking Transfers**: Asynchronous CPU-GPU data transfers
- ✅ **Gradient Scaler**: Proper mixed precision training with loss scaling

### Check GPU Status

```bash
python utils/gpu_info.py
```

This will display:

- GPU device information and memory
- Current optimization settings
- Performance benchmarks
- Recommendations for H100

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

### Prediction

Predict a single file:

```bash
# Auto-detect model type from checkpoint
python simple_predict.py --file path/to/executable.exe --checkpoint checkpoints/resnet_best.pth

# Specify model type explicitly
python simple_predict.py --file data/benign/ab.exe --checkpoint checkpoints/3c2d_best.pth --model-type 3c2d
```

### Configuration Options

| Parameter                     | Default    | Description                           |
| ----------------------------- | ---------- | ------------------------------------- |
| `MODEL_TYPE`                  | `resnet`   | Model architecture                    |
| `resnet_variant`              | `resnet50` | ResNet-18 or ResNet-50                |
| `pretrained`                  | `True`     | Use ImageNet weights                  |
| `freeze_backbone`             | `False`    | Freeze backbone layers                |
| `DATA_DIR`                    | `./data`   | Dataset directory                     |
| `EPOCHS`                      | `25`       | Training epochs                       |
| `BATCH_SIZE`                  | `1024`     | Batch size (H100 optimized)           |
| `LEARNING_RATE`               | `0.001`    | Learning rate                         |
| `DEVICE`                      | `auto`     | auto, cpu, cuda                       |
| `num_workers`                 | `16`       | Data loading workers (H100)           |
| `prefetch_factor`             | `4`        | Batches to prefetch per worker        |
| `persistent_workers`          | `True`     | Keep workers alive between epochs     |
| `gradient_accumulation_steps` | `1`        | For even larger effective batch sizes |

## Output

- `checkpoints/{model_type}_best.pth` - Trained model
- `results/{model_type}_training_history.png` - Training curves
- `results/{model_type}_roc_curve.png` - ROC curve
- `results/{model_type}_confusion_matrix.png` - Confusion matrix

## Requirements

- Python 3.7+
- PyTorch >= 2.0.0 (for torch.compile and BFloat16)
- torchvision
- NumPy, SciPy, scikit-learn, matplotlib, tqdm
- NVIDIA GPU with CUDA support (H100 recommended)

## Performance Tips

For maximum H100 performance:

1. **Increase batch size** based on your dataset size and memory
2. **Adjust num_workers** (try 16-32 for H100)
3. **Use gradient accumulation** for effective batch sizes > 1024
4. **Monitor GPU utilization** with `nvidia-smi -l 1`
5. **Profile your code** to identify bottlenecks
