import os
import torch
from pathlib import Path

# H100 Optimization: Enable TF32 for faster matrix multiplications
torch.set_float32_matmul_precision('high')

from config import get_config
from utils.data_loader import create_data_loaders
from models.cnn_models import C3C2D_TwoChannel
from models.resnet_models import ResNetMalwareDetector, count_parameters
from utils.training import (
    train_model, 
    test_model, 
    plot_training_history,
    plot_roc_curve,
    plot_confusion_matrix
)

def setup_directories(config):
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)


def get_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    return device


def create_model(config):
    """Factory function to create model based on config."""
    model_type = config['model_type']
    
    if model_type == '3c2d':
        model = C3C2D_TwoChannel()
    elif model_type == 'resnet':
        model = ResNetMalwareDetector(
            model_name=config['model']['resnet_variant'],
            num_classes=2,  
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def run_pipeline(config):
    model_type = config['model_type']
    data_mode = config['data']['mode']
    print(f"Model: {model_type.upper()}")
    print(f"Data mode: {data_mode}")
    print()
    
    device = get_device(config['training']['device'])
    setup_directories(config)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['data']['data_dir'],
        mode=config['data']['mode'],
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        max_samples=config['data']['max_samples'],
        num_workers=config['data']['num_workers']
    )
    
    print("\nInitializing model...")
    model = create_model(config)
    
    # H100 Optimization: Compile model
    print("Compiling model for H100...")
    model = torch.compile(model)
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    model_save_path = os.path.join(config['paths']['checkpoint_dir'], f'{model_type}_best.pth')
    
    # Training
    if not config.get('test_only', False):
        print("Training...")        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            device=device,
            save_path=model_save_path,
            patience=config['training']['patience']
        )
        
        if config['eval']['plot_training_history']:
            plot_path = os.path.join(config['paths']['output_dir'], f'{model_type}_training_history.png')
            plot_training_history(history, save_path=plot_path)
    
    print("Testing...")    
    metrics = test_model(model, test_loader, device)
    
    if config['eval']['plot_roc_curve']:
        roc_path = os.path.join(config['paths']['output_dir'], f'{model_type}_roc_curve.png')
        plot_roc_curve(metrics, save_path=roc_path)
        
    if config['eval']['plot_confusion_matrix']:
        cm_path = os.path.join(config['paths']['output_dir'], f'{model_type}_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    return metrics


def main():
    # Choose model: '3c2d' or 'resnet'
    MODEL_TYPE = 'resnet'  # Change to '3c2d' for shallow CNN
    
    config = get_config(model_type=MODEL_TYPE)
    
    # Optional: Override ResNet variant
    # config['model']['resnet_variant'] = 'resnet50'  # or 'resnet18'
    
    print("\nConfiguration:")
    print(f"\tModel type:     {config['model_type']}")
    if config['model_type'] == 'resnet':
        print(f"\tResNet variant: {config['model']['resnet_variant']}")
        print(f"\tPretrained:     {config['model']['pretrained']}")
    print(f"\tData mode:      {config['data']['mode']}")
    print(f"\tData directory: {config['data']['data_dir']}")
    print(f"\tBatch size:     {config['training']['batch_size']}")
    print(f"\tEpochs:         {config['training']['num_epochs']}")
    print(f"\tLearning rate:  {config['training']['learning_rate']}")
    print(f"\tDevice:         {config['training']['device']}")
    print()
    
    metrics = run_pipeline(config)
    
    print(f"\n{config['model_type'].upper()} Model Results:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"AUC:      {metrics.get('auc', 0):.4f}")
    print(f"Results saved to: {config['paths']['output_dir']}")
    print(f"Model saved to:   {config['paths']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
