DATA_CONFIG = {
    'data_dir': './data',
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    'max_samples': None,
    'num_workers': 16,
    'prefetch_factor': 4,
    'persistent_workers': True,
}

MODEL_CONFIG = {
    'image_size': 256,
    
    # ResNet config
    'resnet_variant': 'resnet50',
    'pretrained': True,
    'freeze_backbone': False,  # If True, only train final layer
}

TRAINING_CONFIG = {
    'batch_size': 1024,
    'num_epochs': 25,
    'learning_rate': 0.001,
    'patience': 10,
    'device': 'auto',
    'gradient_accumulation_steps': 1,
    'use_amp': True,
}

IMAGE_CONFIG = {
    'zero_out_bigram_0000': True,
    'bigram_normalization': True,
    
    'dct_normalization': 'ortho',
    
    'byteplot_resize_method': 'bilinear',
}

PATH_CONFIG = {
    'checkpoint_dir': './checkpoints',
    'output_dir': './results',
    'log_dir': './logs',
}

EVAL_CONFIG = {
    'plot_training_history': True,
    'plot_roc_curve': True,
    'plot_confusion_matrix': True,
    'save_predictions': False,
    'prediction_threshold': 0.5,
}

LOG_CONFIG = {
    'log_level': 'INFO',
    'log_to_file': False,
    'log_file': './logs/training.log',
}


def get_config(model_type: str = 'resnet'):
    config = {
        'model_type': 'resnet',
        'data': DATA_CONFIG.copy(),
        'model': MODEL_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'image': IMAGE_CONFIG.copy(),
        'paths': PATH_CONFIG.copy(),
        'eval': EVAL_CONFIG.copy(),
        'log': LOG_CONFIG.copy(),
    }
    
    # ResNet setup
    config['data']['mode'] = 'three_way_xor'
    config['model']['input_channels'] = 3
    
    return config


def print_config(config):
    for section, params in config.items():
        if isinstance(params, dict):
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {params}")
    

if __name__ == "__main__":
    print("ResNet Model Configuration:")
    config = get_config(model_type='resnet')
    print_config(config)
