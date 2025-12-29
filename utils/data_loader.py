import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import random

# Handle both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image_generation import create_two_channel_image
else:
    from utils.image_generation import create_two_channel_image


class MalwareImageDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        mode: str = 'two_channel',  # 'bigram_dct' or 'two_channel'
        max_samples: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.samples = []  # List of (file_path, label)
        
        
        self.create_two_channel_image = create_two_channel_image
        
        self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: Optional[int]):        
        malware_dir = os.path.join(self.data_dir, 'malware')
        benign_dir = os.path.join(self.data_dir, 'benign')
        
        # Load malware samples (label = 1)
        if os.path.exists(malware_dir):
            for filename in os.listdir(malware_dir):
                file_path = os.path.join(malware_dir, filename)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, 1))
        
        # Load benign samples (label = 0)
        if os.path.exists(benign_dir):
            for filename in os.listdir(benign_dir):
                file_path = os.path.join(benign_dir, filename)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, 0))
        
        random.shuffle(self.samples)
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples...")
        malware_count = sum(1 for _, label in self.samples if label == 1)
        benign_count = len(self.samples) - malware_count
        print(f"Malware: {malware_count}, Benign: {benign_count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        
        try:
            if self.mode == 'two_channel':
                # 2-way XOR: byteplot XOR bigram-DCT
                image = self.create_two_channel_image(file_path)
                image = np.expand_dims(image, axis=0)  # (1, H, W)
            
            elif self.mode == 'three_way_xor':
                # ResNet: 3 separate channels (sparse, DCT, byteplot)
                from utils.image_generation import create_three_channel_image
                image = create_three_channel_image(file_path)  # (3, H, W)
            
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            image_tensor = torch.from_numpy(image).float()
            return image_tensor, label
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return appropriate zero tensor based on mode
            channels = 3 if self.mode == 'three_way_xor' else 1
            image_tensor = torch.zeros((channels, 256, 256), dtype=torch.float32)
            return image_tensor, label


def create_data_loaders(
    data_dir: str,
    mode: str = 'two_channel',
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    full_dataset = MalwareImageDataset(data_dir, mode=mode, max_samples=max_samples)
    
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset splits:")
    print(f"\tTrain: {train_size} ({train_split*100:.0f}%)")
    print(f"\tVal:   {val_size} ({val_split*100:.0f}%)")
    print(f"\tTest:  {test_size} ({test_split*100:.0f}%)")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor
        if torch.cuda.is_available():
            loader_kwargs['pin_memory_device'] = 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_dir = "./data"
    
    print("\nTesting Pipeline 2 (Two-Channel)...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir,
        mode='two_channel',
        batch_size=8,
        max_samples=100
    )
    
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
