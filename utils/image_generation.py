import numpy as np
from scipy.fft import dctn
from typing import Tuple
import math
from scipy.ndimage import zoom

def read_binary_file(file_path: str) -> bytes:
    with open(file_path, 'rb') as f:
        return f.read()


def extract_bigrams(byte_data: bytes) -> np.ndarray:
    bigram_freq = np.zeros(65536, dtype=np.float64)
    
    for i in range(len(byte_data) - 1):
        # Combine two consecutive bytes into a single bigram value
        bigram = (byte_data[i] << 8) | byte_data[i + 1]
        bigram_freq[bigram] += 1
    
    return bigram_freq


def create_bigram_image(bigram_freq: np.ndarray, zero_out_0000: bool = True) -> np.ndarray:
    # Zero out the bigram "0000" if specified (as mentioned in the paper)
    if zero_out_0000:
        bigram_freq[0] = 0
    
    total = np.sum(bigram_freq)
    if total > 0:
        bigram_freq = bigram_freq / total
    
    bigram_image = bigram_freq.reshape(256, 256)
    
    return bigram_image


def apply_2d_dct(image: np.ndarray) -> np.ndarray:
    dct_image = dctn(image, type=2, norm='ortho')
    
    dct_image = np.abs(dct_image)
    if np.max(dct_image) > 0:
        dct_image = dct_image / np.max(dct_image)
    
    return dct_image


# ---------------------------------------------------

def create_byteplot_image(file_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    byte_data = read_binary_file(file_path)
    return create_byteplot_from_bytes(byte_data, target_size)


def create_byteplot_from_bytes(byte_data: bytes, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    
    total_bytes = len(byte_array)
    side_length = int(math.sqrt(total_bytes))
    
    truncated_length = side_length * side_length
    byte_array = byte_array[:truncated_length]
    
    byteplot = byte_array.reshape(side_length, side_length)
    
    byteplot_resized = resize_image(byteplot, target_size)
    
    byteplot_resized = byteplot_resized.astype(np.float32) / 255.0
    
    return byteplot_resized


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    target_h, target_w = target_size
    
    zoom_factors = (target_h / h, target_w / w)
    resized = zoom(image, zoom_factors, order=1)
    
    return resized


def create_three_channel_image(file_path: str) -> np.ndarray:
    """
    Create 3-channel image for ResNet as per paper.
    Channel 0: Sparse bigram frequency image
    Channel 1: DCT-transformed bigram image
    Channel 2: Byteplot image
    Returns: (3, H, W) - three different images stacked
    """
    byte_data = read_binary_file(file_path)
    
    bigram_freq = extract_bigrams(byte_data)
    sparse_bigram = create_bigram_image(bigram_freq, zero_out_0000=True)
    
    dct_bigram = apply_2d_dct(sparse_bigram)
    
    byteplot = create_byteplot_from_bytes(byte_data, target_size=(256, 256))
    
    three_channel = np.stack([sparse_bigram, dct_bigram, byteplot], axis=0)
    
    return three_channel.astype(np.float32)

if __name__ == "__main__":
    import os
    # Find a sample file for testing if possible, otherwise use a dummy path
    exe_path = "data/benign/ab.exe" 
    
    if os.path.exists(exe_path):
        print("\nGenerating 3-channel ResNet image...")
        three_channel = create_three_channel_image(exe_path)
        print(f"Three-channel image shape: {three_channel.shape}")
    else:
        print(f"\nTest file not found: {exe_path}. Skipping image generation test.")


