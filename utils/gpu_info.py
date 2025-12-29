import torch
import sys

def print_separator(char='=', length=70):
    print(char * length)

def check_gpu_info():
    print_separator()
    print("GPU INFORMATION & H100 OPTIMIZATION CHECK")
    print_separator()
    
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("No CUDA-capable GPU detected!")
        return
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    print_separator('-')
    print("GPU DEVICES")
    print_separator('-')
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}\n")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processors: {props.multi_processor_count}")
        print(f"  Max Threads per MP: {props.max_threads_per_multi_processor}")
        print()
    
    current_device = torch.cuda.current_device()
    print(f"Current Device: {current_device} ({torch.cuda.get_device_name(current_device)})")
    
    print_separator('-')
    print("MEMORY STATUS")
    print_separator('-')
    
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Free:      {total - reserved:.2f} GB")
    print(f"Total:     {total:.2f} GB")
    
    print_separator('-')
    print("H100 OPTIMIZATION SETTINGS")
    print_separator('-')
    
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    print(f"TF32 for matmul: {tf32_matmul}")
    print(f"TF32 for cuDNN:  {tf32_cudnn}")
    
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    print(f"cuDNN Enabled:       {cudnn_enabled}")
    print(f"cuDNN Benchmark:     {cudnn_benchmark}")
    print(f"cuDNN Deterministic: {cudnn_deterministic}")
    
    print(f"Float32 Matmul Precision: {torch.get_float32_matmul_precision()}")
    
    print(f"AMP (BFloat16) Supported: {torch.cuda.is_bf16_supported()}")
    
    print_separator('-')
    print("PERFORMANCE TEST")
    print_separator('-')
    
    size = 4096
    print(f"Matrix multiplication test ({size}x{size})...")
    
    # FP32
    torch.cuda.synchronize()
    import time
    start = time.time()
    a = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, size, device='cuda', dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    print(f"  FP32 time: {fp32_time*1000:.2f} ms")
    
    # BF16
    if torch.cuda.is_bf16_supported():
        torch.cuda.synchronize()
        start = time.time()
        a = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        bf16_time = time.time() - start
        print(f"  BF16 time: {bf16_time*1000:.2f} ms")
        print(f"  Speedup:   {fp32_time/bf16_time:.2f}x")
    
    print_separator()
    print("RECOMMENDATIONS FOR H100")
    print_separator()
    
    recommendations = []
    
    if not tf32_matmul:
        recommendations.append("❌ Enable TF32: torch.set_float32_matmul_precision('high')")
    else:
        recommendations.append("✅ TF32 enabled")
    
    if not cudnn_benchmark:
        recommendations.append("❌ Enable cuDNN benchmark: torch.backends.cudnn.benchmark = True")
    else:
        recommendations.append("✅ cuDNN benchmark enabled")
    
    if torch.cuda.is_bf16_supported():
        recommendations.append("✅ Use BFloat16 mixed precision training")
    
    recommendations.append("✅ Use torch.compile() for model optimization")
    recommendations.append("✅ Use large batch sizes (1024+) to utilize 80GB memory")
    recommendations.append("✅ Use multiple data loading workers (16+)")
    recommendations.append("✅ Enable persistent_workers and prefetch_factor in DataLoader")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print_separator()

if __name__ == "__main__":
    check_gpu_info()
