import torch
import sys

def test_cuda_connection():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print()
    
    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Number of GPUs:", torch.cuda.device_count())
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        print()
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
        # Test tensor operations on GPU
        try:
            print("\nTesting tensor operations on GPU...")
            device = torch.device("cuda")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print("✓ GPU tensor operations successful!")
            print(f"Result tensor shape: {z.shape}")
            print(f"Result tensor device: {z.device}")
            
            # Test moving tensor back to CPU
            z_cpu = z.cpu()
            print("✓ GPU to CPU transfer successful!")
            
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
    
    else:
        print("CUDA is not available. Using CPU only.")
        print("Possible reasons:")
        print("- No NVIDIA GPU installed")
        print("- CUDA drivers not installed")
        print("- PyTorch was installed without CUDA support")
        print("- GPU is not compatible with installed CUDA version")
    
    print("\nDevice that will be used:", "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    test_cuda_connection()