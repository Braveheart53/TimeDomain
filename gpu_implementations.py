
import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
import time

def gpu_fft_implementation():
    """
    GPU-accelerated FFT using CuPy
    Demonstrates plan caching and multi-GPU support
    """
    # Setup
    N = 1024 * 1024  # 1M points

    # Generate test data on GPU
    x_gpu = cp.random.random(N).astype(cp.complex64)

    # Method 1: Direct FFT with automatic plan caching
    start_time = time.time()
    X_gpu = cp.fft.fft(x_gpu)
    cp.cuda.Device().synchronize()  # Wait for completion
    gpu_time1 = time.time() - start_time

    # Method 2: Using explicit plan for better control
    plan = cufft.get_fft_plan(x_gpu, value_type='C2C')
    start_time = time.time()
    with plan:
        X_gpu2 = cp.fft.fft(x_gpu)
    cp.cuda.Device().synchronize()
    gpu_time2 = time.time() - start_time

    # Method 3: Multi-GPU FFT (if multiple GPUs available)
    try:
        cp.fft.config.use_multi_gpus = True
        cp.fft.config.set_cufft_gpus([0, 1])  # Use GPU 0 & 1
        start_time = time.time()
        X_gpu_multi = cp.fft.fft(x_gpu)
        cp.cuda.Device().synchronize()
        gpu_multi_time = time.time() - start_time
        print(f"Multi-GPU FFT time: {gpu_multi_time:.4f} seconds")
    except:
        print("Multi-GPU not available")

    print(f"GPU FFT (auto plan): {gpu_time1:.4f} seconds")
    print(f"GPU FFT (explicit plan): {gpu_time2:.4f} seconds")

    # Show plan cache info
    cache = cp.fft.config.get_plan_cache()
    cache.show_info()

    return X_gpu

def gpu_czt_implementation():
    """
    GPU-accelerated CZT implementation using CuPy
    Note: CZT is not directly available in CuPy, so we implement via convolution
    """
    N = 1024
    M = 512  # Output points

    # Generate chirp parameters
    A = cp.exp(1j * 0.1)  # Starting point
    W = cp.exp(-1j * 2 * cp.pi / M)  # Step size

    # Input signal
    x = cp.random.random(N).astype(cp.complex64)

    # CZT via chirp convolution (Bluestein's algorithm)
    # Pre-multiply with chirp
    n = cp.arange(N)
    y = x * (W ** (n**2 / 2))

    # Convolution length
    L = 2**int(cp.ceil(cp.log2(N + M - 1)))

    # Chirp filter
    h = W ** (-(cp.arange(-(N-1), M)**2) / 2)

    # Zero-pad both sequences
    y_pad = cp.zeros(L, dtype=cp.complex64)
    y_pad[:N] = y
    h_pad = cp.zeros(L, dtype=cp.complex64)
    h_pad[:N+M-1] = h[:N+M-1]

    # Convolution via FFT
    Y = cp.fft.fft(y_pad)
    H = cp.fft.fft(h_pad)
    Z = Y * H
    z = cp.fft.ifft(Z)

    # Post-multiply and extract result
    k = cp.arange(M)
    X_czt = z[N-1:N-1+M] * (W ** (k**2 / 2)) * (A ** (-k))

    return X_czt

if __name__ == "__main__":
    # Run GPU implementations
    print("=== GPU FFT Implementation ===")
    result_fft = gpu_fft_implementation()

    print("\n=== GPU CZT Implementation ===")
    result_czt = gpu_czt_implementation()
    print(f"CZT computed {len(result_czt)} points")
