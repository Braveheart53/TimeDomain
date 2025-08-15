
import numpy as np
import multiprocessing as mp
from scipy.fft import fft
import time

def fft_worker(args):
    """Worker function for parallel FFT computation"""
    data_chunk, chunk_id = args
    # Compute FFT of chunk
    result = fft(data_chunk)
    return chunk_id, result

def multiprocessing_fft(data, num_processes=None):
    """
    Parallel FFT using multiprocessing
    Splits data into chunks and processes in parallel
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    N = len(data)
    chunk_size = N // num_processes

    # Split data into chunks
    chunks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        if i == num_processes - 1:
            end_idx = N  # Last chunk gets remainder
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append((data[start_idx:end_idx], i))

    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        results = pool.map(fft_worker, chunks)

    # Reconstruct result
    results.sort(key=lambda x: x[0])  # Sort by chunk_id
    fft_result = np.concatenate([result[1] for result in results])

    return fft_result

def benchmark_parallel_transforms():
    """Benchmark parallel implementations"""
    N = 2**16  # 64K points

    # Generate test data
    x = np.random.random(N) + 1j * np.random.random(N)

    # Sequential FFT
    start_time = time.time()
    fft_seq = fft(x)
    seq_time = time.time() - start_time

    # Parallel FFT
    start_time = time.time()
    fft_par = multiprocessing_fft(x, num_processes=4)
    par_time = time.time() - start_time

    print(f"Sequential FFT time: {seq_time:.4f} seconds")
    print(f"Parallel FFT time: {par_time:.4f} seconds")
    print(f"Speedup: {seq_time/par_time:.2f}x")

    # Verify results are close
    error = np.mean(np.abs(fft_seq - fft_par))
    print(f"Mean error: {error:.2e}")

if __name__ == "__main__":
    benchmark_parallel_transforms()
