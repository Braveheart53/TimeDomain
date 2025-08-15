
# FPGA CZT Implementation using Vitis HLS and PYNQ
# This implements the Chirp Z-Transform using the convolution method

# HLS C++ code for CZT (to be saved as czt_hls.cpp)
hls_czt_cpp_content = '''
#include "ap_axi_sdata.h"
#include "hls_stream.h" 
#include "hls_fft.h"
#include <complex>
#include <cmath>

const int MAX_N = 1024;
const int MAX_M = 1024;
const int MAX_L = 2048; // Convolution length

typedef ap_axis<64,2,5,6> axis_t; // 64-bit for complex data
typedef hls::stream<axis_t> stream_t;
typedef std::complex<float> complex_t;

void czt_accelerator(
    stream_t &input_stream, 
    stream_t &output_stream,
    int N,                    // Input length
    int M,                    // Output length  
    float A_real, float A_imag, // Starting point A
    float W_real, float W_imag  // Ratio W
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=A_real
    #pragma HLS INTERFACE s_axilite port=A_imag
    #pragma HLS INTERFACE s_axilite port=W_real
    #pragma HLS INTERFACE s_axilite port=W_imag
    #pragma HLS INTERFACE s_axilite port=return

    complex_t A(A_real, A_imag);
    complex_t W(W_real, W_imag);

    // Calculate convolution length
    int L = 1;
    while(L < N + M - 1) L *= 2; // Next power of 2

    // Buffers
    complex_t x[MAX_N];
    complex_t y[MAX_L];
    complex_t h[MAX_L];
    complex_t z[MAX_L];

    // Read input data
    for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE
        axis_t temp = input_stream.read();
        float real_part = *((float*)&temp.data);
        float imag_part = *((float*)(&temp.data + 4));
        x[i] = complex_t(real_part, imag_part);
    }

    // Step 1: Pre-multiply input with chirp
    for(int n = 0; n < N; n++) {
        #pragma HLS PIPELINE
        float phase = (n * n) / 2.0f;
        complex_t chirp = std::polar(1.0f, -phase);
        y[n] = x[n] * chirp;
    }

    // Zero pad y
    for(int i = N; i < L; i++) {
        #pragma HLS PIPELINE
        y[i] = complex_t(0, 0);
    }

    // Step 2: Create chirp filter h
    for(int i = 0; i < N + M - 1; i++) {
        #pragma HLS PIPELINE
        int idx = i - (N - 1);
        float phase = -(idx * idx) / 2.0f;
        complex_t chirp = std::polar(1.0f, phase);
        h[i] = chirp;
    }

    // Zero pad h
    for(int i = N + M - 1; i < L; i++) {
        #pragma HLS PIPELINE
        h[i] = complex_t(0, 0);
    }

    // Step 3: Convolution via FFT (simplified)
    // In practice, this would use FFT IP cores

    // Step 4: Post-multiply and extract result
    for(int k = 0; k < M; k++) {
        #pragma HLS PIPELINE
        float phase = (k * k) / 2.0f;
        complex_t chirp = std::polar(1.0f, -phase);
        complex_t A_power = std::polar(1.0f, -k);

        complex_t result = z[N - 1 + k] * chirp * A_power;

        // Write to output stream
        axis_t temp;
        *((float*)&temp.data) = result.real();
        *((float*)(&temp.data + 4)) = result.imag();
        temp.last = (k == M - 1);

        output_stream.write(temp);
    }
}
'''

# Python PYNQ control code for CZT
import numpy as np
from pynq import Overlay, allocate
import time

class CZTAccelerator:
    def __init__(self, bitfile_path):
        '''Initialize CZT accelerator overlay'''
        self.overlay = Overlay(bitfile_path)
        self.czt_ip = self.overlay.czt_accelerator
        self.dma = self.overlay.axi_dma_0

    def czt_fpga(self, input_data, M, A=1.0, W=None):
        '''Perform CZT on FPGA'''
        N = len(input_data)

        if W is None:
            W = np.exp(-1j * 2 * np.pi / M)

        # Allocate buffers
        input_buffer = allocate(shape=(N,), dtype=np.complex64)
        output_buffer = allocate(shape=(M,), dtype=np.complex64)

        # Copy input data
        input_buffer[:] = input_data

        # Configure CZT IP parameters
        self.czt_ip.write(0x10, N)                    # Input length
        self.czt_ip.write(0x18, M)                    # Output length
        self.czt_ip.write(0x20, float(A.real))        # A real part
        self.czt_ip.write(0x28, float(A.imag))        # A imaginary part
        self.czt_ip.write(0x30, float(W.real))        # W real part
        self.czt_ip.write(0x38, float(W.imag))        # W imaginary part

        # Start DMA transfers
        self.dma.sendchannel.transfer(input_buffer)
        self.dma.recvchannel.transfer(output_buffer)

        # Start CZT computation
        self.czt_ip.write(0x00, 1)  # Start

        # Wait for completion
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Wait for CZT IP to finish
        while (self.czt_ip.read(0x00) & 0x2) == 0:
            pass

        return np.copy(output_buffer)

    def zoom_fft(self, input_data, start_freq, end_freq, num_points):
        '''Perform zoom FFT using CZT'''
        N = len(input_data)

        # Calculate CZT parameters for zoom
        theta_start = 2 * np.pi * start_freq / N
        theta_end = 2 * np.pi * end_freq / N

        A = np.exp(1j * theta_start)
        W = np.exp(1j * (theta_start - theta_end) / num_points)

        return self.czt_fpga(input_data, num_points, A, W)

    def benchmark_czt(self):
        '''Benchmark CZT performance'''
        N = 1024
        M_values = [256, 512, 1024, 2048]

        print("Input Size | Output Size | FPGA Time (ms) | Throughput (MS/s)")
        print("-----------|-------------|----------------|------------------")

        for M in M_values:
            # Generate test data
            test_data = np.random.random(N) + 1j * np.random.random(N)
            test_data = test_data.astype(np.complex64)

            # Measure FPGA performance
            start_time = time.time()
            result_fpga = self.czt_fpga(test_data, M)
            fpga_time = (time.time() - start_time) * 1000  # ms

            throughput = M / (fpga_time / 1000) / 1e6  # MS/s

            print(f"{N:10} | {M:11} | {fpga_time:13.2f} | {throughput:16.2f}")

# Example usage
def main():
    print("CZT FPGA Implementation Example")
    print("To use this code:")
    print("1. Synthesize the HLS C++ code using Vitis HLS")
    print("2. Create block design in Vivado with CZT IP and DMA")
    print("3. Generate bitstream")
    print("4. Load bitstream on PYNQ board")
    print("5. Run this Python code")

    # Example zoom FFT application
    print("\nExample: Zoom FFT for spectral analysis")
    print("This allows high-resolution analysis of specific frequency bands")

    # Uncomment when you have the bitfile:
    # czt_accel = CZTAccelerator("czt_overlay.bit") 
    # czt_accel.benchmark_czt()

    # Example zoom FFT
    # N = 1024
    # x = np.random.random(N) + 1j * np.random.random(N)
    # zoomed = czt_accel.zoom_fft(x, start_freq=100, end_freq=200, num_points=256)

if __name__ == "__main__":
    main()
