/*
 * ======================================================================================
 * How to Run on Google Colab:
 * ======================================================================================
 * 1. Create a new Notebook on Google Colab (https://colab.research.google.com/).
 * 2. Change Runtime Type:
 *    - Click "Runtime" -> "Change runtime type"
 *    - Select "T4 GPU" (or any other available GPU) under "Hardware accelerator".
 * 3. Copy-paste this entire file content into a code cell.
 * 4. Add the magic command `%%writefile vector_add.cu` at the very top of the cell.
 *    This saves the code to a file named `vector_add.cu` in the Colab environment.
 * 5. Run that cell.
 * 6. Create a new code cell below and run the compilation and execution commands:
 *    
 *    !nvcc -arch=sm_75 vector_add.cu -o vector_add
 *    !./vector_add
 * 
 *    (Note: -arch=sm_75 targets the T4 GPU commonly found in Colab. 
 *     If you have a different GPU, use -arch=native or check your GPU's compute capability)
 * 
 * ======================================================================================
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Write a GPU program that performs element-wise addition of two vectors containing 32-bit floating point numbers. The program should take two input vectors of equal length and produce a single output vector containing their sum.

// Implementation Requirements
// External libraries are not permitted
// The solve function signature must remain unchanged
// The final result must be stored in vector C
// Input vectors A and B have identical lengths
// 1 ≤ N ≤ 100,000,000
// Optimization:
// 1. Use float4 for vectorized memory access (128-bit loads/stores).
// 2. Use Grid-Stride Loop to handle arbitrary N and allow flexible grid sizing.
// 3. Use __restrict__ to promise no pointer aliasing, enabling aggressive compiler optimizations.
// 4. Handle tail elements within the same kernel to avoid a second kernel launch overhead.

__global__ void vector_add_optimized(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Align to float4 boundary
    int n_vec = N / 4;
    
    // Reinterpret pointers as float4
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    // Grid-Stride Loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_vec; i += stride) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }

    // Handle remaining elements (Tail)
    // Only the very first thread handles the few remaining items (0-3 items).
    // This avoids launching a separate kernel.
    if (idx == 0) {
        for (int i = n_vec * 4; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // Tunable parameters
    int threadsPerBlock = 256;
    
    // Calculate Grid Size
    // For bandwidth-bound kernels, we want enough waves to hide latency, 
    // but not so many that we incur scheduling overhead.
    // A heuristic is typically 32 * number_of_SMs. 
    // Since we don't know the SM count statically, we can pick a reasonable fixed number 
    // or calculate based on N.
    // For very large N, a fixed large grid size with Grid-Stride Loop is better.
    
    int numBlocks = 32 * 40; // Assume ~40 SMs (e.g., T4), 32 blocks per SM. 
                             // 1280 blocks * 256 threads = ~320k threads resident.
    
    // Clamp numBlocks so we don't launch too many for small N
    int n_vec = N / 4;
    if (n_vec == 0) n_vec = 1; // Avoid division by zero or empty grid for N < 4
    int neededBlocks = (n_vec + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks > neededBlocks) {
        numBlocks = neededBlocks;
    }
    
    // Launch optimized kernel
    vector_add_optimized<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    // 1. Prepare data (N = 10 for simple testing)
    int N = 10;
    size_t size = N * sizeof(float);

    // Host memory
    std::vector<float> h_A(N, 1.0f); // [1, 1, ..., 1]
    std::vector<float> h_B(N, 2.0f); // [2, 2, ..., 2]
    std::vector<float> h_C(N, 0.0f); // Output

    // Device memory pointers
    float *d_A, *d_B, *d_C;

    // 2. Allocate memory on GPU
    checkCudaError(cudaMalloc((void**)&d_A, size), "Alloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "Alloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "Alloc d_C");

    // 3. Copy data from Host to Device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Copy A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Copy B");

    // 4. Run the kernel
    std::cout << "Running vector_add on GPU..." << std::endl;
    solve(d_A, d_B, d_C, N);

    // 5. Copy result back from Device to Host
    checkCudaError(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost), "Copy C");

    // 6. Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (std::abs(h_C[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << h_C[i] << " != " << expected << std::endl;
            success = false;
            break;
        }
        std::cout << "Index " << i << ": " << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    if (success) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    // 7. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
