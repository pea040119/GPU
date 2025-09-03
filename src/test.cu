/*
File:           src/test.cu
Created:        2025-08-27
Updated:        2025-08-27 by PEA
Description:    Demonstration of simple CUDA program.
*/



#include <iostream>
#include <cuda_runtime.h>



void check_device() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Detected CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name 
                  << " (Compute Capability " << prop.major << "." << prop.minor << ")" 
                  << std::endl;
    }
}

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N) {
        // std::cout << A[i] << B[i] << C[i] << std::endl;
        C[i] = A[i] + B[i];
    }
}


int main() {
    check_device();

    int N = 256;
    size_t size = N * sizeof(float);
    
    int threads_per_block = 256;
    int blocks_per_grid = (N+threads_per_block - 1) / threads_per_block;

    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    float *d_A, *d_B, *d_C;

    for (int i=0; i<N; i++) {
        h_A[i] = i;
        h_B[i] = i * 0.1f;
        h_C[i] = 0;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++) {
        std::cout << "[*]" << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
