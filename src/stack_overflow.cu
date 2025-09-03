/*
File:           src/stack_overflow.cu
Created:        2025-08-27
Updated:        2025-08-29 by PEA
Description:    Demonstration of stack overflow vulnerability in CUDA kernel.
*/



#include <iostream>
#include <cuda_runtime.h>

#define BUF_LEN 16



void check_device() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "[+] " << "Detected CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "[+] " << "Device " << i << ": " << prop.name 
                  << " (Compute Capability " << prop.major << "." << prop.minor << ")" 
                  << std::endl;
    }
    std::cout << std::endl;
}


typedef unsigned long(*dummy_ptr_l) (void);

__device__ __noinline__ unsigned long dummy_1() {
    return 0x1111111111111111;
}

__device__ __noinline__ unsigned long dummy_2() {
    return 0x2222222222222222;
}

__device__ __noinline__ unsigned long dummy_3() {
    return 0x3333333333333333;
}

__device__ __noinline__ unsigned long dummy_4() {
    return 0x4444444444444444;
}

__device__ __noinline__ unsigned long dummy_5() {
    return 0x5555555555555555;
}

__device__ __noinline__ unsigned long dummy_6() {
    return 0x6666666666666666;
}

__device__ __noinline__ unsigned long dummy_7() {
    return 0x7777777777777777;
}

__device__ __noinline__ unsigned long dummy_8() {
    return 0x8888888888888888;
}

__device__ __noinline__ unsigned long dummy_9() {
    printf("[*] Hello Admin!\n");
    return 0x9999999999999999;
}


__device__ __noinline__ unsigned long unsafe_hash(unsigned int *input, int len) {
    unsigned int buf[BUF_LEN];

    dummy_ptr_l fp[8];
    fp[0] = dummy_1;
    fp[1] = dummy_2;
    fp[2] = dummy_3;
    fp[3] = dummy_4;
    fp[4] = dummy_5;
    fp[5] = dummy_6;
    fp[6] = dummy_7;
    fp[7] = dummy_8;

    unsigned long hash = 5381;

    for (int i=0; i<len; i++) 
        buf[i] = input[i];

    for (int i=0; i<BUF_LEN; i++)
        hash = ((hash << 5) + hash) + buf[i];
    
    return (unsigned long)(fp[hash % 8])();
}


__global__ void test_kernel(unsigned long *hashes, unsigned int *inputs, int len, int *admin) {
    unsigned long hash;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (*admin)
        hash = dummy_9();
    else
        hash = unsafe_hash(&inputs[idx * len], len);
    printf("[*] dummy_9 address: %p\n", (void*)dummy_9);
    hashes[idx] = hash;
}



int main() {
    check_device();

    const int num_blocks = 1;
    const int num_threads = 1;
    const int total_threads = num_threads * num_blocks;

    unsigned long h_hashes[total_threads];
    int h_admin = 0;

    const int input_len = 27;
    unsigned int h_inputs[input_len];
    for (int i=0; i<input_len; i++)
        h_inputs[i] = 0xe10;

    unsigned long *d_hashes;
    unsigned int *d_inputs;
    int *d_admin; 

    cudaMalloc((void**)&d_hashes, total_threads * sizeof(unsigned long));
    cudaMalloc((void**)&d_inputs, input_len * sizeof(unsigned int));
    cudaMalloc((void**)&d_admin, sizeof(int));

    cudaMemcpy(d_inputs, h_inputs, input_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_admin, &h_admin, sizeof(int), cudaMemcpyHostToDevice);

    test_kernel<<<num_blocks, num_threads>>>(d_hashes, d_inputs, input_len, d_admin);

    cudaMemcpy(h_hashes, d_hashes, total_threads * sizeof(unsigned long), cudaMemcpyDeviceToHost);

    for (int i=0; i<total_threads; i++)
        std::cout << "[*] "<< "Thread " << i << ": Hash = 0x" << std::hex << h_hashes[i] << std::dec << std::endl;

    cudaFree(d_hashes);
    cudaFree(d_inputs);
    cudaFree(d_admin);

    return 0;
}


