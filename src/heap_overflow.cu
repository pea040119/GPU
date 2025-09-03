/*
File:           src/heap_overflow.cu
Created:        2025-09-01
Updated:        2025-09-01 by PEA
Description:    Demonstration of heap overflow vulnerability in CUDA kernel.
*/



#include <iostream>
#include <cuda_runtime.h>

#define BUF_LEN 8



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



class B {
    public:
        __device__ virtual unsigned long func1(unsigned int hash) {return 0;}
        __device__ virtual unsigned long func2(unsigned int hash) {return 0;}
        __device__ virtual unsigned long func3(unsigned int hash) {return 0;}
        __device__ virtual unsigned long func4(unsigned int hash) {return 0;}
};


class D : public B {
    public:
        __device__ __noinline__ unsigned long func1(unsigned int hash) ;
        __device__ __noinline__ unsigned long func2(unsigned int hash) ;
        __device__ __noinline__ unsigned long func3(unsigned int hash) ;
        __device__ __noinline__ unsigned long func4(unsigned int hash) ;
};

__device__ __noinline__ unsigned long D::func1(unsigned int hash) { return 1*hash; }
__device__ __noinline__ unsigned long D::func2(unsigned int hash) { return 2*hash; }
__device__ __noinline__ unsigned long D::func3(unsigned int hash) { return 3*hash; }
__device__ __noinline__ unsigned long D::func4(unsigned int hash) { return 4*hash; }

__device__ __noinline__ unsigned long secret() {
    printf("[*] Hello Admin!\n");
    return 0x9999999999999999; 
}


__device__ __noinline__ unsigned long unsafe_hash(unsigned long *input, unsigned int len) {
    unsigned long res = 0;
    unsigned long hash = 5381;
    unsigned long *buf = (unsigned long *)malloc(BUF_LEN * sizeof(unsigned long));
    D *obj = new D;

    for (int i=0; i<len; i++) 
        buf[i] = input[i];

    for (int i=0; i<BUF_LEN; i++)
        hash = ((hash << 5) + hash) + buf[i];

    res = obj->func1(hash);
    res = obj->func2(res);
    res = obj->func3(res);
    res = obj->func4(res);
    free(buf);
    delete obj;
    return res;
}


__global__ void test_kernel(unsigned long *hashes, unsigned long *input, unsigned int len, int *admin) {
    unsigned long my_hash;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(*admin)
        my_hash = secret();
    else
        my_hash = unsafe_hash(input+len*idx, len);
    hashes[idx] = my_hash;
}



int main() {
    check_device();

    const int num_blocks = 1;
    const int num_threads = 1;
    const int total_threads = num_threads * num_blocks;

    const int input_len = 8;

    unsigned long h_hashes[total_threads];
    unsigned long h_inputs[input_len];
    int h_admin = 0;

    for (int i=0; i<input_len; i++)
        h_inputs[i] = 0x28e4f200;

    unsigned long *d_hashes;
    unsigned long *d_inputs;
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