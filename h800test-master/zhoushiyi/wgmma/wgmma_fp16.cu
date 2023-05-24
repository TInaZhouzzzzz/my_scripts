#include "cuda_fp16.h"
#include <cstdint>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
using namespace std;

#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstdlib>


union FP32
{
    unsigned int i;
    float f;
};


union FP16
{
    unsigned short int i;
    __half f;
};

void InitOne(__half* a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 1.0;
  }
}

void InitZero(__half* a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 0.0;
  }
}


void InitZero_float(float* a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 0.0;
  }
}

void show(float * a, const int n) {
  std::cout << std::endl;
  for ( int i=0; i<n; i++){ 
    std::cout<<a[i] << std::endl;
  }
  std::cout << std::endl;
}




__global__ void wgmma_test1(float *gm_cd, __half *a_desc,  __half *b_desc) {
  float d_array[4];
  for (int i = 0; i < 4; ++i) {
    d_array[i] = gm_cd[i];
  }
  printf("hello");
//asm volatile("{\n\t"
//             "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n\t"
//             "{%0, %1, %2, %3}, %4, %5,1,1,1,0,0;\n\t"
//             "}\n\t"
//             : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]),
//               "+f"(d_array[3])
//             : "l"(a_desc), "l"(b_desc)
//             :);

  for (int i = 0; i < 4; ++i) {
    gm_cd[i] = d_array[i];
  }
}


int main(int argc, char** argv){
  printf("hello");
  int size = 256;
  __half* host_a=(__half*)malloc(sizeof(__half) * size);
  __half* host_b=(__half*)malloc(sizeof(__half) * size);
//float* host_c=(float*)malloc(sizeof(float) * size);
  float* host_d=(float*)malloc(sizeof(float) * size);
  __half* device_a=NULL;
  __half* device_b=NULL;
//float* device_c=NULL;
  float* device_d=NULL;
  cudaMalloc((void**)(&device_a), sizeof(__half) * size);
  cudaMalloc((void**)(&device_b), sizeof(__half) * size);
//cudaMalloc((void**)(&device_c), sizeof(float) * size);
  cudaMalloc((void**)(&device_d), sizeof(float) * size);
  InitZero(host_a, size);
  InitOne(host_b, size);
//InitZero_float(host_c, size);
  InitZero_float(host_d, size);

  FP16 fp16;
  fp16.i = 0x7000; host_a[0]=fp16.f;
  fp16.i = 0x0c00; host_a[1]=fp16.f;
  fp16.i = 0xffff; host_a[2]=fp16.f;
  fp16.i = 0xffff; host_a[3]=fp16.f;
  fp16.i = 0xffff; host_a[4]=fp16.f;
  fp16.i = 0xffff; host_a[5]=fp16.f;
  fp16.i = 0xffff; host_a[6]=fp16.f;
  fp16.i = 0xffff; host_a[7]=fp16.f;

  cudaMemcpy((void*)device_a, (void*)host_a, sizeof(__half)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_b, (void*)host_b, sizeof(__half)* size, cudaMemcpyHostToDevice);
//cudaMemcpy((void*)device_c, (void*)host_c, sizeof(float)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_d, (void*)host_d, sizeof(float)* size, cudaMemcpyHostToDevice);

  wgmma_test1<<<1,32>>>(device_d, device_a, device_b);
  cudaDeviceSynchronize();

  cudaMemcpy((void*)host_d, (void*)device_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  FP32 fp32;
  fp32.f=host_d[0];
//std::cout<< host_d[0] << std::endl;
  std::cout<< hex << fp32.i << std::endl;
//show(host_d, size);
}
