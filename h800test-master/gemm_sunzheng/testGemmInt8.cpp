#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

using namespace std;

#include <cuda_runtime.h>
#include <cublas_v2.h>



void InitChar(char * a, const int rows, const int cols, const int ld) {
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      a[i * ld + j] = rand() % 10 + 1;
    }
  }
}

void PrintChar(char * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
    cout << (int)a[i] << " ";
  }
  cout << endl;
}

void PrintInt(int * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
    cout << a[i] << " ";
  }
  cout << endl;
}

int main(int argc, char *argv[]) {
  int m = 327680;
  int k = 128;
  int n = 256;
  if (argc >= 4) {
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
  }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float time_malloc_copyin = 0.0;
  float time_compute = 0.0;
  float time_copyout = 0.0;


  int sizeA = sizeof(char) * m * k;
  int sizeB = sizeof(char) * k * n;
  int sizeC = sizeof(int) * m * n;

  char *A, *B;
  int *C;

  A = (char *)malloc( sizeA );
  B = (char *)malloc( sizeB );
  C = (int *)malloc( sizeC );

  InitChar(A, m, k, k);
  InitChar(B, k, n, n);

  //PrintChar(A, m * k);
  //PrintChar(B, k * n);

  char *d_A, *d_B;
  int *d_C;
  struct timeval Cstart;
  struct timeval Cend;
  float time_use;

  cudaEventRecord(start, 0);

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_malloc_copyin, start, end);
  cout << "time for malloc and copyin: " << time_malloc_copyin << "ms\n";

  cublasHandle_t handle;
  cublasCreate(&handle);

  int alpha = 1;
  int beta = 0;
  gettimeofday(&Cstart, NULL);
  // warm up
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  cudaEventRecord(start, 0);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
      &alpha, d_A, CUDA_R_8I, m, d_B, CUDA_R_8I, k, &beta,
      (float *)d_C, CUDA_R_32I, m, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  gettimeofday(&Cend, NULL);
  time_use = ((Cend.tv_sec - Cstart.tv_sec) * 1000000 + (Cend.tv_usec - Cstart.tv_usec));
  // std::cout << "Trans-Enc Time Elapsed(MLU excute time, " << i <<"th):" << tv_usec << " us; ";
  std::cout << "Trans-Enc Time Elapsed(CPU excute time, " << 0 <<"th):" <<  time_use << " us" << std::endl;
  cudaEventElapsedTime(&time_compute, start, end);
  std::cout << "time for computing 1st time(default math type): " << time_compute << "ms\n\n";

  int iters = 100;

  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    // gettimeofday(&Cstart, NULL);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &alpha, d_A, CUDA_R_8I, m, d_B, CUDA_R_8I, k, &beta,
        d_C, CUDA_R_32I, m, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    // gettimeofday(&Cend, NULL);
    // time_use = ((Cend.tv_sec - Cstart.tv_sec) * 1000000 + (Cend.tv_usec - Cstart.tv_usec));
    // std::cout << "Trans-Enc Time Elapsed(MLU excute time, " << i <<"th):" << tv_usec << " us; ";
    // std::cout << "Trans-Enc Time Elapsed(CPU excute time, " << i + 1 <<"th):" <<  time_use << " us" << std::endl;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  float t = time_compute / (float)iters * 1000.0;
  double tflops = 2 * m / 1024.0 * k / 1000.0 * (double)n / t;
  cout << "time for computing 1 time: "
       << t / 1000.0 << "ms, TOPS = " << tflops << "\n";
  float bw = (n * k * 1.0 + m * k * 1.0 + n * m * 4.0) / 1000.0 / t;  
  std::cout << "Equivalent bandwidth: " << bw << "Gbps" << std::endl << std::endl;


  iters = 10;
  float a = 1, b = 0;
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    // gettimeofday(&Cstart, NULL);
    // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
    //     &alpha, d_A, CUDA_R_8I, m, d_B, CUDA_R_8I, k, &beta,
    //     (float *)d_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
      &a, d_A, CUDA_R_8I, m, d_B, CUDA_R_8I, k, &b,
      (float *)d_C, CUDA_R_32F, m);
    // gettimeofday(&Cend, NULL);
    // time_use = ((Cend.tv_sec - Cstart.tv_sec) * 1000000 + (Cend.tv_usec - Cstart.tv_usec));
    // std::cout << "Trans-Enc Time Elapsed(MLU excute time, " << i <<"th):" << tv_usec << " us; ";
    // std::cout << "Trans-Enc Time Elapsed(CPU excute time, " << i + 1 <<"th):" <<  time_use << " us" << std::endl;
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "Time for computing 1 time(FP16-OUT): "
       << time_compute / (float)iters << "ms\n";

  cublasDestroy(handle);

  cudaEventRecord(start, 0);

  cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_copyout, start, end);
  cout << "time for copyout: " << time_copyout << "ms\n";

  free(A);
  free(B);
  free(C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
}
