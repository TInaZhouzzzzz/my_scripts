#include <iostream>
#include <cstdlib>
#include <random>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#define TF32 0

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

void InitFloat(float * a, const int rows, const int cols, const int ld, float mu, float sigma, int seed, bool allzero = false) {
  // srand((int)time(0));
  // int seed = rand() % 65536;
  cout << "seed = " << seed << endl;
  std::default_random_engine re(seed);
  std::uniform_real_distribution<float> dis(-65504, 65504);
  // std::normal_distribution<float> dis(0, 10.0);  // 15.090 ms for 8192
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      // srand((int)time(0));
      // a[i * ld + j] = float((double)(rand()) / (double)RAND_MAX * (65504 * 2)) - 65504; // 12.197ms
      if (allzero) {
	a[i * ld + j] = 0;
      } else {
        a[i * ld + j] = dis(re);  // different seed
      }
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
  int is_transA = 0;
  int is_transB = 0;
  bool allzero = false;
  if (argc >= 4) {
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
  }
  if (argc >= 6) {
    is_transA = atoi(argv[4]);
    is_transB = atoi(argv[5]);
  }
  if (argc >= 7) {
    allzero = atoi(argv[6]) == 0;
    if (allzero) 
      cout << " Test all zero!" << endl;
  }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float time_malloc_copyin = 0.0;
  float time_compute = 0.0;
  float time_copyout = 0.0;

  cout << "Trans param: " << "TransA =" << is_transA << ", TransB=" << is_transB << "; ";
  cout << "Dim param: " << "m=" << m << ", k=" << k << ", n=" << n << "\n";
  cublasOperation_t TransA = is_transA == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t TransB = is_transB == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = is_transA == 1 ? k : m;
  int ldb = is_transB == 1 ? n : k;


  int sizeA = sizeof(float) * m * k;
  int sizeB = sizeof(float) * k * n;
  int sizeC = sizeof(float) * m * n;

  // cout << "Dim param: " << "m=" << m << ", k=" << k << ", n=" << n << "\n";

  float *A, *B;
  float *C;

  A = (float *)malloc( sizeA );
  B = (float *)malloc( sizeB );
  C = (float *)malloc( sizeC );

  InitFloat(A, m, k, k, 0, 2.0, 25, allzero);
  InitFloat(B, k, n, n, 0, 2.0, 255, allzero);

  //PrintChar(A, m * k);
  //PrintChar(B, k * n);

  float *d_A, *d_B;
  float *d_C;

  cudaEventRecord(start, 0);

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_malloc_copyin, start, end);
  cout << " Time for malloc and copyin: " << time_malloc_copyin << "ms\n";

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1;
  float beta = 0;

  // warm up
  cudaEventRecord(start, 0);
  cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
      &alpha, d_A, CUDA_R_32F, m, d_B, CUDA_R_32F, k, &beta,
      d_C, CUDA_R_32F, m);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  // cout << "  Time for warmup time: " << time_compute << "ms\n";

  int wmp = 5;
  cudaEventRecord(start, 0);
  for (int i = 0; i < wmp; i++) {
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &alpha, d_A, CUDA_R_32F, m, d_B, CUDA_R_32F, k, &beta,
        d_C, CUDA_R_32F, m);
  }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "  CUBLAS Version: " << CUBLAS_VER_MAJOR << ", time for warmup time(sgemmEx): " 
       << time_compute / (float)wmp << "ms\n";


  int iters = 100;

  cublasMath_t math_type;
  cublasComputeType_t com_type; 
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#if (CUBLAS_VER_MAJOR >= 11 && TF32)
  math_type = CUBLAS_TF32_TENSOR_OP_MATH;
  com_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  algo = CUBLAS_GEMM_ALGO0_TENSOR_OP;
  cout << "  Use tf32 acceleration! " << endl; 
#else 
  math_type = CUBLAS_DEFAULT_MATH; algo = CUBLAS_GEMM_DEFAULT;
  math_type = CUBLAS_TENSOR_OP_MATH; algo = CUBLAS_GEMM_ALGO0_TENSOR_OP;
  com_type = CUBLAS_COMPUTE_32F;
  cout << "  Use fp16 acceleration! " << endl; 
#endif

  cublasSetMathMode(handle, math_type);
  // com_type = CUBLAS_COMPUTE_32F;
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    // cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
    //   &alpha, d_A, CUDA_R_32F, m, d_B, CUDA_R_32F, k, &beta,
    //   d_C, CUDA_R_32F, m);
    cublasGemmEx(handle, TransA, TransB, m, n, k,
        &alpha, d_A, CUDA_R_32F, lda, d_B, CUDA_R_32F, ldb, &beta,
        d_C, CUDA_R_32F, m, com_type, algo);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  float elapsed = time_compute / (float)iters * 1000.0;
  double tflops = double(m) * double(k) * double(n) * 2.0 / elapsed / 1024.0 / 1000.0;
  cout << "  Average time for computing one time (tensor core): "
       << elapsed << " us, TFLOPS = " << tflops << endl << endl;

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
