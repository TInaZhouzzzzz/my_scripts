#include "cuda_fp16.h"
#include <cuda_runtime_api.h>
#include <random>
// #include "cudaCode.h"
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

// #include <cuda_runtime.h>
#include <cublas_v2.h>
typedef short int  int16_t;

int16_t float2half(float x) {
  const int fs_shift = 31;
  const int fe_shift = 23;
  const int fe_mark = 0xff;
  const int hs_shift = 15;
  const int he_shift = 10;
  int *in1 = (int *)&x;
  int in = *in1;
  int sign = in >> fs_shift;
  int exp = ((in >> fe_shift) & fe_mark) - 127;
  int denorm = 0;
  int eff = 0;
  int g = 0;  // for round
  if (exp >= 16) {
    exp = 0xf;
    eff = 0x3ff;
  } else if (exp >= -14) {
    g = (in >> 12) & 1;
    eff = (in >> 13) & 0x3ff;
  } else if (exp >= -24) {
    g = (((in & 0x7fffff) | 0x800000) >> (-exp - 2)) & 1;
    eff = (((in & 0x7fffff) | 0x800000) >> (-exp - 1)) & 0x3ff;
    denorm = 1;
    exp = 0;
  } else {
    exp = 0;
    denorm = 1;
    eff = in ? 1 : 0;
  }
  eff += g;  // round
  exp = (denorm == 1) ? exp : (exp + 15);
  int result = (sign << hs_shift) + (exp << he_shift) + eff;
  return result;
}


void trans_float2half(half *dst, float *src, int size) {
  int fltint32;
  short fltint16;
  for (int i = 0; i < size; i++) {
    // memcpy(&fltint32, (float *)src + i, sizeof(float));
    // fltint16 = ((fltint32 & 0x007FFFFF) >> 13) - (0x38000000 >> 13);
    // fltint16 |= (fltint32 & 0x80000000) >> 16;
    // *(dst + i) = fltint16;
    *(dst + i) = (half)(*((float *)src + i));
  }
}

void InitFloat(float * a, const int rows, const int cols, const int ld, float mu, float sigma, int seed, bool all_zero) {
  // srand((int)time(0));
  // int seed = rand() % 65536;
  cout << "seed = " << seed << endl;
  std::default_random_engine re(seed);
  std::uniform_real_distribution<float> dis(-2, 2);
  // std::normal_distribution<float> dis(0, 10.0);  // 15.090 ms for 8192
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      // srand((int)time(0));
      // a[i * ld + j] = float((double)(rand()) / (double)RAND_MAX * (65504 * 2)) - 65504; // 12.197ms
      a[i * ld + j] = all_zero ? 0 : dis(re);
      // a[i * ld + j] = dis(re);  // different seed, 15.239ms for 8192 uniform
    }
  }
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
  bool all_zero = 0;
  int repeats = 1000;
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
    all_zero = atoi(argv[6]);
  }
  if (argc >= 8) {
    repeats = atoi(argv[7]);
  }

  int startAlgo, endAlgo, fast_algo;
  // if(sizeof(T) == sizeof(float)){
  //   startAlgo = (int)CUBLAS_GEMM_DEFAULT;
  //   endAlgo = (int)CUBLAS_GEMM_ALGO23;
  // }
  // else{
    startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  // }
  fast_algo = startAlgo;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float time_malloc_copyin = 0.0;
  float time_compute = 0.0;
  float time_copyout = 0.0;


  cout << "Trans param: " << "TransA =" << is_transA << ", TransB=" << is_transB << "; ";
  cout << "Dim param: " << "m=" << m << ", k=" << k << ", n=" << n << "\n";
  cout << "Test param: " << "all_zero: " << all_zero << ", repeats: " << repeats << "\n";
  cublasOperation_t TransA = is_transA == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t TransB = is_transB == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = is_transA == 1 ? k : m;
  int ldb = is_transB == 1 ? n : k;

  int sizeA = sizeof(float) * m * k;
  int sizeB = sizeof(float) * k * n;
  int sizeC = sizeof(float) * m * n;

  float *A, *B;
  float *C;
  int16_t *hA, *hB;

  A = (float *)malloc( sizeA );
  B = (float *)malloc( sizeB );
  C = (float *)malloc( sizeC );
  hA = (int16_t *)malloc( sizeA / 2 );
  hB = (int16_t *)malloc( sizeB / 2 );

  InitFloat(A, m, k, k, 0, 2.0, 25, all_zero);
  InitFloat(B, k, n, n, 0, 2.0, 255, all_zero);
  printf("%f %f %f %f\n", A[10], B[10], A[20], B[20]);
  // for (int p = 0; p < m * k; p++) {
  //   // hA[p] = float2half(A[p]);
  //   hA[p] = float2half(A[p]);
  // }
  // for (int p = 0; p < n * k; p++) {
  //   hB[p] = float2half(B[p]);
  // }
  trans_float2half((half *)hA, A, m * k);
  trans_float2half((half *)hB, B, n * k);

  //PrintChar(A, m * k);
  //PrintChar(B, k * n);

  float *d_A, *d_B;
  float *d_C;
  
  half2 *h_A, *h_B, *h_C;
  struct timeval Cstart;
  struct timeval Cend;
  struct timeval astart;
  struct timeval aend;
  float time_use;

  cudaEventRecord(start, 0);

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);
  cudaMalloc((void **)&h_A, sizeA / 2);
  cudaMalloc((void **)&h_B, sizeB / 2);
  cudaMalloc((void **)&h_C, sizeC / 2);

  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
  cudaMemcpy(h_A, hA, sizeA / 2, cudaMemcpyHostToDevice);
  cudaMemcpy(h_B, hB, sizeB / 2, cudaMemcpyHostToDevice);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_malloc_copyin, start, end);
  cout << "Time for malloc and copyin: " << time_malloc_copyin << "ms\n";

  cublasHandle_t handle;
  cublasCreate(&handle);

  int alpha = 1;
  int beta = 0;
  gettimeofday(&Cstart, NULL);
  // warm up
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  cudaEventRecord(start, 0);
  cublasGemmEx(handle, TransA, TransB, m, n, k,
      &alpha, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &beta,
      h_C, CUDA_R_16F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  gettimeofday(&Cend, NULL);
  time_use = ((Cend.tv_sec - Cstart.tv_sec) * 1000000 + (Cend.tv_usec - Cstart.tv_usec));
  // std::cout << "GemmEx Time Elapsed(CPU fp16-tensor_core, " << 0 <<"th):" <<  time_use << " us" << std::endl;
  cudaEventElapsedTime(&time_compute, start, end);
  std::cout << "GemmEx computing time for warm up(fp16, GEMM_DEFAULT): " << time_compute 
            << "/" << time_use / 1000.0 << "ms\n\n";

  int iters = 5;

  // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    cublasGemmEx(handle, TransA, TransB, m, n, k,
        &alpha, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &beta,
        h_C, CUDA_R_16F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "GemmEx default time for FP16 computing(1 time): "
       << time_compute / (float)iters << "ms\n";
  float t = time_compute / (float)iters * 1000.0;
  float bw = (n * k * 2.0 + m * k * 2.0 + n * m * 2.0) / 1000.0 / t;  
  std::cout << "Equivalent bandwidth: " << bw << "Gbps" << std::endl << std::endl;


  iters = 10;
  float a = 1, b = 0;
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    cublasSgemmEx(handle, TransA, TransB, m, n, k,
      &a, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &b,
      h_C, CUDA_R_16F, m);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "Time for SgemmEx computing 1 time(FP16-OUT, tensor_core): "
       << time_compute / (float)iters << "ms\n";

  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    cublasSgemmEx(handle, TransA, TransB, m, n, k,
      &a, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &b,
      (float *)d_C, CUDA_R_32F, m);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "Time for SgemmEx computing 1 time(FP16-IN, FP32-OUT, tensor_core): "
       << time_compute / (float)iters << "ms\n";
  // return 1;
  iters = 10;
  if (m >= 4096) {
    iters = 10;
  }
  float fast_time = 1000000000, time_tmp = 0;
  for(int algo = startAlgo; algo <= endAlgo; algo++)
  {
    cublasStatus_t status;
    cudaDeviceSynchronize();
    gettimeofday(&Cstart, NULL);
    for(int ite = 0; ite < iters; ++ite) {
      gettimeofday(&astart, NULL);
      status = cublasGemmEx(handle, TransA, TransB, m, n, k,
                 &alpha, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &beta,
                 h_C, CUDA_R_16F, m, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(algo));
      gettimeofday(&aend, NULL);
      float txtmp = ((aend.tv_sec - astart.tv_sec) * 1000000 + (aend.tv_usec - astart.tv_usec));
      // std::cout << "    time launch: " << txtmp << "us" << std::endl;
      if(status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " Not support Algo " << algo << std::endl;
        break;
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&Cend, NULL);
    time_tmp = ((Cend.tv_sec - Cstart.tv_sec) * 1000000 + (Cend.tv_usec - Cstart.tv_usec));
    // std::cout << time_tmp / (float)iters << "  " << status << "/" 
    //           << CUBLAS_STATUS_SUCCESS << ";  ";
    if ((status == CUBLAS_STATUS_SUCCESS) && ((fast_time * iters) > time_tmp)) {
      fast_time = time_tmp / (float)iters;
      fast_algo = algo;
    } 
  }
  std::cout << "Algo select finished!" << std::endl;
  iters= repeats;
  cublasStatus_t status;
  cudaDeviceSynchronize();
  gettimeofday(&Cstart, NULL);
  for(int ite = 0; ite < iters; ++ite) {
    status = cublasGemmEx(handle, TransA, TransB, m, n, k,
               &alpha, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &beta,
               h_C, CUDA_R_16F, m, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(fast_algo));
  }
  cudaDeviceSynchronize();
  gettimeofday(&Cend, NULL);
  time_tmp = ((Cend.tv_sec - Cstart.tv_sec) * 1000000 + (Cend.tv_usec - Cstart.tv_usec));
  fast_time = time_tmp / (float)iters;
  float tflops = 2 * m / 1000.0 * k / 1000.0 * n / fast_time;
  printf("tensor core fp16: fast_algo %d costs %.3f ms, TFLOPS = %.3f\n", fast_algo, fast_time / 1000.0, tflops);
  
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    cublasSgemmEx(handle, TransA, TransB, m, n, k,
      &a, h_A, CUDA_R_16F, lda, h_B, CUDA_R_16F, ldb, &b,
      h_C, CUDA_R_16F, m);
  }

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "Time for SgemmEx computing(FP16-OUT, tensor_core): "
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
  free(hA);
  free(hB);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(h_A);
  cudaFree(h_B);
  cudaFree(h_C);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
}
