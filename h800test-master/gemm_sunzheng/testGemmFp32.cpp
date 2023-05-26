#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TF32 1
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
// #define TEST_RESULT

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
  std::uniform_real_distribution<float> dis(-2, 2);
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

void matmul(float *A, 
	    float *B, 
	    float *C, 
	    int m, 
	    int k, 
	    int n, 
	    int lda, 
	    int ldb, 
	    int ldc, 
	    float alpha = 1, 
	    float beta = 0) {
 for (int i = 0 ; i < m ; i ++){
    for (int j = 0 ; j < n; j ++){
      float ans = 0;
      double cnt = 0.0;
      for (int x = 0 ; x < k ; x ++ ){
        ans += A[IDX2C(i, x, lda)] * B[IDX2C(x, j, ldb)];
      }
      C[IDX2C(i, j, ldc)] = alpha * ans + beta * C[IDX2C(i, j, ldc)];
    }
  }
}

void compares(float *C, float *base, int size){
  double ans = 0, cnt=0;
  for (int i = 0 ; i < size ; i ++){
    ans += abs(C[i] - base[i]);
    cnt += abs(base[i]);
  }
  double diff = double(ans/cnt);
  cout << "diffs: " << diff << endl;
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
  int m = 320;
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
  int ldc = m;


  int sizeA = sizeof(float) * m * k;
  int sizeB = sizeof(float) * k * n;
  int sizeC = sizeof(float) * m * n;

  cout << "Dim param: " << "m=" << m << ", k=" << k << ", n=" << n << "\n";

  float *A, *B;
  float *C;

  A = (float *)malloc( sizeA );
  B = (float *)malloc( sizeB );
  C = (float *)malloc( sizeC );

  InitFloat(A, m, k, k, 0, 2.0, 25, allzero);
  InitFloat(B, k, n, n, 0, 2.0, 255, allzero);
  InitFloat(C, k, n, n, 0, 2.0, 127, allzero);

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
  cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_malloc_copyin, start, end);
  cout << " Time for malloc and copyin: " << time_malloc_copyin << "ms\n";

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1;
  float beta = 0;

#ifndef TEST_RESULT
  cudaEventRecord(start, 0);
  cout << "****** Testing Sgemm ****** \n";
  cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
      &alpha, d_A, CUDA_R_32F, m, d_B, CUDA_R_32F, k, &beta,
      d_C, CUDA_R_32F, m);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_compute, start, end);
  cout << "  Time for Sgemm warmup time: " << time_compute << "ms\n";

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
#endif


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
//math_type = CUBLAS_DEFAULT_MATH; algo = CUBLAS_GEMM_DEFAULT;
  math_type = CUBLAS_TENSOR_OP_MATH; algo = CUBLAS_GEMM_ALGO0_TENSOR_OP;
  com_type = CUBLAS_COMPUTE_32F;
  cout << "  Use fp16 acceleration! " << endl; 
#endif

  cublasSetMathMode(handle, math_type);
  // com_type = CUBLAS_COMPUTE_32F;
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

#ifdef TEST_RESULT
  char *test_res = std::getenv("GEMM_TEST_RES");
  if (test_res != NULL) {
    std::string env_str = test_res;
    int env_num = std::stoi(env_str);
    if (env_num == 1) {
      std::cout << "Testing result" << std::endl; 
      float *Cres;
      Cres = (float *)malloc( sizeC );
      std::memcpy(Cres, C, sizeC);
      
      cudaDeviceSynchronize();
      cublasGemmEx(handle, TransA, TransB, m, n, k,
        &alpha, d_A, CUDA_R_32F, lda, d_B, CUDA_R_32F, ldb, &beta,
        d_C, CUDA_R_32F, ldc, com_type, algo);
      cudaDeviceSynchronize();
      cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
      
      // matmul(A, B, Cres, m, k, n, lda, ldb, ldc, alpha, beta);
      // compares(C, Cres, m * n);
      ofstream ofc;
      ofc.open("gpu_c.txt");
      for (int i = 0 ; i < m * n  ; i ++){
        ofc << fixed << setprecision(20) << C[i] << endl; //列优先转化行优先
      }
      ofc.close();
      ofc.open("cpu_c.txt");
      for (int i = 0 ; i < m * n  ; i ++){
        ofc << fixed << setprecision(20) << Cres[i] << endl; //列优先转化行优先
      }
      ofc.close();
      free(Cres);
      std::cout << "Finishing Testing result" << std::endl; 
    }
  }
#endif

  // warm up
  cudaEventRecord(start, 0);
  for ( int i = 0; i < iters; i++ ) {
    // cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
    //   &alpha, d_A, CUDA_R_32F, m, d_B, CUDA_R_32F, k, &beta,
    //   d_C, CUDA_R_32F, m);
    break;
    cublasGemmEx(handle, TransA, TransB, m, n, k,
        &alpha, d_A, CUDA_R_32F, lda, d_B, CUDA_R_32F, ldb, &beta,
        d_C, CUDA_R_32F, ldc, com_type, algo);
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
