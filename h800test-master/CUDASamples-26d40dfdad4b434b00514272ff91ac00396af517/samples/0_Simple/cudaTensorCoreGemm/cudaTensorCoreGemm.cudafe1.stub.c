#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "cudaTensorCoreGemm.fatbin.c"
extern void __device_stub__Z20init_device_matricesPKfS0_S0_P6__halfS2_PfS3_(const float *, const float *, const float *, half *, half *, float *, float *);
extern void __device_stub__Z12compute_gemmPK6__halfS1_PKfPfff(const half *, const half *, const float *, float *, float, float);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z20init_device_matricesPKfS0_S0_P6__halfS2_PfS3_(const float *__par0, const float *__par1, const float *__par2, half *__par3, half *__par4, float *__par5, float *__par6){__cudaLaunchPrologue(7);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 40UL);__cudaSetupArgSimple(__par6, 48UL);__cudaLaunch(((char *)((void ( *)(const float *, const float *, const float *, half *, half *, float *, float *))init_device_matrices)));}
# 141 "cudaTensorCoreGemm.cu"
void init_device_matrices( const float *__cuda_0,const float *__cuda_1,const float *__cuda_2,half *__cuda_3,half *__cuda_4,float *__cuda_5,float *__cuda_6)
# 142 "cudaTensorCoreGemm.cu"
{__device_stub__Z20init_device_matricesPKfS0_S0_P6__halfS2_PfS3_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 154 "cudaTensorCoreGemm.cu"
}
# 1 "cudaTensorCoreGemm.cudafe1.stub.c"
void __device_stub__Z12compute_gemmPK6__halfS1_PKfPfff( const half *__par0,  const half *__par1,  const float *__par2,  float *__par3,  float __par4,  float __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaLaunch(((char *)((void ( *)(const half *, const half *, const float *, float *, float, float))compute_gemm))); }
# 156 "cudaTensorCoreGemm.cu"
void compute_gemm( const half *__cuda_0,const half *__cuda_1,const float *__cuda_2,float *__cuda_3,float __cuda_4,float __cuda_5)
# 157 "cudaTensorCoreGemm.cu"
{__device_stub__Z12compute_gemmPK6__halfS1_PKfPfff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 328 "cudaTensorCoreGemm.cu"
}
# 1 "cudaTensorCoreGemm.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T6) {  __nv_dummy_param_ref(__T6); __nv_save_fatbinhandle_for_managed_rt(__T6); __cudaRegisterEntry(__T6, ((void ( *)(const half *, const half *, const float *, float *, float, float))compute_gemm), _Z12compute_gemmPK6__halfS1_PKfPfff, (-1)); __cudaRegisterEntry(__T6, ((void ( *)(const float *, const float *, const float *, half *, half *, float *, float *))init_device_matrices), _Z20init_device_matricesPKfS0_S0_P6__halfS2_PfS3_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
