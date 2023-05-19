#include "cuda_fp16.h"
#include <cstdint>

__global__ void wgmma_test1(float *gm_cd, uint64_t *desc) {
  uint64_t a_desc = desc[0], b_desc = desc[1];
  float d_array[4];
  for (int i = 0; i < 4; ++i) {
    d_array[i] = gm_cd[i];
  }
  asm volatile("{\n\t"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n\t"
               "{%0, %1, %2, %3}, %4, %5,1,1,1,0,0;\n\t"
               "}\n\t"
               : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]),
                 "+f"(d_array[3])
               : "l"(a_desc), "l"(b_desc)
               :);

  for (int i = 0; i < 4; ++i) {
    gm_cd[i] = d_array[i];
  }
}
