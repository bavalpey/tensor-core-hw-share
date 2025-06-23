#include <cuda.h>
#include <cuda_runtime.h>

/* 
__device__ void fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11},"
                 "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7));
} 
*/ 
__device__ void fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
                 "{%0, %1,  %2,  %3},"
                 "{%4, %5},"
                 "{%6, %7},"
                 "{%8, %9, %10, %11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
}
