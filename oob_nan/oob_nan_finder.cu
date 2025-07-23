#include <cuda_fp16.h>
#include <stdlib.h>
#include <stdint.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void kernel(__half *a, __half *b, __half *c, __half *d)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    short y = a[gid];
    short z = b[gid];
    short w = c[gid];
    short x;
    asm volatile("{fma.rn.oob.f16 %0, %1, %2, %3;}" : "=h"(x) : "h"(y), "h"(z), "h"(w));
    memcpy(&d[gid], &x, sizeof(short));
}

int main()
{
    const uint16_t value = 0x7c00; // infinity in half-precision.
    thrust::host_vector<__half> h_A(4096);
    thrust::host_vector<__half> h_B(4096);
    thrust::host_vector<__half> h_C(4096);
    thrust::host_vector<__half> h_D(4096);

    for (uint16_t i = 0; i < h_A.size(); ++i)
    {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(1.0f);
    }

    // Make C the NaN representation.
    for (uint16_t i = 0; i < 2048; ++i)
    {
        h_C[i] = (__half)(value | (i + 1));
        h_C[i + 2048] = (__half)(value | 0x8000 | (i + 1));
    }

    thrust::device_vector<__half> d_A = h_A;
    thrust::device_vector<__half> d_B = h_B;
    thrust::device_vector<__half> d_C = h_C;
    thrust::device_vector<__half> d_D(4096);
    kernel<<<4, 1024>>>(thrust::raw_pointer_cast(d_A.data()),
                        thrust::raw_pointer_cast(d_B.data()),
                        thrust::raw_pointer_cast(d_C.data()),
                        thrust::raw_pointer_cast(d_D.data()));

    thrust::copy(d_D.begin(), d_D.end(), h_D.begin());

    // Now, go through each element in h_D and check if the result was 0 instead of NaN.
    for (uint16_t i = 0; i < h_D.size(); ++i)
    {
        if (((uint16_t )h_D[i]) == 0)
        {
            printf("OOB nan is: %#06hx\n", (uint16_t)h_C[i]);
        } else {
            printf("Non-OOB NaN: %#06hx\n", (uint16_t)h_D[i]);
        }
    }

    return EXIT_SUCCESS;
}