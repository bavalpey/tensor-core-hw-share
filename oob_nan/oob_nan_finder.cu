#include <cuda_fp16.h>
#include <stdlib.h>
#include <stdint.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void kernel(void *a, void *b, void *c, void *d)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t y = ((uint16_t *)a)[gid];
    uint16_t z = ((uint16_t *)b)[gid];
    uint16_t w = ((uint16_t *)c)[gid];
    asm volatile("{fma.rn.oob.f16 %0, %1, %2, %3;}" : "=h"(((uint16_t *)d)[gid]) : "h"(y), "h"(z), "h"(w));
    if (threadIdx.x == 1)
    {
        printf("Thread %d: a = %#06hx, b = %#06hx, c = %#06hx, d = %#06hx\n", threadIdx.x, y, z, w, ((uint16_t *)d)[gid]);
    }
}

int main()
{
    const uint16_t value = 0x7c00; // infinity in half-precision.
    thrust::host_vector<uint16_t> h_A(2048);
    thrust::host_vector<uint16_t> h_B(2048);
    thrust::host_vector<uint16_t> h_C(2048);
    thrust::host_vector<uint16_t> h_D(2048);

    for (uint16_t i = 0; i < h_A.size(); ++i)
    {
        h_A[i] = ((uint16_t) __float2half(1.0f));
        h_B[i] = ((uint16_t) __float2half(1.0f));
        h_C[i] = ((uint16_t) __float2half(1.0f));
    }

    // Make C the NaN representation.
    for (uint16_t i = 0; i < 1024; ++i)
    {
        if (i != 0)
        {
            h_C[i] = value | i;
            // Second block checks signed nans.
            h_C[i + 1024] = (value | 0x8000 | i);
        }
        else
        {
            h_C[i] = (uint16_t) __float2half(1.0f);
            h_C[i + 1024] = (uint16_t) __float2half(1.0f);
        }
    }

    thrust::device_vector<__half> d_A(2048);
    thrust::device_vector<__half> d_B(2048);
    thrust::device_vector<__half> d_C(2048);
    thrust::copy(h_A.begin(), h_A.end(), d_A.begin());
    thrust::copy(h_B.begin(), h_B.end(), d_B.begin());
    thrust::copy(h_C.begin(), h_C.end(), d_C.begin());
    thrust::device_vector<__half> d_D(2048);
    cudaDeviceSynchronize();
    kernel<<<2, 1024>>>(thrust::raw_pointer_cast(d_A.data()),
                        thrust::raw_pointer_cast(d_B.data()),
                        thrust::raw_pointer_cast(d_C.data()),
                        thrust::raw_pointer_cast(d_D.data()));

    thrust::copy(d_D.begin(), d_D.end(), h_D.begin());
    cudaDeviceSynchronize();

    // Now, go through each element in h_D and check if the result was 0 instead of NaN.
    for (uint16_t i = 0; i < h_D.size(); ++i)
    {
        if (((uint16_t)h_D[i]) == 0)
        {
            printf("OOB-Nan is: %#06hx\n", (uint16_t)h_C[i]);
        }
    }

    return EXIT_SUCCESS;
}