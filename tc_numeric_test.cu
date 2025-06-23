#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstdint>
#include <unistd.h>
#include <inttypes.h>
#include <assert.h>
#include <type_traits>


#ifndef EXHAUSTIVE_NAN_TEST
#define EXHAUSTIVE_NAN_TEST 0
#endif

typedef union {
      float f;
      uint32_t u;
} w32_un; 
typedef union {
  half h;
  uint16_t u;
} w16_un;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__host__ void print_matrix(float *matrix, int row, int col)
{
  for(int i = 0; i < row; ++i)
  {
    printf("[");
    for(int j = 0; j < col-1; ++j)
    {
      printf("%f, ", matrix[i*row+j]);
    }
    printf("%f]\n", matrix[i*row+col-1]);
  }
}


struct mat_spec {
  int row;
  int col;
  bool row_major = true;
};

__host__ void print_matrix(half *matrix, int row, int col)
{
  for(int i = 0; i < row; ++i)
  {
    printf("[");
    for(int j = 0; j < col-1; ++j)
    {
      printf("%f, ", __half2float(matrix[i*row+j]));
    }
    printf("%f]\n", __half2float(matrix[i*row+col-1]));
  }
}

bool standard_is_valid(int m, int n, int k)
{
  return (k == 16) && ((m == 16 && n == 16) || (m == 32 && n == 8) || (m==8 && n == 32));
}

__half uint16_as_fp16 (uint16_t a)
{
    __half res;
#if defined (__cplusplus)
    memcpy (&res, &a, sizeof (res));
#else /* __cplusplus */
    volatile union {
        __half f;
        uint16_t i;
    } cvt;
    cvt.i = a;
    res = cvt.f;
#endif /* __cplusplus */
    return res;
}


__host__ void InitMatrix(half *A, half *B, float *C, half *& d_A, half *& d_B, float *& d_C, int m, int n, int k, bool do_random=false)
{
	// for half, half, and float, we have 3 options:
	// 16x16x16,
	// 32x8x16
	// and 8x32x16
  if (standard_is_valid(m, n, k))
  {
    for(int i = 0; i < m*k; ++i) A[i] = __float2half(do_random ? (rand() % 1000 / 1000.0f) : 0.0f);
    for(int i = 0; i < k*n; ++i) B[i] = __float2half(do_random ? (rand() % 1000 / 1000.0f) : 0.0f);
    for(int i = 0; i < m*n; ++i) C[i] = 0.0f;
  } else
  {
    fprintf(stderr, "Invalid matrix configuration\n");
    exit(EXIT_FAILURE);
  }
  gpuErrchk(cudaMemcpy(d_A, A, sizeof(half)*m*k, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_B, B, sizeof(half)*k*n, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_C, C, sizeof(float)*m*n, cudaMemcpyHostToDevice));
}

// host matrices will be indexed by [row][col]


template <typename T>
__host__ T** allocMatrix(mat_spec spec)
{
  T *ptr, **arr;

  int dim1, dim2;
  if (spec.row_major){
    dim1 = spec.row;
    dim2 = spec.col;
  } else {
    dim1 = spec.col;
    dim2 = spec.row;
  }

  arr = (T**) calloc(sizeof(T *) * dim1 + sizeof(T) * dim2 * dim1, 1);
  ptr = (T *)(arr + dim1);
  for(int i = 0; i < dim1; ++i)
    arr[i] = (ptr + dim2 * i);

  return arr;
}

__host__ void IndexMatrix(half **A, mat_spec spec, int invert) 
{
  for(uint16_t row = 0; row < spec.row; ++row)
  {
    for(uint16_t col = 0; col < spec.col; ++col)
    {
      if (spec.row_major) {
        ((uint16_t **) A)[row][col] = (row << 4) | col;
        if (invert)
          ((uint16_t **) A)[row][col] |= 0x8000;
      }
      else {
        ((uint16_t **) A)[col][row] = (row << 4) | col;
        if (invert)
          ((uint16_t **) A)[col][row] |= 0x8000;
      }
    }
  }
}

__host__ void IdentityMatrix(half **A, mat_spec spec)
{
  for(int row = 0; row < spec.row; ++row)
  {
    for(int col = 0; col < spec.col; ++col)
    {
      A[col][row] = __float2half(1.0f);
    }
  }
}

__host__ void IdentityMatrix(float **A, mat_spec spec)
{
  for(int row = 0; row < spec.row; ++row)
  {
    for(int col = 0; col < spec.col; ++col)
    {
        A[row][col] = 1.0f;
    }
  }
}
__host__ void IndexMatrix(float **A, mat_spec spec, int invert)
{
  for(uint32_t row = 0; row < spec.row; ++row)
  {
    for(uint32_t col = 0; col < spec.col; ++col)
    {
      if (spec.row_major)
      {
        ((uint32_t **) A)[row][col] = (row << 4) | col;
        if (invert)
          ((uint32_t **) A)[row][col] |= 0xC0000000;
      }
      else {
        ((uint32_t **) A)[col][row] = (row << 4) | col;
        if (invert)
          ((uint32_t **) A)[col][row] |= 0xC0000000;
      }
    }
  }
}
template <typename T, typename U>
__host__ void copyMatrix(T **A, T **B, U **C, T *& d_A, T *& d_B, U *& d_C, int m, int n, int k)
{
	// for half, half, and half, we have 3 options:
	// 16x16x16
	// 32x8x16
	// and 8x32x16
  if (standard_is_valid(m, n, k))
  {
    gpuErrchk(cudaMemcpy(d_A, A[0], sizeof(T)*m*k, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B[0], sizeof(T)*k*n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C, C[0], sizeof(U)*m*n, cudaMemcpyHostToDevice));
  } else
  {
    fprintf(stderr, "Invalid matrix configuration\n");
    exit(EXIT_FAILURE);
  }
}


__global__ void wmma_ker_16x16x16(half *a, half *b, float *c, float *d)
{
#if __CUDA_ARCH__ >= 700
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> d_frag;

  // Initialize the output to zero
  nvcuda::wmma::fill_fragment(d_frag, 0.0f);

  // Load the inputs
  nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
  nvcuda::wmma::load_matrix_sync(b_frag, b, 16);
  nvcuda::wmma::load_matrix_sync(c_frag, c, 16, nvcuda::wmma::mem_row_major);

  // Perform the matrix multiplication
  nvcuda::wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

  // Store the output
  nvcuda::wmma::store_matrix_sync(d, d_frag, 16, nvcuda::wmma::mem_row_major);
#else
  return;
#endif
}


__global__ void wmma_ker_16x16x16(half *a, half *b, half *c, half *d)
{
#if __CUDA_ARCH__ >= 700
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> c_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> d_frag;

  // Initialize the output to zero
  nvcuda::wmma::fill_fragment(d_frag, __float2half(0.0f));

  // Load the inputs
  nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
  nvcuda::wmma::load_matrix_sync(b_frag, b, 16);
  nvcuda::wmma::load_matrix_sync(c_frag, c, 16, nvcuda::wmma::mem_row_major);

  // Perform the matrix multiplication
  nvcuda::wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

  // Store the output
  nvcuda::wmma::store_matrix_sync(d, d_frag, 16, nvcuda::wmma::mem_row_major);

#else
  return;
#endif
}


void printResultBits(float res)
{
  printf("Result is: %f (0x%08x)\n", res, (w32_un) {.f=res}.u);
}

void printResultBits(half res)
{
  printf("Result is: 0x%04hx\n", (w16_un) {.h=res}.u);
}

void printRoundingTestResult(const char rm1[], float res1, const char rm2[], float res2, float actual)
{
  printf("If result was %s it would be: %x, if it was %s it would be: %x. Hardware result is: %x\n",
      rm1,
      (w32_un) {.f = res1}.u,
      rm2,
      (w32_un) {.f = res2}.u,
      (w32_un) {.f = actual}.u);

}

float testRoundingMode_final(const float a, const float b)
{
  int M = 16;
  int N = 16;
  int K = 16;

  mat_spec spec_A = {.row = M, .col=K, .row_major = true};
  mat_spec spec_B = {.row = K, .col=N, .row_major = false};
  mat_spec spec_C = {.row = M, .col=N};

  half **h_A = allocMatrix<half>(spec_A);
  half **h_B = allocMatrix<half>(spec_B);
  half **h_C = allocMatrix<half>(spec_C);
  half *h_Res = (half *) malloc(sizeof(half)*M*N);

  half *d_A = NULL, *d_B = NULL, *d_C = NULL, *d_Res=NULL;

  gpuErrchk(cudaMalloc(&d_A, sizeof(half)*M*K));
  gpuErrchk(cudaMalloc(&d_B, sizeof(half)*K*N));
  gpuErrchk(cudaMalloc(&d_C, sizeof(half)*M*N));
  gpuErrchk(cudaMalloc(&d_Res, sizeof(half)*M*N));


  h_A[0][0] = __float2half(a);
  h_B[0][0] = __float2half(b);

  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(half) * M*N, cudaMemcpyDeviceToHost));
  float res = __half2float(h_Res[0]);

  free(h_Res);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_Res);

  return res;
}
float testRoundingMode_accumulator(const float a, const float b, const float c)
{
  
  int M = 16;
  int N = 16;
  int K = 16;

  mat_spec spec_A = {.row = M, .col=K, .row_major = true};
  mat_spec spec_B = {.row = K, .col=N, .row_major = false};
  mat_spec spec_C = {.row = M, .col=N};

  half **h_A = allocMatrix<half>(spec_A);
  half **h_B = allocMatrix<half>(spec_B);
  float **h_C = allocMatrix<float>(spec_C);
  float *h_Res = (float *) malloc(sizeof(float)*M*N);

  half *d_A = NULL, *d_B = NULL;
  float *d_C = NULL, *d_Res=NULL;

  gpuErrchk(cudaMalloc(&d_A, sizeof(half)*M*K));
  gpuErrchk(cudaMalloc(&d_B, sizeof(half)*K*N));
  gpuErrchk(cudaMalloc(&d_C, sizeof(float)*M*N));
  gpuErrchk(cudaMalloc(&d_Res, sizeof(float)*M*N));


  h_A[0][0] = __float2half(a);
  h_B[0][0] = __float2half(b);
  h_C[0][0] = c;

  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));
  float res = h_Res[0];

  free(h_Res);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_Res);

  return res;
}


void rounding_final_tests()
{
  float expected_RNE, expected_RTP, expected_RTZ, expected_RTN;
  float a, b, res;

  a = ldexpf(1.3330078125f, -13);
  b = ldexpf(-1.5f, -1);
  expected_RNE = ldexpf(-1.0f, -13);
  expected_RTP = ldexpf(-1.9990234375f, -14);
  res = testRoundingMode_final(a, b);
  printRoundingTestResult("RNE", expected_RNE, "RTP", expected_RTP, res);

  a = ldexpf(1.3330078125f, -9);
  b = ldexpf(1.5f, -1);
  expected_RNE = ldexpf(1.0f, -9);
  expected_RTZ = ldexpf(1.9990234375f, -10);
  res = testRoundingMode_final(a, b);
  printRoundingTestResult("RNE", expected_RNE, "RTZ", expected_RTZ, res);

  // RNE vs RTN

  a = ldexpf(0.96191406f, -14);
  b = ldexpf(-1.7744140625f, -9);
  expected_RNE = ldexpf(-0.0029296875f, -14);
  expected_RTN = ldexpf(-0.00390625f, -14);
  res = testRoundingMode_final(a, b);
  printRoundingTestResult("RNE", expected_RNE, "RTN", expected_RTN, res);

  // RTP vs RTZ
  a = ldexp(1.9912109375f, -7);
  b = ldexp(1.4306640625f, -1);
  expected_RTP = ldexpf(1.4248046875f, -7);
  expected_RTZ = ldexpf(1.423828125f, -7);
  res = testRoundingMode_final(a, b);
  printRoundingTestResult("RTP", expected_RTP, "RTZ", expected_RTZ, res);

  // RTP vs RTN

  a = ldexp(1.0009765625f, -15);
  b = -1.931640625f;
  expected_RTP = ldexpf(-1.0009765625f, -15);
  expected_RTN = ldexpf(-1.001953125f, -15);
  res = testRoundingMode_final(a, b);
  printRoundingTestResult("RTP", expected_RTP, "RTN", expected_RTN, res);

  a = ldexp(1.0009765625f, -15);
  b = -1.931640625f;
  expected_RTZ = ldexpf(-1.0009765625f, -15);
  expected_RTN = ldexpf(-1.001953125f, -15);
  res = testRoundingMode_final(a, b);
  printRoundingTestResult("RTZ", expected_RTZ, "RTN", expected_RTN, res);


}

void accumulation_order_tests() {
  int M = 16;
  int N = 16;
  int K = 16;

  mat_spec spec_A = {.row = M, .col=K, .row_major = true};
  mat_spec spec_B = {.row = K, .col=N, .row_major = false};
  mat_spec spec_C = {.row = M, .col=N};

  half **h_A = allocMatrix<half>(spec_A);
  half **h_B = allocMatrix<half>(spec_B);
  float **h_C = allocMatrix<float>(spec_C);
  float *h_Res = (float *) malloc(sizeof(float)*M*N);

  half *d_A = NULL, *d_B = NULL;
  float *d_C = NULL, *d_Res=NULL;

  gpuErrchk(cudaMalloc(&d_A, sizeof(half)*M*K));
  gpuErrchk(cudaMalloc(&d_B, sizeof(half)*K*N));
  gpuErrchk(cudaMalloc(&d_C, sizeof(float)*M*N));
  gpuErrchk(cudaMalloc(&d_Res, sizeof(float)*M*N));

  printf("\n======= Associativity Test =======\n");
  float a1 = ldexpf(1.9990234375f, -9);
  float a2 = ldexpf(1.9990234375f, -1);
  float b1 = ldexpf(1.9990234375f, -1);
  float b2 = ldexpf(1.9990234375f, -1);
  float c = ldexpf(1.0021368265151978f, 15);

  h_A[0][0] = __float2half(a1);
  h_A[0][1] = __float2half(a2);
  h_B[0][0] = __float2half(b1);
  h_B[0][1] = __float2half(b2);
  h_C[0][0] = c;

  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));
  float res = h_Res[0];

  free(h_Res);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_Res);

  printRoundingTestResult("(a1b1 + a2b2) + c", ldexpf(1.0021673440933228f, 15), "a1b1 + (a2b2 + c)", ldexpf(1.0021672248840332, 15), res);

}

void rounding_accumulator_tests(){
  float expected_RNE, expected_RTP, expected_RTZ, expected_RTN;
  // RNE vs RTP
  float a, b, c, res;
  // effective subtraction
  printf("\n======= RNE vs RTP =======\n");
  a = ldexpf(1.5f, 15);
  b = ldexpf(1.5f, 14);
  c = ldexpf(-1.0029296875f, 15);
  expected_RNE = ldexpf(1.1249693632125854f, 30);
  expected_RTP = ldexpf(1.124969482421875f, 30);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RNE", expected_RNE, "RTP", expected_RTP, res);

  // effective addition
  a = ldexpf(1.0302734375f, 15);
  b = ldexpf(1.748046875f, 15);
  c = ldexpf(1.5009765625f, 8);
  expected_RNE = ldexpf(1.8009666204452515f, 30);
  expected_RTP = ldexpf(1.800966739654541f, 30);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RNE", expected_RNE, "RTP", expected_RTP, res);


  // RNE vs RTZ
  printf("\n======= RNE vs RTZ =======\n");

  // effective subtraction
  expected_RNE = ldexpf(-1.9695377349853516f, 30);
  expected_RTZ = ldexpf(-1.969537615776062f, 30);
  a = ldexpf(1.767578125f, 15);
  b = ldexpf(-1.1142578125f, 15);
  c = ldexpf(1.6435546875f, 2);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RNE", expected_RNE, "RTZ", expected_RTZ, res);

  // effective addition
  a = ldexpf(1.7822265625f, -3);
  b = ldexpf(1.0048828125f, -15);
  c = ldexpf(1.9951171875f, -8);
  expected_RNE = ldexpf(1.9951342344284058f, -8);
  expected_RTZ = ldexpf(1.9951341152191162f, -8);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RNE", expected_RNE, "RTZ", expected_RTZ, res);


  // RNE vs RTN
  // effective subtraction
  printf("\n======= RNE vs RTN =======\n");
  a = ldexpf(1.7060546875f, 15);
  b = ldexpf(-1.7646484375f, 13);
  c = ldexpf(1.9970703125f, 8);
  expected_RNE = ldexpf(-1.5052924156188965f, 29);
  expected_RTN = ldexpf(-1.505292534828186f, 29);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RNE", expected_RNE, "RTN", expected_RTN, res);

  // RTP vs RTZ
  // effective subtraction
  printf("\n======= RTP vs RTZ =======\n");
  expected_RTP = 1.0f;
  expected_RTZ = ldexp(1.9999998807907104f, 1);
  a = ldexpf(1.25f, -15);
  b = ldexpf(-1.25f, -15);
  c = 1.0f;
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RTP", expected_RTP, "RTZ", expected_RTZ, res);

  // effective addition
  a = ldexpf(1.7822265625f, -3);
  b = ldexpf(1.0048828125f, -15);
  c = ldexpf(1.9951171875f, -8);
  expected_RTP = ldexpf(1.9951342344284058f, -8);
  expected_RTZ = ldexpf(1.9951341152191162f, -8);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RTP", expected_RTP, "RTZ", expected_RTZ, res);


  // RTP vs RTN
  printf("\n======= RTP vs RTN =======\n");
  //effective subtraction
  a = ldexpf(1.7939453125f, -12);
  b = ldexpf(-1.9619140625f, 15);
  c = ldexpf(1.4921875f, -15);
  expected_RTP = ldexpf(-1.7597813606262207f, 4);
  expected_RTN = ldexpf(-1.7597814798355103f, 4);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RTP", expected_RTP, "RTN", expected_RTN, res);


  // RTZ vs RTN
  // effective subtraction
  printf("\n======= RTZ vs RTN =======\n");
  expected_RTZ = ldexpf(-1.9999980926513672f, 12);
  expected_RTN = ldexpf(-1.9999982118606567f, 12);
  a = ldexpf(1.0f, 1);
  b = ldexpf(1.9990234375f, -9);
  c = ldexpf(-1.0f, 13);
  res = testRoundingMode_accumulator(a, b, c);
  printRoundingTestResult("RTZ", expected_RTZ, "RTN", expected_RTN, res);
  // (RTZ is never different from RTN for positive inputs)



}

template <typename V, typename T, typename U>
T exceptional_test(const U a, const U b, const T c, const bool do_print=false)
{

  int M = 16;
  int N = 16;
  int K = 16;

  mat_spec spec_A = {.row = M, .col=K, .row_major = true};
  mat_spec spec_B = {.row = K, .col=N, .row_major = false};
  mat_spec spec_C = {.row = M, .col=N};

  half **h_A = allocMatrix<half>(spec_A);
  half **h_B = allocMatrix<half>(spec_B);
  V **h_C = allocMatrix<V>(spec_C);

  half *d_A = NULL, *d_B = NULL;
  V *d_C = NULL, *d_Res=NULL;
  V *h_Res = (V *) malloc(sizeof(V)*M*N);

  cudaMalloc(&d_A, sizeof(half)*M*K);
  cudaMalloc(&d_B, sizeof(half)*K*N);
  cudaMalloc(&d_C, sizeof(V)*M*N);
  cudaMalloc(&d_Res, sizeof(V)*M*N);

  memcpy(&h_A[0][0], &a, sizeof(half));
  memcpy(&h_B[0][0], &b, sizeof(half));
  memcpy(&h_C[0][0], &c, sizeof(T));

  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);

  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(V) * M*N, cudaMemcpyDeviceToHost));
  T res;
  memcpy(&res, &h_Res[0], sizeof(V));
  if (do_print)
    printResultBits(res);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Res);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return res;
}

/** Tests the behavior of NaN payloads
 * Performs tests in single-precision and half-precision to
 * determine if NaN payloads and signs are propagated on hardware
 * Exhaustive tests are enabled by setting -DEXHAUSTIVE_NAN_TEST=1
 * during compilation
 */
void nan_behavior()
{
  printf("\n[Test NaN payloads].........\n");
  /* exhaustively test all NaN payloads */
  uint32_t result;
  uint16_t result_16;
  bool any_not_same_nan_16 = false;
  bool any_not_same_nan = false;

  if (EXHAUSTIVE_NAN_TEST)
  {
    uint16_t c_16 = 0x7C00u;
    uint16_t input_16;
    uint16_t result_16;
    printf("Exhaustively mode (this will take a long time).......\n");
    /* half-precision */
    for(uint16_t i = 1; i <= 0x03FFu; ++i)
    {

      input_16 = c_16 | i;
      result_16 = exceptional_test<half, uint16_t, half>(__float2half(0.0f), __float2half(0.0f), input_16);
      if (result_16 != 0x7fff)
      {
        printf("NaN payload differs for: %#010x, it is: %#010x\n", input_16, result_16);
        any_not_same_nan_16 = true;
      }
      input_16 |= 0x8000;
      result_16 = exceptional_test<half, uint16_t, half>(__float2half(0.0f), __float2half(0.0f), input_16);
      if (result_16 != 0x7fff)
      {
        printf("NaN payload differs for: %#010x, it is: %#010x\n", input_16, result_16);
        any_not_same_nan_16 = true;
      }
    }
    printf("[Test NaN payload is propagated in f16]......... [%s]\n", any_not_same_nan_16 ? "true" : "false");
    /* single-precision */
    any_not_same_nan = false;
    uint32_t c = 0x7F800000u;
    uint32_t input;
    for(uint32_t i = 1; i <= 0x007FFFFFu; ++i)
    {

      input = c | i;
      result = exceptional_test<float, uint32_t, half>(__float2half(0.0f), __float2half(0.0f), input);
      if (result != 0x7fffffff)
      {
        printf("NaN payload differs for: %#010x, it is: %#010x\n", input, result);
        any_not_same_nan = true;
      }
      input |= 0x80000000;
      result = exceptional_test<float, uint32_t, half>(__float2half(0.0f), __float2half(0.0f), input);
      if (result != 0x7fffffff)
      {
        printf("NaN payload differs for: %#010x, it is: %#010x\n", input, result);
        any_not_same_nan = true;
      }
    }
    printf("[Test NaN payload is propagated in f32]......... [%s]\n", any_not_same_nan ? "true" : "false");
  }
  else {
    printf("Test 1\n");
    /* quiet nan */
    result = exceptional_test<float, uint32_t, half>(__float2half(0.0f), __float2half(0.0f), (uint32_t) 0x7FC00000);
    printf("Test 2\n");
    result_16 = exceptional_test<half, uint16_t, half>(__float2half(0.0f), __float2half(0.0f), (uint16_t) 0x7E00u);
    printf("Test 3\n");
    if (result != 0x7fffffffu)
    {
      printf("NaN payload differs for qNaN in single-precision, resulting in %#010x\n", result);
      any_not_same_nan = true;
    }
    if (result_16 != 0x7fffu)
    {
      printf("NaN payload differs for qNaN in half-precision, resulting in %#06x\n", result_16);
      any_not_same_nan_16 = true;
    }
    /* signaling nan */
    result = exceptional_test<float, uint32_t, half>(__float2half(0.0f), __float2half(0.0f), (uint32_t) 0x7FE00000);
    result_16 = exceptional_test<half, uint16_t, half>(__float2half(0.0f), __float2half(0.0f), (uint16_t) 0x7F00u);
    if (result != 0x7fffffffu)
    {
      printf("NaN payload differs for sNaN in single-precision, resulting in %#010x\n", result);
      any_not_same_nan = true;
    }
    if (result_16 != 0x7fffu)
    {
      printf("NaN payload differs for sNaN in half-precision, resulting in %#06x\n", result_16);
      any_not_same_nan_16 = true;
    }


    /* signed quiet nan */
    result = exceptional_test<float, uint32_t, half>(__float2half(0.0f), __float2half(0.0f), (uint32_t) 0xFFC00000);
    result_16 = exceptional_test<half, uint16_t, half>(__float2half(0.0f), __float2half(0.0f), (uint16_t) 0xFE00u);
    if (result != 0x7fffffffu)
    {
      printf("NaN payload differs for qNaN in single-precision, resulting in %#010x\n", result);
      any_not_same_nan = true;
    }
    if (result_16 != 0x7fffu)
    {
      printf("NaN payload differs for qNaN in half-precision, resulting in %#06x\n", result_16);
      any_not_same_nan_16 = true;
    }
    /* signed signaling nan */
    result = exceptional_test<float, uint32_t, half>(__float2half(0.0f), __float2half(0.0f), (uint32_t) 0xFFE00000);
    result_16 = exceptional_test<half, uint16_t, half>(__float2half(0.0f), __float2half(0.0f), (uint16_t) 0xFF00u);
    if (result != 0x7fffffffu)
    {
      printf("NaN payload differs for sNaN in single-precision, resulting in %#010x\n", result);
      any_not_same_nan = true;
    }
    if (result_16 != 0x7fffu)
    {
      printf("NaN payload differs for sNaN in half-precision, resulting in %#06x\n", result_16);
      any_not_same_nan_16 = true;
    }
    printf("[Test NaN payload is propagated in f16]......... [%s]\n", any_not_same_nan_16 ? "true" : "false");
    printf("[Test NaN payload is propagated in f32]......... [%s]\n", any_not_same_nan ? "true" : "false");
  }


  // Now, check if nan results are correctly computed

  // Multiplication between infinities and zeros
  printf("\n\n[Test NaN results, single-precision].........\n");
  printf("[Test 0 times +oo]\n");
  exceptional_test<float, float, half>(__float2half(0.0f), __float2half(+INFINITY), 0.0f, true);
  printf("[Test +oo times 0]\n");
  exceptional_test<float, float, half>(__float2half(+INFINITY), __float2half(0.0f), 0.0f, true);
  printf("[Test -0 times +oo]\n");
  exceptional_test<float, float, half>(__float2half(-0.0f), __float2half(+INFINITY), 0.0f, true);

  printf("\n\n[Signed Zero].........\n");
  printf("[1.0 + -1.0]\n");
  exceptional_test<float, float, half>(__float2half(1.0f), __float2half(1.0f), -1.0f, true);
  printf("[-0.0*0.0 + -0.0]\n");
  exceptional_test<float, float, half>(__float2half(-0.0f), __float2half(0.0f), -0.0f, true);

  printf("Test signed zero in rounded result\n");
  exceptional_test<half, half, uint16_t>(__float2half(ldexp(-1.0f, -15)), __float2half(ldexp(1.0f, -15)), __float2half(0.0f), true);

}
void behavior()
{
  int M = 16;
  int N = 16;
  int K = 16;

  mat_spec spec_A = {.row = M, .col=K, .row_major = true};
  mat_spec spec_B = {.row = K, .col=N, .row_major = false};
  mat_spec spec_C = {.row = M, .col=N};
  mat_spec spec_C_half = {.row = M, .col=N};

  half **h_A = allocMatrix<half>(spec_A);
  half **h_B = allocMatrix<half>(spec_B);
  float **h_C = allocMatrix<float>(spec_C);
  half **h_C_half = allocMatrix<half>(spec_C_half);

  /* Exact Multiplication */

  float expected_result;

  h_A[0][0] = __float2half(ldexp(1.7177734375f, -7));
  h_B[0][0] = __float2half(ldexp(-1.62109375f, -7));

  expected_result = ldexp(-1.3923358917236328f, -13);

  half *d_A = NULL, *d_B = NULL, *d_C_half = NULL, *d_Res_half = NULL;
  float *d_C = NULL, *d_Res=NULL;

  cudaMalloc(&d_A, sizeof(half)*M*K);
  cudaMalloc(&d_B, sizeof(half)*K*N);
  cudaMalloc(&d_C, sizeof(float)*M*N);
  cudaMalloc(&d_C_half, sizeof(half)*M*N);
  cudaMalloc(&d_Res, sizeof(float)*M*N);
  cudaMalloc(&d_Res_half, sizeof(half)*M*N);

  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);

  /* Exact Accumulation when C is fp16 */

  float *h_Res = (float *) malloc(sizeof(float)*M*N);
  /*
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));

  if (h_Res[0] == expected_result) {
    printf("[Exact Multiplication]...... [Yes]\n");
  } else {
    printf("[Exact Multiplication]...... [No]\n");
  }
  */

  half *h_Res_half = (half *) malloc(sizeof(half)*M*N);

  /*

  expected_result = ldexp(-1.8076171875f, 5);
  h_A[0][0] = __float2half(ldexp(-1.5205078125f, 2));
  h_A[0][1] = __float2half(ldexp(1.04296875f, -4));
  
  h_B[0][0] = __float2half(-1.25f);
  h_B[0][1] = __float2half(ldexp(-1.9609375, 9));

  copyMatrix(h_A, h_B, h_C_half, d_A, d_B, d_C_half, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C_half, d_Res_half);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res_half, d_Res_half, sizeof(half) * M*N, cudaMemcpyDeviceToHost));

  if (__half2float(h_Res_half[0]) == expected_result)
  {
    printf("[Exact Accumulation]...... [Yes]\n");
  } else {
    printf("[Exact Accumulation]...... [No]\n");
  }
  */

  printf("[Test extra significand bit]");
  h_A[0][0] = __float2half(ldexpf(1.0, -12));
  h_A[0][1] = __float2half(ldexpf(1.0, -12));
  h_B[0][0] = __float2half(ldexpf(1.0, -12));
  h_B[0][0] = __float2half(ldexpf(1.0, -12));
  h_C[0][0] = 1.0f;
  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));
  printf("Result for extra significand bit: %x\n", (w32_un) {.f = h_Res[0]}.u);

  return;


  printf("\n\n[Test Rounding Result to FP16]......\n");
  rounding_final_tests();

  printf("\n\n[Test Rounding of Accumulator]......\n");
  rounding_accumulator_tests();
  // RNE vs RTP


  printf("[Test Subnormals]......\n");
  h_A[0][0] = __float2half(1.0f);
  h_A[0][1] = __float2half(1.0f);
  h_B[0][0] = __float2half(-1.0f);
  h_B[0][1] = __float2half(1.0f);
  h_C[0][0] = ((w32_un) {.u = 0x007fffff}).f;

  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));

  printf("Accumulation result when c is subnormal and others are 0: %x\n", (w32_un) {.f = h_Res[0]}.u);

  h_A[0][0] = __float2half(0.0f);
  h_A[0][1] = __float2half(0.0f);
  h_B[0][0] = __float2half(0.0f);
  h_B[0][1] = __float2half(0.0f);
  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));
  if (h_Res[0] == ldexp(1.0f, -127)) {
    printf("[Test Result can be subnormal]......\n");
  }


  h_A[0][0] = __float2half(1.0f);
  h_A[0][1] = __float2half(1.0f);
  h_B[0][0] = __float2half(1.0f);
  h_B[0][1] = __float2half(-1.0f);
  copyMatrix(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
  wmma_ker_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_Res);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_Res, d_Res, sizeof(float) * M*N, cudaMemcpyDeviceToHost));
  printf("Accumulation result when c is subnormal and others add to 0: %x\n", (w32_un) {.f = h_Res[0]}.u);


  printf("[Test Accumulation Order]......\n");


  printf("Test exceptional values].....\n");

  nan_behavior();

  accumulation_order_tests();


  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_half);
  free(h_Res);
  free(h_Res_half);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_half);

}

int main(int argc, char* argv[])
{
	cudaError_t cuda_status;
  int devCount;
  cudaGetDeviceCount(&devCount);
  int selected_device = -1;
  int major = 7;
  int minor = 0;
  if (argc == 2)
  {
    int version = atoi(argv[1]);
    switch (version)
    {
      case 70:
        major = 7;
        minor = 0;
        printf("Running experiments on sm_70\n");
        break;
      case 75:
        major = 7;
        minor = 5;
        printf("Running experiments on sm_75\n");
        break;
      case 80:
        major = 8;
        minor = 0;
        break;
      default:
        printf("Unsupported version: sm_%d\n", version); 
        return EXIT_FAILURE;
    }

  } else if (argc > 2)
  {
    printf("Usage: %s [sm version (defaut=70)]\n", argv[0]);
  }
  for(int i = 0; i < devCount; ++i)
  {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    if (devProp.major == major && devProp.minor == minor)
    {
      selected_device = i;
      printf("Using device with sm_%d%d: %s\n", major, minor, devProp.name);
      break;
    }
  }
  if (selected_device == -1)
  {
    printf("Could not find installed device for sm_%d%d\n", major, minor);
    return EXIT_FAILURE;
  }
  // set to the volta GPU.  This is device 0
	cuda_status = cudaSetDevice(selected_device);
  gpuErrchk(cuda_status);

  behavior();

  return EXIT_SUCCESS;
}
