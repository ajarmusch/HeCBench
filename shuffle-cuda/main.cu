/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <iostream>
#include <cuda.h>

#define BUF_SIZE 256
#define PATTERN 0xDEADBEEF

#define CUDACHECK(code)                                                         \
  do {                                                                         \
    cudaerr = code;                                                             \
    if (cudaerr != cudaSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)cudaerr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, 
    unsigned int numGroups, unsigned int subGroupSize) {
  for (unsigned i = 0; i < numGroups; ++i) {
    for (unsigned j = 0; j < subGroupSize; j++) {
      output[i * subGroupSize + j] = input[i * subGroupSize + subGroupSize - j - 1];
    }
  }
}

void verifyBroadcast(const int *out, const int subGroupSize, int pattern = 0)
{
  int expected = pattern;
  if (pattern == 0) {
    for (int i = 0; i < subGroupSize; i++) 
      expected += i;
  }
  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != expected) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
      break;
    }
  }
  if (errors == 0)
    std::cout << "PASSED\n";
}

void verifyTransposeMatrix(const float *TransposeMatrix, const float* cpuTransposeMatrix, 
            const int total, const int subGroupSize)
{
  int errors = 0;
  float eps = 1.0E-6;
  for (int i = 0; i < total; i++) {
    if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ITEM: " << i <<
        " cpu: " << cpuTransposeMatrix[i] <<
        " gpu: " << TransposeMatrix[i] << "\n";
      errors++;
      break;
    }
  }
  std::cout << "PASSED\n";
}


//==================================================================================
// Broadcast
//==================================================================================
__global__ void bcast_shfl_sg8(const int arg, int *out) {
  int value = ((threadIdx.x & 0x7) == 0) ? arg : 0;
  // Synchronize all threads in warp, and get "value" from lane 0
  int out_v = __shfl( value, 0); 
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

__global__ void bcast_shfl_xor_sg8(int *out) {
  int value = (threadIdx.x & 0x7);
  for (int mask = 1; mask < 0x7; mask *= 2)
    value += __shfl_xor(value, mask);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = value;
}

__global__ void bcast_shfl_sg16(const int arg, int *out) {
  int value = ((threadIdx.x & 0xf) == 0) ? arg : 0;
  // Synchronize all threads in warp, and get "value" from lane 0
  int out_v = __shfl( value, 0); 
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

__global__ void bcast_shfl_xor_sg16(int *out) {
  int value = (threadIdx.x & 0xf);
  for (int mask = 1; mask < 0xf; mask *= 2)
    value += __shfl_xor(value, mask);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = value;
}
__global__ void bcast_shfl_sg32(const int arg, int *out) {
  int value = ((threadIdx.x & 0x1f) == 0) ? arg : 0;
  // Synchronize all threads in warp, and get "value" from lane 0
  int out_v = __shfl( value, 0); 
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

__global__ void bcast_shfl_xor_sg32(int *out) {
  int value = (threadIdx.x & 0x1f);
  for (int mask = 1; mask < 0x1f; mask *= 2)
    value += __shfl_xor(value, mask);
  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = value;
}
//==================================================================================
// Matrix transpose
//==================================================================================
__global__ void transpose_shfl(float* out, const float* in) {
  unsigned b_start = blockDim.x * blockIdx.x;
  unsigned b_offs = b_start + threadIdx.x;
  unsigned s_offs = blockDim.x - threadIdx.x - 1;
  float val = in[b_offs];
  out[b_offs] = __shfl(val, s_offs);
}


int main() {

  std::cout << "Broadcast using shuffle functions\n";

  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);
  int *d_out;
  cudaMalloc((void **)&d_out, sizeof(int) * BUF_SIZE);

  std::cout << "Broadcast using the shuffle xor function (subgroup sizes 8, 16, and 32) \n";
  for (int n = 0; n < 100; n++)
    bcast_shfl_xor_sg8 <<< dim3(1), dim3(BUF_SIZE) >>> (d_out);
  cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost);
  verifyBroadcast(out, 8);

  for (int n = 0; n < 100; n++)
    bcast_shfl_xor_sg16 <<< dim3(1), dim3(BUF_SIZE) >>> (d_out);
  cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost);
  verifyBroadcast(out, 16);

  for (int n = 0; n < 100; n++)
    bcast_shfl_xor_sg32 <<< dim3(1), dim3(BUF_SIZE) >>> (d_out);
  cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost);
  verifyBroadcast(out, 32);

  std::cout << "Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) \n";

  for (int n = 0; n < 100; n++)
    bcast_shfl_sg8 <<< dim3(1), dim3(BUF_SIZE) >>> (PATTERN, d_out);
  cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost);
  verifyBroadcast(out, 8, PATTERN);

  for (int n = 0; n < 100; n++)
    bcast_shfl_sg16 <<< dim3(1), dim3(BUF_SIZE) >>> (PATTERN, d_out);
  cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost);
  verifyBroadcast(out, 16, PATTERN);

  for (int n = 0; n < 100; n++)
    bcast_shfl_sg32 <<< dim3(1), dim3(BUF_SIZE) >>> (PATTERN, d_out);
  cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost);
  verifyBroadcast(out, 32, PATTERN);

  free(out);
  cudaFree(d_out);

  std::cout << "matrix transpose using the shuffle function (subgroup sizes are 8, 16, and 32)\n";

  const int total = 1 << 27;  // total number of elements in a matrix

  float* Matrix = (float*)malloc(total * sizeof(float));
  float* TransposeMatrix = (float*)malloc(total * sizeof(float));
  float* cpuTransposeMatrix = (float*)malloc(total * sizeof(float));

  // initialize the input data
  for (int i = 0; i < total; i++) {
    Matrix[i] = (float)i * 10.0f;
  }

  float *gpuMatrix;
  float *gpuTransposeMatrix;
  // allocate the memory on the device side
  cudaMalloc((void **)&gpuMatrix, total * sizeof(float));
  cudaMalloc((void **)&gpuTransposeMatrix, total * sizeof(float));

  cudaMemcpy(gpuMatrix, Matrix, total * sizeof(float), cudaMemcpyHostToDevice);

  for (int n = 0; n < 100; n++)
    transpose_shfl <<< dim3(total/8), dim3(8) >>> (gpuTransposeMatrix, gpuMatrix);

  // Memory transfer from device to host
  cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, total * sizeof(float), cudaMemcpyDeviceToHost);
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/8, 8);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 8);

  for (int n = 0; n < 100; n++)
    transpose_shfl <<< dim3(total/16), dim3(16) >>> (gpuTransposeMatrix, gpuMatrix);

  // Memory transfer from device to host
  cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, total * sizeof(float), cudaMemcpyDeviceToHost);
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/16, 16);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 16);

  for (int n = 0; n < 100; n++)
    transpose_shfl <<< dim3(total/32), dim3(32) >>> (gpuTransposeMatrix, gpuMatrix);

  // Memory transfer from device to host
  cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, total * sizeof(float), cudaMemcpyDeviceToHost);
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/32, 32);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 32);


  // free the resources
  cudaFree(gpuMatrix);
  cudaFree(gpuTransposeMatrix);

  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return 0;
}
