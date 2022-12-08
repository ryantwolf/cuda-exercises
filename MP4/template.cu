#include "wb.h"
#include <stdio.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 13
#define KERNEL_WIDTH 3
#define BLOCK_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)

//@@ Define constant memory for device kernel here
__constant__ float M[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];
  // Calculate the row, column, and depth index to load into shared memory
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int x_o = bx * TILE_WIDTH + tx;
  int y_o = by * TILE_WIDTH + ty;
  int z_o = bz * TILE_WIDTH + tz;
  int x_i = x_o - (KERNEL_WIDTH / 2);
  int y_i = y_o - (KERNEL_WIDTH / 2);
  int z_i = z_o - (KERNEL_WIDTH / 2);
  // Load the input into shared memory
  if (x_i >= 0 && x_i < x_size && y_i >= 0 && y_i < y_size && z_i >= 0 && z_i < z_size) {
    N_ds[tz][ty][tx] = input[z_i * y_size * x_size + y_i * x_size + x_i];
  } else {
    N_ds[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  // Calculate the convolution if the output index is within the tile
  float Pvalue = 0.0f;
  if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH) {
    for (int k = 0; k < KERNEL_WIDTH; ++k) {
      for (int j = 0; j < KERNEL_WIDTH; ++j) {
        for (int i = 0; i < KERNEL_WIDTH; ++i) {
          Pvalue += N_ds[tz + k][ty + j][tx + i] * M[k][j][i];
        }
      }
    }

    // Store the convolution result in global memory
    if (x_o < x_size && y_o < y_size && z_o < z_size) {
      output[z_o * y_size * x_size + y_o * x_size + x_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M, hostKernel, kernelLength * sizeof(float));
  // Print the input
  // for (int i = 0; i < inputLength - 3; ++i) {
  //   printf("%f ", hostInput[i + 3]);
  // }
  // printf("\n");
  
  // Print the kernel
  // for (int k = 0; k < KERNEL_WIDTH; ++k) {
  //   for (int j = 0; j < KERNEL_WIDTH; ++j) {
  //     for (int i = 0; i < KERNEL_WIDTH; ++i) {
  //       printf("%f ", hostKernel[k * KERNEL_WIDTH * KERNEL_WIDTH + j * KERNEL_WIDTH + i]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size / (1.0*TILE_WIDTH)), ceil(y_size / (1.0*TILE_WIDTH)),
               ceil(z_size / (1.0*TILE_WIDTH)));
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float),
             cudaMemcpyDeviceToHost);
  // Print the output
  // for (int i = 0; i < inputLength - 3; ++i) {
  //   printf("%f ", hostOutput[i + 3]);
  // }
  // printf("\n");
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
