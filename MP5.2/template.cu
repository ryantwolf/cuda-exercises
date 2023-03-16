// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void addAux(float *main, float *aux, int len) {
  int first = 2 * (blockIdx.x + 1) * BLOCK_SIZE + threadIdx.x;
  int second = first + BLOCK_SIZE;
  if (first < len) {
    main[first] += aux[blockIdx.x];
  }
  if (second < len) {
    main[second] += aux[blockIdx.x];
  }
}

__global__ void scan(float *input, float *output, float* aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];

  int first_load = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (first_load < len) {
    T[threadIdx.x] = input[first_load];
  } else {
    T[threadIdx.x] = 0;
  }

  int second_load = 2 * blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
  if (second_load < len) {
    T[blockDim.x + threadIdx.x] = input[second_load];
  } else {
    T[blockDim.x + threadIdx.x] = 0;
  }

  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }

  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();
  if (first_load < len) {
    output[first_load] = T[threadIdx.x];
  }
  if (second_load < len) {
    output[second_load] = T[blockDim.x + threadIdx.x];
  }

  if (threadIdx.x == BLOCK_SIZE - 1 && aux != NULL) {
    aux[blockIdx.x] = T[2 * BLOCK_SIZE - 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // Create aux block sum array
  float *aux_block_sum;
  cudaMalloc((void **)&aux_block_sum, DimGrid.x * sizeof(float));
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, aux_block_sum, numElements);
  scan<<<dim3(1, 1, 1), DimBlock>>>(aux_block_sum, aux_block_sum, NULL, DimGrid.x);
  addAux<<<DimGrid, DimBlock>>>(deviceOutput, aux_block_sum, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  // Print output
  // for (int i = 0; i < numElements; i++) {
  //   printf("%f ", hostOutput[i]);
  // }
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(aux_block_sum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
