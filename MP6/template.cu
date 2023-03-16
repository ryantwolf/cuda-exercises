// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128

//@@ insert code here
__global__ void float_to_char(unsigned char *out, float *in, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = (unsigned char)(255 * in[i]);
    }
}

__global__ void rgb_to_gray(unsigned char *out, unsigned char *in, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = (unsigned char)(0.21 * in[3 * i] + 0.71 * in[3 * i + 1] + 0.07 * in[3 * i + 2]);
    }
}

__global__ void compute_histogram(unsigned int *hist, unsigned char *in, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        atomicAdd(&hist[in[i]], 1);
    }
}

__global__ void scan(float *output, unsigned int *input, int len, int width, int height) {
  __shared__ float T[HISTOGRAM_LENGTH];

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
  while (stride < HISTOGRAM_LENGTH) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < HISTOGRAM_LENGTH && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }

  stride = HISTOGRAM_LENGTH;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < HISTOGRAM_LENGTH) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();
  if (first_load < len) {
    output[first_load] = T[threadIdx.x] / (width * height);
  }
  if (second_load < len) {
    output[second_load] = T[blockDim.x + threadIdx.x] / (width * height);
  }
}

__global__ void find_min(float *output, float *input) {
  __shared__ float T[HISTOGRAM_LENGTH];
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    T[threadIdx.x] = input[threadIdx.x];
  } else {
    T[threadIdx.x] = 1;
  }
  for (int stride = HISTOGRAM_LENGTH / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      T[threadIdx.x] = min(T[threadIdx.x], T[threadIdx.x + stride]);
    }
  }
  if (threadIdx.x == 0) {
    *output = T[0];
  }
}

__global__ void equalize(unsigned char* output, unsigned char* input, float* cdf, float* min_val, int len) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len) {
    output[i] = min(max(255 * (cdf[input[i]] - *min_val) / (1.0 - *min_val), 0.0), 255.0);
  }
}

__global__ void char_to_float(float *out, unsigned char *in, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = (float)in[i] / 255.0;
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceCharImageData;
  unsigned char *deviceGrayImageData;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  float *deviceMinCDF;
  unsigned char *equalizedImageChar;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayImageData, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceMinCDF, sizeof(float));
  cudaMalloc((void **)&equalizedImageChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(ceil(1.0 * imageWidth * imageHeight / BLOCK_SIZE), 1, 1);
  dim3 dimLargeGrid(ceil(1.0 * imageWidth * imageHeight * imageChannels / BLOCK_SIZE), 1, 1);

  float_to_char<<<dimLargeGrid, dimBlock>>>(deviceCharImageData, deviceInputImageData, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  rgb_to_gray<<<dimGrid, dimBlock>>>(deviceGrayImageData, deviceCharImageData, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  compute_histogram<<<dimGrid, dimBlock>>>(deviceHistogram, deviceGrayImageData, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  scan<<<1, HISTOGRAM_LENGTH / 2>>>(deviceCDF, deviceHistogram, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  find_min<<<1, HISTOGRAM_LENGTH / 2>>>(deviceMinCDF, deviceCDF);
  cudaDeviceSynchronize();

  equalize<<<dimLargeGrid, dimBlock>>>(equalizedImageChar, deviceCharImageData, deviceCDF, deviceMinCDF, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  char_to_float<<<dimLargeGrid, dimBlock>>>(deviceOutputImageData, equalizedImageChar, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceCharImageData);
  cudaFree(deviceGrayImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  cudaFree(deviceMinCDF);
  cudaFree(equalizedImageChar);

  return 0;
}
