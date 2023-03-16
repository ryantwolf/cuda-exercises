#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"

// __constant__ float cMask[3136];

#define STREAMS

#define ROWS_A 64
#define COLS_B 32
#define ROWS_B (ROWS_A / COLS_B)

/*
    REGISTER TILED WITH THE MASK MATRIX AS REGISTERS
    This kernel implements:
     - An advanced matrix multiplication algorithm (register-tiled, for example) (5 points)
*/
__global__ void mask_conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // The output matrix will be size Map_out x (Height_out * Width_out)
    // - In the first layer, it is 4 x 6400
    // - In the second layer, it is 16 x 1156
    // The first input matrix will be size Map_out x (K * K * Channel) (M x L)
    // - In the first layer, it is 4 x 49
    // - In the second layer, it is 16 x 196
    // The second input matrix will be size (K * K * Channel) x (Height_out * Width_out) (L x N)
    // - In the first layer, it is 49 x 6400
    // - In the second layer, it is 196 x 1156
    // blockIdx.z will be the batch index
    // blockIdx.x and blockIdx.y will be the normal tiling of the output matrix
    // threadIdx.x will be the normal tiling of the output matrix within a tile

    const unsigned int M = Map_out;
    const unsigned int N = Height_out * Width_out;
    const unsigned int L = K * K * Channel;

    __shared__ __half2 sB[ROWS_B / 2][COLS_B];

    // The row is also the output feature map
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * COLS_B;

    // Create registers to hold the output values
    __half2 rC[COLS_B];

    // Initialize the output values
    for (int i = 0; i < COLS_B; i++) {
        rC[i] = __float2half2_rn(0.0f);
    }

    // Loop over the input matrices
    for (unsigned int tile = 0; tile < (L - 1) / ROWS_B + 1; tile++) {

        if (threadIdx.x < ROWS_A / 2) {
            // Load the second matrix into shared memory (unrolled)
            unsigned int i = 2 * threadIdx.x / COLS_B;
            unsigned int j = (2 * threadIdx.x) % COLS_B;

            // Load the first float
            float first = 0.0f, second = 0.0f;

            unsigned int unrolled_i = tile * ROWS_B + i;
            unsigned int unrolled_j = col + j;
            unsigned int h = unrolled_j / Width_out;
            unsigned int w = unrolled_j % Width_out;
            if (unrolled_i < L && unrolled_j < N) {
                // Figure out which feature we are loading
                unsigned int feature = unrolled_i / (K * K);
                // Figure out which row and column we are loading
                unsigned int p = (unrolled_i % (K * K)) / K;
                unsigned int q = (unrolled_i % (K * K)) % K;
                first = in_4d(blockIdx.z, feature, h + p, w + q);
            }

            // Load the second float
            unrolled_i = tile * ROWS_B + i + 1;
            if (unrolled_i < L && unrolled_j < N) {
                // Figure out which feature we are loading
                unsigned int feature = unrolled_i / (K * K);
                // Figure out which row and column we are loading
                unsigned int p = (unrolled_i % (K * K)) / K;
                unsigned int q = (unrolled_i % (K * K)) % K;
                second = in_4d(blockIdx.z, feature, h + p, w + q);
            }

            sB[i / 2][j] = __floats2half2_rn(first, second);
        }

        __syncthreads();
        for (unsigned int idx = 0; idx < ROWS_B / 2; idx++) {
            // Load the first matrix into registers
            // TODO: See if we can get by just using the constant memory with masks
            __half2 rA;
            float first = 0.0f, second = 0.0f;
            if (row < M && tile * ROWS_B + 2 * idx < L) {
                first = mask[row * L + tile * ROWS_B + 2 * idx];
            }
            if (row < M && tile * ROWS_B + 2 * idx + 1 < L) {
                second = mask[row * L + tile * ROWS_B + 2 * idx + 1];
            }
            rA = __floats2half2_rn(first, second);

            // Multiply and accumulate
            for (unsigned int out_idx = 0; out_idx < COLS_B; out_idx++) {
                rC[out_idx] = __hfma2(rA, sB[idx][out_idx], rC[out_idx]);
            }
        }
        __syncthreads();
    }

    // Store the results
    for (unsigned int out_idx = 0; out_idx < COLS_B; out_idx++) {
        float2 result = __half22float2(rC[out_idx]);
        unsigned int unrolled_j = col + out_idx;
        if (row < M && unrolled_j < N) {
            out_4d(blockIdx.z, row, unrolled_j / Width_out, unrolled_j % Width_out) = result.x + result.y;
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

#undef ROWS_A
#undef COLS_B
#undef ROWS_B


#define COLS_B 128
#define ROWS_A 16
#define COLS_A (COLS_B / ROWS_A)

/*
    HALF2 REGISTER TILED WITH THE MASK MATRIX AS SHARED MEMORY
    This kernel implements:
     - An advanced matrix multiplication algorithm (register-tiled, for example) (5 points)
*/
__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
        Modify this function to implement the forward pass described in Chapter 16.
        We have added an additional dimension to the tensors to support an entire mini-batch
        The goal here is to be correct AND fast.

        Function paramter definitions:
        output - output
        input - input
        mask - convolution kernel
        Batch - batch_size (number of images in x)
        Map_out - number of output feature maps
        Channel - number of input feature maps
        Height - input height dimension
        Width - input width dimension
        K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // The output matrix will be size Map_out x (Height_out * Width_out)
    // - In the first layer, it is 4 x 6400
    // - In the second layer, it is 16 x 1156
    // The first input matrix will be size Map_out x (K * K * Channel) (M x L)
    // - In the first layer, it is 4 x 49
    // - In the second layer, it is 16 x 196
    // The second input matrix will be size (K * K * Channel) x (Height_out * Width_out) (L x N)
    // - In the first layer, it is 49 x 6400
    // - In the second layer, it is 196 x 1156
    // blockIdx.z will be the batch index
    // blockIdx.x and blockIdx.y will be the normal tiling of the output matrix
    // threadIdx.x will be the normal tiling of the output matrix within a tile

    const unsigned int M = Map_out;
    const unsigned int N = Height_out * Width_out;
    const unsigned int K_squared = K * K;
    const unsigned int L = K_squared * Channel;

    __shared__ __half2 sA[ROWS_A][COLS_A / 2];

    // The row is also the output feature map
    unsigned int row = blockIdx.x * ROWS_A;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Create registers to hold the output values
    __half2 rC[ROWS_A];

    // Initialize the output values
    for (int i = 0; i < ROWS_A; i++) {
        rC[i] = __float2half2_rn(0.0f);
    }

    // Loop over the input matrices
    for (unsigned int tile = 0; tile < (L - 1) / COLS_A + 1; tile++) {

        if (threadIdx.y < COLS_B / 2) {
            // Load the first matrix into shared memory (unrolled)
            unsigned int i = 2 * threadIdx.y / COLS_A;
            unsigned int j = (2 * threadIdx.y) % COLS_A;

            // Load the first float
            float first = 0.0f, second = 0.0f;

            unsigned int unrolled_i = row + i;
            unsigned int unrolled_j = tile * COLS_A + j;
            if (unrolled_i < M && unrolled_j < L) {
                first = mask[unrolled_i * L + unrolled_j];
            }

            // Load the second float
            if (unrolled_i < M && ++unrolled_j < L) {
                second = mask[unrolled_i * L + unrolled_j];
            }

            sA[i][j / 2] = __floats2half2_rn(first, second);
        }

        __syncthreads();
        for (unsigned int idx = 0; idx < COLS_A / 2; idx++) {
            // Load the second matrix into registers
            __half2 rB;
            float first = 0.0f, second = 0.0f;
            if (col < N) {
                unsigned int unrolled_i = tile * COLS_A + 2 * idx;
                unsigned int unrolled_j = col;
                unsigned int h = __float2uint_rd((float) unrolled_j / Width_out);
                unsigned int w = unrolled_j % Width_out;
                if (unrolled_i < L && unrolled_j < N) {
                    // Figure out which feature we are loading
                    unsigned int feature = __float2uint_rd((float) unrolled_i / K_squared);
                    // Figure out which row and column we are loading
                    unsigned int inter = unrolled_i % K_squared;
                    unsigned int p = inter / K;
                    unsigned int q = inter % K;
                    first = in_4d(blockIdx.z, feature, h + p, w + q);
                }
                if (++unrolled_i < L && unrolled_j < N) {
                    // Figure out which feature we are loading
                    unsigned int feature = unrolled_i / K_squared;
                    // Figure out which row and column we are loading
                    unsigned int inter = unrolled_i % K_squared;
                    unsigned int p = inter / K;
                    unsigned int q = inter % K;
                    second = in_4d(blockIdx.z, feature, h + p, w + q);
                }
            }

            rB = __floats2half2_rn(first, second);

            // Multiply and accumulate
            for (unsigned int out_idx = 0; out_idx < ROWS_A; out_idx++) {
                rC[out_idx] = __hfma2(sA[out_idx][idx], rB, rC[out_idx]);
            }
        }
        __syncthreads();
    }

    // Store the results
    for (unsigned int out_idx = 0; out_idx < ROWS_A; out_idx++) {
        float2 result = __half22float2(rC[out_idx]);
        unsigned int unrolled_i = row + out_idx;
        if (unrolled_i < M && col < N) {
            out_4d(blockIdx.z, unrolled_i, col / Width_out, col % Width_out) = result.x + result.y;
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

/*
    FLOAT REGISTER TILED WITH THE MASK MATRIX AS SHARED MEMORY
    This kernel implements:
     - An advanced matrix multiplication algorithm (register-tiled, for example) (5 points)
*/
__global__ void float_conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const unsigned int M = Map_out;
    const unsigned int N = Height_out * Width_out;
    const unsigned int L = K * K * Channel;

    __shared__ float sA[ROWS_A][COLS_A];

    // The row is also the output feature map
    unsigned int row = blockIdx.x * ROWS_A;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Create registers to hold the output values
    float rC[ROWS_A] = {0.0f};

    // Loop over the input matrices
    for (unsigned int tile = 0; tile < (L - 1) / COLS_A + 1; tile++) {
        // Load the first matrix into shared memory (unrolled)
        unsigned int i = threadIdx.y / COLS_A;
        unsigned int j = threadIdx.y % COLS_A;

        unsigned int unrolled_i = row + i;
        unsigned int unrolled_j = tile * COLS_A + j;
        if (unrolled_i < M && unrolled_j < L) {
            sA[i][j] = mask[unrolled_i * L + unrolled_j];
        } else {
            sA[i][j] = 0.0f;
        }

        __syncthreads();
        for (unsigned int idx = 0; idx < COLS_A; idx++) {
            // Load the second matrix into registers
            float rB = 0.0f;
            if (col < N) {
                unsigned int unrolled_i = tile * COLS_A + idx;
                unsigned int unrolled_j = col;
                unsigned int h = unrolled_j / Width_out;
                unsigned int w = unrolled_j % Width_out;
                if (unrolled_i < L && unrolled_j < N) {
                    // Figure out which feature we are loading
                    unsigned int feature = unrolled_i / (K * K);
                    // Figure out which row and column we are loading
                    unsigned int p = (unrolled_i % (K * K)) / K;
                    unsigned int q = (unrolled_i % (K * K)) % K;
                    rB = in_4d(blockIdx.z, feature, h + p, w + q);
                }
            }

            // Multiply and accumulate
            for (unsigned int out_idx = 0; out_idx < ROWS_A; out_idx++) {
                rC[out_idx] += sA[out_idx][idx] * rB;
            }
        }
        __syncthreads();
    }

    // Store the results
    for (unsigned int out_idx = 0; out_idx < ROWS_A; out_idx++) {
        unsigned int unrolled_i = row + out_idx;
        if (unrolled_i < M && col < N) {
            out_4d(blockIdx.z, unrolled_i, col / Width_out, col % Width_out) = rC[out_idx];
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

#define SMALL_COLS_B 128
#define SMALL_ROWS_A 4
#define SMALL_COLS_A (SMALL_COLS_B / SMALL_ROWS_A)

/*
    Uses fewer rows in A
    REGISTER TILED WITH THE MASK MATRIX AS SHARED MEMORY
    This kernel implements:
     - An advanced matrix multiplication algorithm (register-tiled, for example) (5 points)
*/
__global__ void small_conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // The output matrix will be size Map_out x (Height_out * Width_out)
    // - In the first layer, it is 4 x 6400
    // - In the second layer, it is 16 x 1156
    // The first input matrix will be size Map_out x (K * K * Channel) (M x L)
    // - In the first layer, it is 4 x 49
    // - In the second layer, it is 16 x 196
    // The second input matrix will be size (K * K * Channel) x (Height_out * Width_out) (L x N)
    // - In the first layer, it is 49 x 6400
    // - In the second layer, it is 196 x 1156
    // blockIdx.z will be the batch index
    // blockIdx.x and blockIdx.y will be the normal tiling of the output matrix
    // threadIdx.x will be the normal tiling of the output matrix within a tile

    const unsigned int M = Map_out;
    const unsigned int N = Height_out * Width_out;
    const unsigned int K_squared = K * K;
    const unsigned int L = K_squared * Channel;

    __shared__ __half2 sA[SMALL_ROWS_A][SMALL_COLS_A / 2];

    // The row is also the output feature map
    unsigned int row = blockIdx.x * SMALL_ROWS_A;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Create registers to hold the output values
    __half2 rC[SMALL_ROWS_A];

    // Initialize the output values
    for (int i = 0; i < SMALL_ROWS_A; i++) {
        rC[i] = __float2half2_rn(0.0f);
    }

    // Loop over the input matrices
    for (unsigned int tile = 0; tile < (L - 1) / SMALL_COLS_A + 1; tile++) {

        if (threadIdx.y < SMALL_COLS_B / 2) {
            // Load the first matrix into shared memory (unrolled)
            unsigned int i = 2 * threadIdx.y / SMALL_COLS_A;
            unsigned int j = (2 * threadIdx.y) % SMALL_COLS_A;

            // Load the first float
            float first = 0.0f, second = 0.0f;

            unsigned int unrolled_i = row + i;
            unsigned int unrolled_j = tile * SMALL_COLS_A + j;
            if (unrolled_i < M && unrolled_j < L) {
                first = mask[unrolled_i * L + unrolled_j];
            }

            // Load the second float
            if (unrolled_i < M && ++unrolled_j < L) {
                second = mask[unrolled_i * L + unrolled_j];
            }

            sA[i][j / 2] = __floats2half2_rn(first, second);
        }

        __syncthreads();
        for (unsigned int idx = 0; idx < SMALL_COLS_A / 2; idx++) {
            // Load the second matrix into registers
            __half2 rB;
            float first = 0.0f, second = 0.0f;
            if (col < N) {
                unsigned int unrolled_i = tile * SMALL_COLS_A + 2 * idx;
                unsigned int unrolled_j = col;
                unsigned int h = unrolled_j / Width_out;
                unsigned int w = unrolled_j % Width_out;
                if (unrolled_i < L && unrolled_j < N) {
                    // Figure out which feature we are loading
                    unsigned int feature = unrolled_i / K_squared;
                    // Figure out which row and column we are loading
                    unsigned int inter = unrolled_i % K_squared;
                    unsigned int p = inter / K;
                    unsigned int q = inter % K;
                    first = in_4d(blockIdx.z, feature, h + p, w + q);
                }
                if (++unrolled_i < L && unrolled_j < N) {
                    // Figure out which feature we are loading
                    unsigned int feature = unrolled_i / K_squared;
                    // Figure out which row and column we are loading
                    unsigned int inter = unrolled_i % K_squared;
                    unsigned int p = inter / K;
                    unsigned int q = inter % K;
                    second = in_4d(blockIdx.z, feature, h + p, w + q);
                }
            }

            rB = __floats2half2_rn(first, second);

            // Multiply and accumulate
            for (unsigned int out_idx = 0; out_idx < SMALL_ROWS_A; out_idx++) {
                rC[out_idx] = __hfma2(sA[out_idx][idx], rB, rC[out_idx]);
            }
        }
        __syncthreads();
    }

    // Store the results
    for (unsigned int out_idx = 0; out_idx < SMALL_ROWS_A; out_idx++) {
        float2 result = __half22float2(rC[out_idx]);
        unsigned int unrolled_i = row + out_idx;
        if (unrolled_i < M && col < N) {
            out_4d(blockIdx.z, unrolled_i, col / Width_out, col % Width_out) = result.x + result.y;
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

#define INPUT_TILE_SIZE 16
#define KERNEL_SIZE 7
#define OUTPUT_TILE_SIZE (INPUT_TILE_SIZE - KERNEL_SIZE + 1)
__global__ void tiled_conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    /*
        Modify this function to implement the forward pass described in Chapter 16.
        We have added an additional dimension to the tensors to support an entire mini-batch
        The goal here is to be correct AND fast.

        Function paramter definitions:
        output - output
        input - input
        mask - convolution kernel
        Batch - batch_size (number of images in x)
        Map_out - number of output feature maps
        Channel - number of input feature maps
        Height - input height dimension
        Width - input width dimension
        K - kernel height and width (K x K)
    */

    // Insert your GPU convolution kernel code here
    // The output matrix will be size Map_out x (Height_out * Width_out)
    // - In the first layer, it is 4 x 6400
    // - In the second layer, it is 16 x 1156
    // The first input matrix will be size Map_out x (K * K * Channel) (M x L)
    // - In the first layer, it is 4 x 49
    // - In the second layer, it is 16 x 196
    // The second input matrix will be size (K * K * Channel) x (Height_out * Width_out) (L x N)
    // - In the first layer, it is 49 x 6400
    // - In the second layer, it is 196 x 1156
    // blockIdx.z will be the batch index
    // blockIdx.x and blockIdx.y will be the normal tiling of the output matrix
    // threadIdx.x will be the normal tiling of the output matrix within a tile

    // Image dimensions: 86 x 86 and 80 x 80
    // Mask dimensions: 4 1 7 7
    // Image dimensions: 40 x 40 and 34 x 34
    // Mask dimensions: 16 4 7 7

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const unsigned int W_grid = ceil(1.0 * Width_out / OUTPUT_TILE_SIZE);
    const unsigned int m = blockIdx.x;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const int h = (blockIdx.y / W_grid) * OUTPUT_TILE_SIZE + ty;
    const int w = (blockIdx.y % W_grid) * OUTPUT_TILE_SIZE + tx;
    const unsigned int b = blockIdx.z;
    

    __shared__ float sA[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    float acc = 0.0f;
    // for (int c = 0; c < Channel; c++) {
        // Load the tile of the image into shared memory
        if (h < Height && w < Width) {
            sA[ty][tx] = in_4d(b, 0, h, w);
        } else {
            sA[ty][tx] = 0.0;
        }

        __syncthreads();

        if (ty < OUTPUT_TILE_SIZE && tx < OUTPUT_TILE_SIZE) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += sA[ty + p][tx + q] * mask_4d(m, 0, p, q);
                }
            }

            if (h < Height_out && w < Width_out) {
                out_4d(b, m, h, w) = acc;
            }
        }
        // __syncthreads();
    // }

    // if (h < Height_out && w < Width_out && ty < OUTPUT_TILE_SIZE && tx < OUTPUT_TILE_SIZE) {
    //     out_4d(b, m, h, w) = acc;
    // }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

/* 
    ====================================
    The Checkpoint 2 version of conv_forward 
    ====================================
*/
#define TILE_WIDTH 16

__global__ void old_conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;
    float acc = 0.0;
    if (h < Height_out && w < Width_out) {
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }
    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

/*
    ====================================
    The Checkpoint 3 version (with streams) of prolog
    ====================================
*/

#ifdef STREAMS
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    // Copy the whole mask over
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // Copy the input and output in chunks
    // while interweaving the kernel using streams
    const int chunk_size = 100;
    const int image_size = Channel * Height * Width;
    const int output_size = Map_out * Height_out * Width_out;
    const int scount = 3;
    cudaStream_t streams[scount];
    for (int i = 0; i < scount; i++) {
        cudaStreamCreate(&streams[i]);
    }
    // Copy over the first chunk and compute the first chunk
    unsigned int full_input_size = chunk_size * image_size * sizeof(float);
    unsigned int full_output_size = chunk_size * output_size * sizeof(float);
    cudaMemcpyAsync(*device_input_ptr, host_input, full_input_size, cudaMemcpyHostToDevice, streams[0]);
    int m = Map_out;
    int n = Height_out * Width_out;
    if (m < ROWS_A) {
        const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);
        const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
        dim3 dimGrid(Map_out, W_grid * H_grid, chunk_size);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        // Print out the mask dimensions
        // 16 4 7 7 = 3136
        // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
        old_conv_forward_kernel<<<dimGrid, dimBlock>>>(*device_output_ptr, *device_output_ptr, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
        // dim3 dimGrid((m + SMALL_ROWS_A - 1) / SMALL_ROWS_A, (n + SMALL_COLS_B - 1) / SMALL_COLS_B, chunk_size);
        // dim3 dimBlock(1, SMALL_COLS_B, 1);
        // small_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[0]>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync((float *) host_output, *device_output_ptr, full_output_size, cudaMemcpyDeviceToHost, streams[0]);

        for (int i = chunk_size; i < Batch; i += 3 * chunk_size) {
            float* first_dev_input = *device_input_ptr + i * image_size;
            float* first_dev_output = *device_output_ptr + i * output_size;
            float* second_dev_input = first_dev_input + chunk_size * image_size;
            float* second_dev_output = first_dev_output + chunk_size * output_size;
            float* third_dev_input = second_dev_input + chunk_size * image_size;
            float* third_dev_output = second_dev_output + chunk_size * output_size;

            cudaMemcpyAsync(first_dev_input, host_input + i * image_size, full_input_size, cudaMemcpyHostToDevice, streams[0]);
            cudaMemcpyAsync(second_dev_input, host_input + (i + chunk_size) * image_size, full_input_size, cudaMemcpyHostToDevice, streams[1]);
            cudaMemcpyAsync(third_dev_input, host_input + (i + 2 * chunk_size) * image_size, full_input_size, cudaMemcpyHostToDevice, streams[2]);
            
            // small_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[0]>>>(first_dev_output, first_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
            // small_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[1]>>>(second_dev_output, second_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
            // small_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[2]>>>(third_dev_output, third_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);

            old_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[0]>>>(first_dev_output, first_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
            old_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[1]>>>(second_dev_output, second_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
            old_conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[2]>>>(third_dev_output, third_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);

            cudaMemcpyAsync((float *) host_output + i * output_size, first_dev_output, full_output_size, cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync((float *) host_output + (i + chunk_size) * output_size, second_dev_output, full_output_size, cudaMemcpyDeviceToHost, streams[1]);
            cudaMemcpyAsync((float *) host_output + (i + 2 * chunk_size) * output_size, third_dev_output, full_output_size, cudaMemcpyDeviceToHost, streams[2]);
        }
    } else {
        dim3 dimGrid((m + ROWS_A - 1) / ROWS_A, (n + COLS_B - 1) / COLS_B, chunk_size);
        dim3 dimBlock(1, COLS_B, 1);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[0]>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync((float *) host_output, *device_output_ptr, full_output_size, cudaMemcpyDeviceToHost, streams[0]);

        for (int i = chunk_size; i < Batch; i += 3 * chunk_size) {
            float* first_dev_input = *device_input_ptr + i * image_size;
            float* first_dev_output = *device_output_ptr + i * output_size;
            float* second_dev_input = first_dev_input + chunk_size * image_size;
            float* second_dev_output = first_dev_output + chunk_size * output_size;
            float* third_dev_input = second_dev_input + chunk_size * image_size;
            float* third_dev_output = second_dev_output + chunk_size * output_size;
            
            cudaMemcpyAsync(first_dev_input, host_input + i * image_size, full_input_size, cudaMemcpyHostToDevice, streams[0]);
            cudaMemcpyAsync(second_dev_input, host_input + (i + chunk_size) * image_size, full_input_size, cudaMemcpyHostToDevice, streams[1]);
            cudaMemcpyAsync(third_dev_input, host_input + (i + 2 * chunk_size) * image_size, full_input_size, cudaMemcpyHostToDevice, streams[2]);
            
            conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[0]>>>(first_dev_output, first_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[1]>>>(second_dev_output, second_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);
            conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[2]>>>(third_dev_output, third_dev_input, *device_mask_ptr, chunk_size, Map_out, Channel, Height, Width, K);

            cudaMemcpyAsync((float *) host_output + i * output_size, first_dev_output, full_output_size, cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync((float *) host_output + (i + chunk_size) * output_size, second_dev_output, full_output_size, cudaMemcpyDeviceToHost, streams[1]);
            cudaMemcpyAsync((float *) host_output + (i + 2 * chunk_size) * output_size, third_dev_output, full_output_size, cudaMemcpyDeviceToHost, streams[2]);
        }
    }

    for (int i = 0; i < 3; i++) {
        cudaStreamDestroy(streams[i]);
    }
}
#endif

/*
    ====================================
    The Checkpoint 2 and 3 (non stream) version of prolog
    ====================================
*/
#ifndef STREAMS
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    // Copy the mask to constant memory
    // cudaMemcpyToSymbol(cMask, host_mask, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}
#endif

/*
    Competition tiled shared memory
*/
// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);
//     const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
//     dim3 dimGrid(Map_out, W_grid * H_grid, Batch);
//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//     // Print out the mask dimensions
//     // 16 4 7 7 = 3136
//     // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
// }

/*
    ====================================
    The Checkpoint 3 version of conv_forward with mask shared memory and different kernels for layers
    with varying streams
    ====================================
*/
#ifdef STREAMS
__host__ void GPUInterface::conv_forward_gpu(float* __restrict__ device_output, const float* __restrict__ device_input, const float* __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    
}
#endif

/*
    ====================================
    The Checkpoint 3 version of conv_forward with mask shared memory and different kernels for layers
    ====================================
*/
#ifndef STREAMS
__host__ void GPUInterface::conv_forward_gpu(float* __restrict__ device_output, const float* __restrict__ device_input, const float* __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int m = Map_out;
    int n = Height_out * Width_out;
    if (m < ROWS_A) {
        const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);
        const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
        dim3 dimGrid(Map_out, W_grid * H_grid, Batch);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        // Print out the mask dimensions
        // 16 4 7 7 = 3136
        // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
        old_conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
        // const int W_grid = ceil(1.0 * Width_out / OUTPUT_TILE_SIZE);
        // const int H_grid = ceil(1.0 * Height_out / OUTPUT_TILE_SIZE);
        // dim3 dimGrid(Map_out, W_grid * H_grid, Batch);
        // dim3 dimBlock(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1);
        // tiled_conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
        // dim3 dimGrid((m + SMALL_ROWS_A - 1) / SMALL_ROWS_A, (n + SMALL_COLS_B - 1) / SMALL_COLS_B, Batch);
        // dim3 dimBlock(1, SMALL_COLS_B, 1);
        // // printf("Image dimensions: %d x %d and %d x %d\n", Height, Width, Height_out, Width_out);
        // // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
        // small_conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    } else {
        dim3 dimGrid((m + ROWS_A - 1) / ROWS_A, (n + COLS_B - 1) / COLS_B, Batch);
        dim3 dimBlock(1, COLS_B, 1);
        // printf("Image dimensions: %d x %d and %d x %d\n", Height, Width, Height_out, Width_out);
        // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
        conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    }
}
#endif

/*
    ====================================
    The Checkpoint 3 version of conv_forward with mask shared memory
    ====================================
*/
// __host__ void GPUInterface::conv_forward_gpu(float* __restrict__ device_output, const float* __restrict__ device_input, const float* __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     int m = Map_out;
//     int n = Height_out * Width_out;
//     dim3 dimGrid((m + ROWS_A - 1) / ROWS_A, (n + COLS_B - 1) / COLS_B, Batch);
//     dim3 dimBlock(1, COLS_B, 1);
//     // printf("Image dimensions: %d x %d and %d x %d\n", Height, Width, Height_out, Width_out);
//     // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
// }

/*
    ====================================
    The Checkpoint 3 version of conv_forward with mask register
    ====================================
*/

// __host__ void GPUInterface::conv_forward_gpu(float* __restrict__ device_output, const float* __restrict__ device_input, const float* __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     int m = Map_out;
//     int n = Height_out * Width_out;
//     dim3 dimGrid((m + ROWS_A - 1) / ROWS_A, (n + COLS_B - 1) / COLS_B, Batch);
//     dim3 dimBlock(ROWS_A, 1, 1);
//     // printf("Image dimensions: %d x %d and %d x %d\n", Height, Width, Height_out, Width_out);
//     // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
// }

/*
    ====================================
    The Checkpoint 2 version of conv_forward 
    ====================================
*/
// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);
//     const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
//     dim3 dimGrid(Map_out, W_grid * H_grid, Batch);
//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//     // Print out the mask dimensions
//     // 16 4 7 7 = 3136
//     // printf("Mask dimensions: %d %d %d %d\n", Map_out, Channel, K, K);
//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
// }

/*
    ====================================
    The Checkpoint 3 (with streams) version of epilog 
    ====================================
*/
#ifdef STREAMS
__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_output, float* __restrict__ device_output, float* __restrict__ device_input, float* __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}
#endif

/*
    ====================================
    The Checkpoint 2 and 3 (non stream) version of epilog
    ====================================
*/
#ifndef STREAMS
__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_output, float* __restrict__ device_output, float* __restrict__ device_input, float* __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}
#endif


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
