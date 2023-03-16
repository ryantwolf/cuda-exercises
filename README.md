# CUDA Programming Exercises
This repository contains the assignments I did as a part of ECE 408/CS 483 at UIUC.
Each assignment was implementing some CUDA kernel that we discussed in lectures.
I recieved at least 100% each coding assignment (extra credit on the final project) and got an A overall in the class.

## Assignments
All assigments were done individually with the lecture slides and CUDA documentation as reference. Check the folder with the same label as the assignment for code and more details in the READMEs.

1. **MP1: Vector Addition.** It was intended to teach us the basics of kernel programming including concepts like threads, thread blocks and data transfer. We also discussed how the slow memory transfer speed would make this kernel impractical.
1. **MP2: Simple Matrix Multiply.** This was our baseline implementation that we would improve on an compare against in further assignments. We discussed how this implementation suffers from repeated calls to global memory leading to a poor memory access to FLOP ratio.
1. **MP3: Tiled Matrix Multiply.** This assignment had us load tiles of the input matrices into shared memory to improve efficiency. We were also introduced to `__syncthreads()` for ensuring that tiles were loaded properly.
1. **MP4: Tiled 3D convolution.** We applied the same concept of tiling from MP3 to 3D convolutions. This also served as a warmup for the eventual final project.
1. **MP5.1: List Reduction.** This assignment was the introduction to reduction operations. We implemented a list reduction kernel for computing the sum of a list that is too big to fit into a single block of threads.
1. **MP5.2: Scan.** This assignment had us implement a scan (prefix sum) kernel using the Brent-Kung algorithm.
1. **MP6: Histogram Equalization.** For this assignment, we wrote a series of kernels to compute the histogram of intensity for an image then [equalize](https://en.wikipedia.org/wiki/Histogram_equalization) it. We applied atomic operations for the parallel calculation of the histogram and used the scan implementation from the previous MP for computing the CDF of the histogram.
1. **MP7: Sparse Matrix Multiply.** We implmented a Sparse Matrix-Vector Multiplication (SpMV) kernel based on the Jagged Diagonal Sparse (JDS) transposed format. JDS is used when there is a large (but regular) disparity in the number of non-zero elements in a matrix. By using this format, warp divergence is minimized as adjacent rows (after sorting) have similar number of non-zero elements, so adjacent threads will have a similar amount of work to do. The transposed variant utilizes the DRAM bursting capabilities of the GPU by transforming the memory layout for the data array from row-major to column-major so that the threads are accessing adjacent memory locations at the same time.
1. **Final Project: Convolutional Layer Forward Pass.** The final project combined a lot of ideas from previous assignments and it allowed us to explore concepts not taught in class as well. `Project/README.md` goes in depth on all the details, but the overall idea is that we implement a convolutional layer in a neural network with the ability to handle batched inputs and an arbitrary number of input and output channels. We also were tasked with profiling our implementation using the CPU profiling tool `gprof` and the NVIDIA's GPU profiling tools Nsight-Systems (`nsys`) and Nsight-Compute (`nv-nsight-cu-cli`).

    There were 3 checkpoints for the project.
    1. The first checkpoint had us implement and profile a basic CPU implementation.
    1. The second checkpoint had us implement and profile a basic GPU implementation.
    1. The third checkpoint was the main focus of the project. There were a list of optimizations we could explore and implement. Here are some of the major optimizations I made:
        - Shared memory tiling of convolution. See MP4.
        - Register tiled matrix multiplication. I (implicitly) unrolled the input images into a matrix and performed joint register and shared memory tiling to multiply it with the filter matrix.
        - Filters in constant memory. Self-explanatory, putting the filters into constant memory speeds things up.
        - Half precision (FP16 arithmetic). During the unrolling process, I converted the `float`s into `__half2` to speed up computation.
        - Streams. I used streams to overlap the data transfer to and from the GPU with the computation. This was definitely the largest improvement in performance I observed.
    
        There was also a competition that was run to see who could write the fastest kernel. I got extra credit for being in the top 20 fastest implementations. 