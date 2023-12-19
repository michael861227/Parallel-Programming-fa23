#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;

    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(int *device_out, size_t pitch,float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // Calculate the index of the current thread
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = thisY * resX + thisX;    

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    // device_out[index] = mandel(x, y, maxIterations);
    *((int *)((char *)device_out + thisY * pitch) + thisX) = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *host_out, *device_out;
    int size = resX * resY * sizeof(int);

    // Allocate memory on the host and device
    // 1600 * 1200
    size_t pitch;
    cudaHostAlloc((void**)&host_out, size, cudaHostAllocDefault);
    cudaMallocPitch((void**)&device_out, &pitch, resX * sizeof(int), resY);

    // Launch the kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(device_out, pitch, lowerX, lowerY, stepX, stepY, resX, maxIterations);

    // Copy the result back to the host
    cudaMemcpy2D(host_out, resX * sizeof(int), device_out, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, host_out, size);

    // Free the memory
    cudaFree(device_out);
    cudaFreeHost(host_out);
}