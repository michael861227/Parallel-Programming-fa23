#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
extern "C"{
#include "hostFE.h"
}

__global__ void convKernel(float *inputImage, float *outputImage, float *filter,
                     const int imageHeight, const int imageWidth, const int filterWidth)
{
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    int halffilterWidth = filterWidth / 2;
    int sum = 0;
    int k, l;

    sum = 0;
    for (k = -halffilterWidth; k <= halffilterWidth; k++)
    {
        for (l = -halffilterWidth; l <= halffilterWidth; l++)
        {
            if (thisY + k >= 0 && thisY + k < imageHeight &&
                thisX + l >= 0 && thisX + l < imageWidth)
            {
                sum += inputImage[(thisY + k) * imageWidth + thisX + l] *
                        filter[(k + halffilterWidth) * filterWidth +
                                l + halffilterWidth];
            }
        }
    }
    outputImage[thisY * imageWidth + thisX] = sum;
}

// Host front-end function that allocates the memory and launches the GPU kernel
extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    float *device_inputImage;
    float *device_outputImage;
    float *device_filter;
    
    cudaMalloc(&device_inputImage, imageSize);
    cudaMalloc(&device_outputImage, imageSize);
    cudaMalloc(&device_filter, filterSize);

    cudaMemcpy(device_filter, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
    convKernel<<<numBlocks, threadsPerBlock>>>(device_inputImage, device_outputImage, device_filter, imageHeight, imageWidth, filterWidth);

    cudaMemcpy(outputImage, device_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(device_filter);
    cudaFree(device_inputImage);
    cudaFree(device_outputImage);
}