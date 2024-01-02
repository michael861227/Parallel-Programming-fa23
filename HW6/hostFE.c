#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;
    
    // Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");
    
    // Create memory buffers on the device
    cl_mem inputImg = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), inputImage, &status);
    CHECK(status, "clCreateBuffer");

    cl_mem filterMemObj = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize * sizeof(float), filter, &status);
    CHECK(status, "clCreateBuffer");

    cl_mem outputImg = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), outputImage, &status);
    CHECK(status, "clCreateBuffer");

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputImg);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filterMemObj);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputImg);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

    // Execute the OpenCL kernel on the list
    size_t globalWorkSize[2] = {imageWidth, imageHeight};
    size_t localWorkSize[2] = {8, 8}; // Note : global_work_size must be divisible by local_work_size

    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");
    
    status = clEnqueueReadBuffer(commandQueue, outputImg, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");
}