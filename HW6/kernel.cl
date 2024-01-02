__kernel void convolution(
    __global const float *inputImage,
    __global const float *filter,
    __global float *outputImage,
    const int imageHeight,
    const int imageWidth,
    const int filterWidth) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halffilterWidth = filterWidth / 2;
    float sum;
    int k, l;

    sum = 0;
    for (k = -halffilterWidth; k <= halffilterWidth; k++)
    {
        for (l = -halffilterWidth; l <= halffilterWidth; l++)
        {
            if (y + k >= 0 && y + k < imageHeight &&
                x + l >= 0 && x + l < imageWidth)
            {
                sum += inputImage[(y + k) * imageWidth + x + l] *
                        filter[(k + halffilterWidth) * filterWidth +
                                l + halffilterWidth];
            }
        }
    }
    outputImage[y * imageWidth + x] = sum;
}
