#include <cuda.h>

#define THRESHOLD 10010000

__device__ void bubbleSort(int *pixelsToSort, int length){
    for(int i = 0; i < length; i++ )
    {
        for(int j = 0; j < length-1; j++)
        {
            if( pixelsToSort[j] > pixelsToSort[j+1]){
                int tmp = pixelsToSort[j];
                pixelsToSort[j] = pixelsToSort[j+1];
                pixelsToSort[j+1] = tmp;
            }
        }
    }
}

__device__ int cudaGetFirstNotInColor(int *image, int x, int row, int imageWidth, int color){
    for (int i = x; i < image->GetWidth(); ++i)
    {
        int pixelValue = image[row*imageWidth + i];
        if(THRESHOLD < (color - pixelValue)){
            return i;
        }
    }
    return -1;
}

__device__ int cudaGetNextInColor(int *image, int x, int row, int imageWidth, int color){
    for (int i = x + 1; i < image->GetWidth(); ++i)
    {
        int pixelValue = image[row*imageWidth + i];
        if(THRESHOLD >= (color - pixelValue)){
            return i-1;
        }
    }
    return image->GetWidth() - 1;
}

__global__ void sortRows(int *image, int imageHeight, int imageWidth, int colorMode){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < imageHeight)
    {
        int startingX = 0;
        int finishX = 0;

        while(finishX < imageWidth)
        {
            startingX = getFirstNotInColor(image, startingX, row, imageWidth, colorMode);
            finishX = getNextInColor(image, startingX, row, imageWidth, colorMode);

            if(startingX < 0)
                break;

            int pixelsToSortLength = finishX - startingX;
            int *pixelsToSort = new int[pixelsToSortLength];
            for (int i = 0; i < pixelsToSortLength; ++i)
            {
                pixelsToSort[i] = image[row*imageWidth + startingX + i];
            }

            bubbleSort(pixelsToSort, pixelsToSortLength);

            for (int i = 0; i < pixelsToSortLength; ++i)
            {
                image[row*imageWidth + startingX + i] = pixelsToSort[i];
            }

            startingX = finishX + 1;
        }
    }
}