#include <fstream>
#include <string>
#include <cuda.h>
#include "BmpImage.h"
#include "kernels.cu"

#define THRESHOLD 10010000

using namespace std;

RGBTRIPLE blackRgb = { 0, 0, 0};
RGBTRIPLE whiteRgb = { 255, 255, 255};

void sortPixelsGpu(BmpImage *image, int colorMode){
    int *pixels_h, *pixels_d;
    int imageWidth = image->GetWidth();
    int imageHeight = image->GetHeight();
    int imageSize = image->GetSize();
    
    // eventes to get time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 


    int allocationSize = imageSize * sizeof(int);

    pixels_h = (int *)malloc(allocationSize);
    cudaMalloc((void **) &pixels_d, allocationSize);

    for (int row = 0; row < imageHeight; row++)
    { 
        for (int column = 0; column < imageWidth; column++)
        {
            pixels_h[row*imageWidth + column] = image->GetPixel24AsInt(column, row);
        }
    }

    cudaMemcpy(pixels_d, pixels_h, allocationSize, cudaMemcpyHostToDevice);

    int block_size = 128;
    int n_blocks = imageHeight/block_size + (imageSize%block_size == 0 ? 0:1);

    cudaEventRecord(start);
    optimizedSortRows<<< n_blocks, block_size >>> (pixels_d, imageHeight, imageWidth, colorMode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    { 
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);



    cudaMemcpy(pixels_h, pixels_d, allocationSize, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Kernel execution time: " << milliseconds << "ms" << endl;

    for (int row = 0; row < imageHeight; row++)
    { 
        for (int column = 0; column < imageWidth; column++)
        {
            image->SetPixel24AsInt(column, row, pixels_h[row*imageWidth + column]);
        }
    }

    free(pixels_h);
    cudaFree(pixels_d);
}

int main(int argc, char *argv[]){
    if(argc < 2){
        cout << "Not enough arguments - please pass filename!" << endl;
        return 1;
    }

    BmpImage image;
    string filename = argv[1];
    if(!image.Load(filename)){
        cout << "File didn't load correctly!" << endl;
        return 1;
    }

    sortPixelsGpu(&image, image.ParseRgbTripleToInt(whiteRgb));

    string sortedfile = "sorted_" + filename;
    image.Save(sortedfile);

    return 0;
}

 