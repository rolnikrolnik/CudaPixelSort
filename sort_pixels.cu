#include <fstream>
#include <algorithm>
#include <string>
#include <cuda.h>
#include "BmpImage.h"
#include "kernels.cu"

#define THRESHOLD 10010000

using namespace std;

RGBTRIPLE blackRgb = { 0, 0, 0};
RGBTRIPLE whiteRgb = { 255, 255, 255};

void sortPixelsGpu(BmpImage *image){
 
    int *pixels_h, *pixels_d;
    int imageWidth = image->GetWidth();
    int imageHeight = image->GetHeight();
    int imageSize = image->GetSize();

    int allocationSize = imageSize * sizeof(int);

    // Allocate host and device pixels
    pixels_h = (int *)malloc(allocationSize);
    cudaMalloc((void **) &pixels_d, allocationSize);

    for (int row = 0; row < imageHeight; row++)
    { 
        for (int column = 0; column < imageWidth; column++)
        {
            pixels_h[row*imageWidth + ] = image->GetPixel24AsInt(column, row);
        }
    }

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    int block_size = 4;
    int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
    square_array <<< n_blocks, block_size >>> (a_d, N);


    // Retrieve result from device and store it in host array
    cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    // Print results
    for (int i=0; i<N; i++) printf("%d %f\n", i, a_h[i]);
    // Cleanup
    free(a_h);
    cudaFree(a_d);
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

    sortPixelsGpu(&image);

    image.Save("example_sorted.bmp");

    return 0;
}

 