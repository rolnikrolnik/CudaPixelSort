#include <fstream>
#include <string>
#include <sstream>
#include <cuda.h>
#include "BmpImage.h"
#include "kernels.cu"

#define THRESHOLD 10010000

using namespace std;

RGBTRIPLE blackRgb = { 0, 0, 0};
RGBTRIPLE whiteRgb = { 255, 255, 255};

string IntToString (int a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}

void SavePixelsToIntArray(BmpImage *image, int *pixels, int imageHeight, int imageWidth)
{
    for (int row = 0; row < imageHeight; row++)
    { 
        for (int column = 0; column < imageWidth; column++)
        {
            pixels[row*imageWidth + column] = image->GetPixel24AsInt(column, row);
        }
    }
}

int SaveSortedPixelsToImage(BmpImage *image, int *pixels, int lastRow, int imageHeight, int imageWidth)
{
    for (int row = lastRow; row < imageHeight; row++)
    { 
        for (int column = 0; column < imageWidth; column++)
        {
            image->SetPixel24AsInt(column, row, pixels[row*imageWidth + column]);
        }
        lastRow = row;
    }
}

void sortPixelsGpu(BmpImage *image, int colorMode, int blockSize, bool optimized){
    int *pixels_h, *pixels_d;
    int imageWidth = image->GetWidth();
    int imageHeight = image->GetHeight();
    int imageSize = image->GetSize();
    string timeInfo;
    float timeElapsed = 0;

    // eventes to get time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    int totalSize = imageSize * sizeof(int);
    pixels_h = (int *)malloc(totalSize);
    SavePixelsToIntArray(image, pixels_h, imageHeight, imageWidth);


    cudaMalloc((void **) &pixels_d, totalSize);
    cudaMemcpy(pixels_d, pixels_h, totalSize, cudaMemcpyHostToDevice);
    int n_blocks = imageHeight/blockSize + (imageHeight%blockSize == 0 ? 0:1);

    cout << "Blocks: " << n_blocks << endl;
    cout << "Block size: " << blockSize << endl;

    // invoking kernel and timing 
    cudaEventRecord(start);

    if(optimized)
    {
        optimizedSortRows<<< n_blocks, blockSize >>> (pixels_d, imageHeight, imageWidth, colorMode);
        timeInfo = "Optimized kernel execution time: ";
    }
    else
    {
        sortRows<<< n_blocks, blockSize >>> (pixels_d, imageHeight, imageWidth, colorMode);
        timeInfo = "Kernel execution time: ";
    }

    cudaEventRecord(stop);

    cudaMemcpy(pixels_h, pixels_d, totalSize, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);
    cout << timeInfo << timeElapsed << "ms" << endl;

    SaveSortedPixelsToImage(image, pixels_h, imageHeight, imageWidth);

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
    int blockSizes[5] = {1, 16, 64, 128, 256};

    cudaSetDevice(0);

    for (int i = 0; i < 5; ++i)
    {
        cout << "----------------TEST RUN FOR BLOCKSIZE: " << blockSizes[i] << "----------------" << endl;
        if(!image.Load(filename))
        {
            cout << "File didn't load correctly!" << endl;
            return 1;
        }

        sortPixelsGpu(&image, image.ParseRgbTripleToInt(whiteRgb), blockSizes[i], false);
        string sortedfile = IntToString(blockSizes[i]) + "_" + filename;
        image.Save(sortedfile);

        if(!image.Load(filename))
        {
            cout << "File didn't load correctly!" << endl;
            return 1;
        }

        sortPixelsGpu(&image, image.ParseRgbTripleToInt(whiteRgb), blockSizes[i], true);
        string optimizedSortedfile = IntToString(blockSizes[i]) + "_optimized_" + filename;
        image.Save(optimizedSortedfile);
    }

    return 0;
}

 