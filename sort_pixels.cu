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

int StringToInt(string str){
    int i;
    istringstream iss(str);
    iss >> i;
    return i;
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

void SaveSortedPixelsToImage(BmpImage *image, int *pixels, int imageHeight, int imageWidth)
{
    for (int row = 0; row < imageHeight; row++)
    { 
        for (int column = 0; column < imageWidth; column++)
        {
            image->SetPixel24AsInt(column, row, pixels[row*imageWidth + column]);
        }
    }
}

void sortPixelsGpu(BmpImage *image, int colorMode, int blockSize, int chunkSize, bool optimized){
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

    int rowsRemaining = imageHeight;
    int iteration = 1;

    cout << "Chunk size: " << chunkSize << endl;
    while(rowsRemaining > 0)
    {
        int currentRows = (rowsRemaining >  chunkSize) ? chunkSize : rowsRemaining;
        int currentRowsSize = currentRows * imageWidth * sizeof(int);
        int offset = imageHeight - rowsRemaining;

        cout << "Iteration " << iteration << ":" << endl;
        cout << "\tRows remaining: " << rowsRemaining << endl;

        cudaMalloc((void **) &pixels_d, currentRowsSize);

        cudaMemcpy(pixels_d, pixels_h + offset * imageWidth, currentRowsSize, cudaMemcpyHostToDevice);

        int n_blocks = currentRows / blockSize + (imageHeight % blockSize == 0 ? 0:1);
        cout << "\tBlocks: " << n_blocks << endl;
        cout << "\tBlock size: " << blockSize << endl;

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

        cudaMemcpy(pixels_h + offset * imageWidth, pixels_d, currentRowsSize, cudaMemcpyDeviceToHost);
        
        cudaEventSynchronize(stop);
        float tempTime;
        cudaEventElapsedTime(&tempTime, start, stop);
        timeElapsed += tempTime;

        cudaFree(pixels_d);

        rowsRemaining -= currentRows;
    }

    SaveSortedPixelsToImage(image, pixels_h, imageHeight, imageWidth);

    cout << timeInfo << timeElapsed << "ms" << endl << endl;

    free(pixels_h);
}

int main(int argc, char *argv[]){
    if(argc < 5){
        cout << "Not enough arguments - example call: apl_project <filename> <onlyOptimized> <blockSize> <chunkSize>" << endl;
        return 1;
    }

    BmpImage image;
    string filename = argv[1];
    int onlyOptimized = StringToInt(argv[2]);
    int blockSize = StringToInt(argv[3]);
    int chunkSize = StringToInt(argv[4]);

    cudaSetDevice(0);

    cout << "----------------TEST RUN FOR BLOCKSIZE: " << blockSize << "----------------" << endl;
    if(!onlyOptimized)
    {
        if(!image.Load(filename))
        {
            cout << "File didn't load correctly!" << endl;
            return 1;
        }

        sortPixelsGpu(&image, image.ParseRgbTripleToInt(whiteRgb), blockSize, chunkSize, false);
        string sortedfile = IntToString(blockSize) + "_" + filename;
        image.Save(sortedfile);
    }

    if(!image.Load(filename))
    {
        cout << "File didn't load correctly!" << endl;
        return 1;
    }

    sortPixelsGpu(&image, image.ParseRgbTripleToInt(whiteRgb), blockSize, chunkSize, true);
    string optimizedSortedfile = IntToString(blockSize) + "_optimized_" + filename;
    image.Save(optimizedSortedfile);

    return 0;
}

 