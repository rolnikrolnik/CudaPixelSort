#include <fstream>
#include <algorithm>
#include <string>
#include "BmpImage.h"

#define THRESHOLD 10010000

using namespace std;

int getNextInColor(BmpImage*, int, int, int);
int getFirstNotInColor(BmpImage*, int, int, int);
void sortRow(BmpImage*, int);

RGBTRIPLE blackRgb = { 0, 0, 0};
RGBTRIPLE whiteRgb = { 255, 255, 255};

void bubbleSort(int *pixelsToSort, int length){
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

void sortPixelsCpu(BmpImage *image){
    for (int i = 0; i < image->GetHeight(); ++i)
    {
        sortRow(image, i);
    }
}

void sortRow(BmpImage *image, int row){
    int startingX = 0;
    int finishX = 0;

    while(finishX < image->GetWidth())
    {
        startingX = getFirstNotInColor(image, startingX, row, image->ParseRgbTripleToInt(whiteRgb));
        finishX = getNextInColor(image, startingX, row, image->ParseRgbTripleToInt(whiteRgb));

        if(startingX < 0)
            break;

        int pixelsToSortLength = finishX - startingX;
        int *pixelsToSort = new int[pixelsToSortLength];
        for (int i = 0; i < pixelsToSortLength; ++i)
        {
            pixelsToSort[i] = image->GetPixel24AsInt(startingX + i, row);
        }

        bubbleSort(pixelsToSort, pixelsToSortLength);

        for (int i = 0; i < pixelsToSortLength; ++i)
        {
            image->SetPixel24AsInt(startingX + i, row, pixelsToSort[i]);
        }

        startingX = finishX + 1;
    }
}

int getFirstNotInColor(BmpImage *image, int x, int row, int color){
    for (int i = x; i < image->GetWidth(); ++i)
    {
        int pixelValue = image->GetPixel24AsInt(i, row);
        if(THRESHOLD < (color - pixelValue)){
            return i;
        }
    }
    return -1;
}

int getNextInColor(BmpImage *image, int x, int row, int color){
    for (int i = x + 1; i < image->GetWidth(); ++i)
    {
        int pixelValue = image->GetPixel24AsInt(i, row);
        if(THRESHOLD >= (color - pixelValue)){
            return i-1;
        }
    }
    return image->GetWidth() - 1;
}

int main(){

    BmpImage image;

    string filename = "example.bmp";
    if(!image.Load(filename)){
        cout << "File didn't load correctly!" << endl;
    }

    sortPixelsCpu(&image);

    image.Save("example_sorted.bmp");

    return 0;
}

 