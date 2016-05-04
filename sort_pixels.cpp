#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "BmpImage.h"

using namespace std;

RGBTRIPLE black = { 0, 0, 0};
RGBTRIPLE white = { 255, 255, 255};

int getNextWhite(BmpImage*, int, int);
int getFirstNotWhiteInRow(BmpImage*, int, int);

int parseRgbTriple(RGBTRIPLE pixel){
    return pixel.rgbtRed << 16 + pixel.rgbtGreen << 8 + pixel.rgbtBlue;
}


void sortPixels(BmpImage *image){
    for (int i = 0; i < image->GetHeight(); ++i)
    {
        for (int j = 0; j < image->GetWidth(); ++j)
        {

        }
    }
}

void sortRow(BmpImage *image, int row){
    int startingX = 0;
    int finishX = 0;

    while(finishX < image->GetWidth())
    {
        startingX = getFirstNotWhiteInRow(image, x, row);
        finishX = getNextWhite(image, x, row);

        if(startingX < 0)
            break;

        int pixelsToSortLength = finishX - startingX;
        int *pixelsToSort = new int[pixelsToSortLength];
        for (int i = 0; i < pixelsToSortLength; ++i)
        {
            pixelsToSort[i] = parseRgbTriple(image->GetPixel24(startingX + i, row));
        }

        sort(pixelsToSort, pixelsToSort + pixelsToSortLength);

        for (int i = 0; i < pixelsToSortLength; ++i)
        {
            image->SetPixel24(startingX + i, row, pixelsToSort[i])
        }
    }
}

int getFirstNotWhiteInRow(BmpImage *image, int x, int row){
    for (int i = x; i < image->GetWidth(); ++i)
    {
        RGBTRIPLE pixel = image->GetPixel24(i, row);
        int pixelValue = parseRgbTriple(pixel);
        if(pixel < white){
            return i;
        }
    }
    return -1;
}

int getNextWhite(BmpImage *image, int x, int row){
    for (int i = x + 1; i < image->GetWidth(); ++i)
    {
        RGBTRIPLE pixel = image->GetPixel24(i, row);
        int pixelValue = parseRgbTriple(pixel);
        if(pixel > white){
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

    return 0;
}

