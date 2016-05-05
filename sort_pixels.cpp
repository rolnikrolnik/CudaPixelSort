#include <fstream>
#include <algorithm>
#include <string>
#include "BmpImage.h"

using namespace std;

int getNextInColor(BmpImage*, int, int, int);
int getFirstNotInColor(BmpImage*, int, int, int);
void sortRow(BmpImage*, int);

RGBTRIPLE blackRgb = { 0, 0, 0};
RGBTRIPLE whiteRgb = { 255, 255, 255};

// TODO: think about this threshold - wtf is going on
int threshold = 10010000;

void sortPixels(BmpImage *image){
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

        sort(pixelsToSort, pixelsToSort + pixelsToSortLength);

        for (int i = 0; i < pixelsToSortLength; ++i)
        {
            // TODO : SetPixel24AsInt
            image->SetPixel24(startingX + i, row, image->ParseIntToRgbTriple(pixelsToSort[i]));
        }

        startingX = finishX + 1;
    }
}

// TODO: refactor those 2 methods to understand them better
int getFirstNotInColor(BmpImage *image, int x, int row, int color){
    for (int i = x; i < image->GetWidth(); ++i)
    {
        int pixelValue = image->GetPixel24AsInt(i, row);
        if(threshold < (color - pixelValue)){
            return i;
        }
    }
    return -1;
}

int getNextInColor(BmpImage *image, int x, int row, int color){
    for (int i = x + 1; i < image->GetWidth(); ++i)
    {
        int pixelValue = image->GetPixel24AsInt(i, row);
        if(threshold >= (color - pixelValue)){
            return i-1;
        }
    }
    return image->GetWidth() - 1;
}

int main(){

    // TODO: handle command line args    
    BmpImage image;

    string filename = "example.bmp";
    if(!image.Load(filename)){
        cout << "File didn't load correctly!" << endl;
    }

    sortPixels(&image);

    image.Save("example_sorted.bmp");

    return 0;
}

