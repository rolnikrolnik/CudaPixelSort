/* This program is free software. It comes without any warranty, to
     * the extent permitted by applicable law. You can redistribute it
     * and/or modify it under the terms of the Do What The Fuck You Want
     * To Public License, Version 2, as published by Sam Hocevar. See
     * http://www.wtfpl.net/ for more details. */

#include "BmpImage.h"

using namespace std;

BmpImage::BmpImage()
{
	mBitmapInfo = NULL;
	mBitmap = NULL;
}

BmpImage::BmpImage(const BmpImage& other)
{
	mHeader = other.mHeader;

	//int sizeToCopy = mHeader.bfSize - (sizeof(mHeader));
	mData = other.mData;
	mBitmapInfo = reinterpret_cast<BITMAPINFO*>(mData.data());
	mBitmap = mData.data() + mHeader.bfOffBits - sizeof(mHeader);
}

BmpImage& BmpImage::operator=(const BmpImage& other)
{
	// check for self-assignment
	if(&other == this)
		return *this;

	mHeader = other.mHeader;

	//int sizeToCopy = mHeader.bfSize - (sizeof(mHeader));
	mData = other.mData;
	mBitmapInfo = reinterpret_cast<BITMAPINFO*>(mData.data());
	mBitmap = mData.data() + mHeader.bfOffBits - sizeof(mHeader);

	return *this;
}

bool BmpImage::Load(string fileName)
{
	std::fstream file;
	file.open(fileName.c_str(), fstream::in | fstream::binary);

	if(!file)
		return false;

	file.read(reinterpret_cast<char*>(&mHeader), sizeof(mHeader));

	if(mHeader.bfType != 0x4D42)
		return false;

	int sizeToRead = mHeader.bfSize - (sizeof(mHeader));
	mData = vector<BYTE>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	mBitmapInfo = reinterpret_cast<BITMAPINFO*>(mData.data());
	mBitmap = mData.data() + mHeader.bfOffBits - sizeof(mHeader);

	file.close();
	return true;
}


bool BmpImage::Save(string fileName)
{
	if(mData.size() == 0)
		return false;

	fstream f;
	f.open(fileName.c_str(), fstream::out | fstream::binary);

	if(!f)
		return false;

	f.write(reinterpret_cast<char*>(&mHeader), sizeof(mHeader));
	f.write(reinterpret_cast<char*>(mData.data()), mHeader.bfSize - sizeof(mHeader));
	f.close();

	return true;
}

bool BmpImage::GetPixel1(int x, int y)
{
	if(mBitmap == NULL)
		return false;

	// ile bajtów zajmuje linia (8 px per bajt)
	int lineSize = mBitmapInfo->bmiHeader.biWidth / 8;

	if(mBitmapInfo->bmiHeader.biWidth % 8 != 0)
		lineSize++;

	if(lineSize % 4)
		lineSize = lineSize + 4 - lineSize % 4;

	// numer bajtu gdzie le¿y pixel
	int byteNo = lineSize * y + x / 8;

	// px ustawione od najstarszych bitów
	int bitMask = 1 << (7 - (x % 8));

	// nasz pixel
	int pixel = mBitmap[byteNo] & bitMask;

	return (pixel != 0);
}

BYTE BmpImage::GetPixel8(int x, int y)
{
	if(mBitmap == NULL)
		return 0;

	int lineSize = mBitmapInfo->bmiHeader.biWidth;

	if(lineSize % 4 != 0)
		lineSize = lineSize + 4 - lineSize % 4;

	return mBitmap[y * lineSize + x];
}

RGBTRIPLE BmpImage::GetPixel24(int x, int y)
{
	RGBTRIPLE ret;

	if(mBitmap == NULL)
		return ret;

	int lineSize = mBitmapInfo->bmiHeader.biWidth * 3;
	if(lineSize % 4)
		lineSize = lineSize + 4 - lineSize % 4;

	int pozInArray = (y * lineSize + x * 3);
	memcpy(&ret, mBitmap + pozInArray, sizeof(ret));

	return ret;
}

int BmpImage::GetPixel24AsInt(int x, int y)
{
	return ParseRgbTripleToInt(GetPixel24(x, y));
}

void BmpImage::SetPixel8(int x, int y, BYTE val)
{
	if(mBitmap == NULL)
		return;

	int lineSize = mBitmapInfo->bmiHeader.biWidth;

	if(lineSize % 4)
		lineSize = lineSize + 4 - lineSize % 4;

	mBitmap[y * lineSize + x] = val;
}

void BmpImage::SetPixel24(int x, int y, RGBTRIPLE val)
{
	if(mBitmap == NULL)
		return;

	int lineSize = mBitmapInfo->bmiHeader.biWidth * 3;
	if(lineSize % 4)
		lineSize = lineSize + 4 - lineSize % 4;

	int pozInArray = (y * lineSize + x * 3);

	memcpy(mBitmap + pozInArray, &val, sizeof(val));
}

int BmpImage::GetWidth()
{
	if(mData.size() == 0)
		return -1;

	return mBitmapInfo->bmiHeader.biWidth;
}

int BmpImage::GetHeight()
{
	if(mData.size() == 0)
		return -1;

	return mBitmapInfo->bmiHeader.biHeight;
}

int BmpImage::GetSize()
{
	//return static_cast<int>(mData.size());
	return GetWidth() * GetHeight();
}

int BmpImage::GetBitsPerPixel()
{
	if(mData.size() == 0)
		return -1;

	return mBitmapInfo->bmiHeader.biBitCount;
}

bool BmpImage::CreateGreyscaleDIB(int width, int height)
{
	mHeader.bfType = 0x4D42;
	mHeader.bfReserved1 = mHeader.bfReserved2 = 0;
	mHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * 256;
	mHeader.bfOffBits = mHeader.bfSize;

	int lineSize = width;

	if(lineSize % 4)
		lineSize = lineSize + 4 - lineSize % 4;

	mHeader.bfSize += lineSize * height;

	mData = vector<BYTE>(mHeader.bfSize - sizeof(mHeader));
	mBitmapInfo = (BITMAPINFO*)mData.data();
	mBitmap = (BYTE*)(mData.data() + mHeader.bfOffBits - sizeof(mHeader));

	mBitmapInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	mBitmapInfo->bmiHeader.biWidth = width;
	mBitmapInfo->bmiHeader.biHeight = height;
	mBitmapInfo->bmiHeader.biPlanes = 1;
	mBitmapInfo->bmiHeader.biBitCount = 8;
	mBitmapInfo->bmiHeader.biCompression = BI_RGB;
	mBitmapInfo->bmiHeader.biSizeImage = 0;
	mBitmapInfo->bmiHeader.biXPelsPerMeter = 0;
	mBitmapInfo->bmiHeader.biYPelsPerMeter = 0;
	mBitmapInfo->bmiHeader.biClrUsed = 0;
	mBitmapInfo->bmiHeader.biClrImportant = 0;

	for(int i = 0; i < 256; i++)
	{
		mBitmapInfo->bmiColors[i].rgbRed = mBitmapInfo->bmiColors[i].rgbGreen = mBitmapInfo->bmiColors[i].rgbBlue = i;
		mBitmapInfo->bmiColors[i].rgbReserved = 0;
	}

	return true;
}

bool BmpImage::ConvertToGreyScale(BmpImage& other)
{
	if(mData.size() == 0)
		return false;

	int bits_per_pixel = other.GetBitsPerPixel();

	for(int i = 0; i < other.GetHeight(); i++)
	{
		for(int j = 0; j < other.GetWidth(); j++)
		{
			BYTE pixel = 0;
			if(bits_per_pixel == 1)
			{
				pixel = other.GetPixel1(j, i) ? 255 : 0;
			}
			else if(bits_per_pixel == 8)
			{
				pixel = other.GetPixel8(j, i);
			}
			else if(bits_per_pixel == 24)
			{
				RGBTRIPLE triple = other.GetPixel24(j, i);
				pixel = (BYTE)(0.299 * triple.rgbtRed + 0.587 * triple.rgbtGreen + 0.114 * triple.rgbtBlue);
			}

			SetPixel8(j, i, pixel);
		}
	}

	return true;
}

std::vector<BYTE> BmpImage::GetRawData()
{
	std::vector<BYTE> rawData;

	if(GetBitsPerPixel() != 8)
		return rawData;

	for(size_t i = 0; i < GetHeight(); i++)
	{
		for(size_t j = 0; j < GetWidth(); j++)
		{
			rawData.push_back(GetPixel8(j, i));
		}
	}

	return rawData;
}

void BmpImage::SetRawData(std::vector<BYTE>& rawData)
{
	if(GetBitsPerPixel() != 8)
		return;

	for(size_t i = 0; i < GetHeight(); i++)
	{
		for(size_t j = 0; j < GetWidth(); j++)
		{
			SetPixel8(j, i, rawData[i * GetWidth() + j]);
		}
	}
}

int BmpImage::ParseRgbTripleToInt(RGBTRIPLE pixel){
    return (pixel.rgbtRed << 16) + (pixel.rgbtGreen << 8) + pixel.rgbtBlue;
}

RGBTRIPLE BmpImage::ParseIntToRgbTriple(int pixelValue){
    RGBTRIPLE pixel = 
    {
    	pixelValue & BLUE_MASK, // blue
    	(pixelValue & GREEN_MASK) >> 8, // green
    	(pixelValue & RED_MASK) >> 16 // red
    };
    return pixel;
}

