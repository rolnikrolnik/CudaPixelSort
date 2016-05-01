#! /bin/bash
if [[ $1 = "" ]]; 
then
    echo 'Not enough arguments. Example usage: run.sh [filename]'
    exit 
fi

TEMP_PIXELS=temp_pixels
python processImage.py `dirname $0` $1 $TEMP_PIXELS
exit