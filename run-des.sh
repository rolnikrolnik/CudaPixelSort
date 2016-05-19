#! /bin/bash
if [[ $1 = "" ]]; 
then
    echo 'Not enough arguments. Example usage: run.sh [filename]'
    exit 
fi
    CHUNK_SIZE=300
    ./des_project $1 0 1 $CHUNK_SIZE 
    ./des_project $1 0 8 $CHUNK_SIZE
    ./des_project $1 0 16 $CHUNK_SIZE
    ./des_project $1 0 64 $CHUNK_SIZE
    ./des_project $1 0 128 $CHUNK_SIZE
    ./des_project $1 0 256 $CHUNK_SIZE
exit