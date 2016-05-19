#! /bin/bash
if [[ $1 = "" ]]; 
then
    echo 'Not enough arguments. Example usage: run.sh [filename]'
    exit 
fi
    CHUNK_SIZE=300
    ./apl_project $1 1 1 $CHUNK_SIZE 
    ./apl_project $1 1 8 $CHUNK_SIZE
    ./apl_project $1 1 16 $CHUNK_SIZE
    ./apl_project $1 1 64 $CHUNK_SIZE
    ./apl_project $1 1 128 $CHUNK_SIZE
    ./apl_project $1 1 256 $CHUNK_SIZE
exit