#!/bin/bash

#SBATCH --time=10:0:0
#SBATCH --mem=12g
#SBATCH --ntasks-per-node=32

SIZE=$1

if [ -z "$SIZE" ]
then
    echo "\$SIZE is empty"
    exit
fi

find images -type f | xargs -P 32 -n 1 ./convert-single.sh $SIZE 

echo "Done"
