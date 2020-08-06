#!/bin/bash
#echo $1;
FN=$(basename $2)
SIZE=$1
TARGET=~/scratch/PC/images2-$SIZE/$FN

if [ -z "$SIZE" ]
then
    echo "\$SIZE is empty"
    exit
fi

if test -f "$TARGET"; then
    echo "$TARGET exist"
else
    echo "processing $FN"
    convert -resize ${SIZE}x${SIZE}^ -gravity center -extent ${SIZE}x${SIZE} $2 $TARGET
fi
