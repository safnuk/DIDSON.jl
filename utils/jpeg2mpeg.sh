#!/bin/bash

echo $1 $2
mkdir -pv $2
for f in $1/*
do
    [ -e "$f" ] || continue
    width=$(identify -format "%w" ${f}/00001.jpg)
    height=$(identify -format "%h" ${f}/00001.jpg)
    file=$(basename "${f}")
    echo Creating $2/${file}.avi
    mencoder mf://${f}/*.jpg -mf w=${width}:h=${height}:fps=10:type=jpg -ovc \
        lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy \
        -o $2/${file}.avi
done
