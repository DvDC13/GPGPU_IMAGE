#!/bin/bash

mkdir -p frames
rm -rf frames/*.png

filename="frames/frame"

ffmpeg -i "$1" $filename%03d.png



