#!/bin/bash

mkdir -p dataset/video_frames/
mkdir -p dataset/results/
rm -rf dataset/video_frames/*.png
rm -rf dataset/results/*.png

filename="dataset/video_frames/frame"

ffmpeg -i "$1" $filename%03d.png 2> /dev/null

time ./build/gpgpu dataset/video_frames/

ffmpeg -framerate 30 -pattern_type glob -i 'dataset/results/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4 2> /dev/null
