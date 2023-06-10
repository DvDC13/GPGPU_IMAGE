#!/bin/sh

mkdir -p dataset/video_frames/
mkdir -p dataset/results/
rm -rf dataset/video_frames/*.png
rm -rf dataset/results/*.png

filename="dataset/video_frames/frame"

ffmpeg -hide_banner -loglevel error -i "$1" $filename%05d.png

./build/gpgpu dataset/video_frames/

ffmpeg -hide_banner -loglevel error -framerate 30 -i "dataset/results/mask_%05d.png" -c:v libx264 -pix_fmt yuv420p out.mp4
