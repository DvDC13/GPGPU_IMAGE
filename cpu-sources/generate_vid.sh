#!/bin/bash

mkdir -p dataset/video_frames
rm -rf dataset/video_frames/*.png

filename="dataset/video_frames/frame"

ffmpeg -i "$1" $filename%03d.png

mkdir -p result/
rm -rf result/*