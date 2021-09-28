#!/bin/bash
pkill bench
pkill -f "bench"
#export CUDA_VISIBLE_DEVICES=1
python3 /mnt/cube/home/rraina/segmentation_benchmark/semseg/backbone_val.py -a resnet50_divnorm --dist-url 'tcp://127.0.0.1:6006' --dist-backend 'gloo' --multiprocessing-distributed --world-size 1 --rank 0 /mnt/cube/home/rraina/segmentation_benchmark/ImageNet
pkill -f "bench"
pkill bench