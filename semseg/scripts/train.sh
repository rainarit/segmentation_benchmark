#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

# Runing segmentation

# torchrun --nproc_per_node=2 train.py \
#          --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset \
#          --lr 0.02 \
#          --epochs 50 \
#          --dataset voc_aug \
#          -b 16 \
#          --arch deeplabv3 \
#          --backbone resnet50 \
#          --output deeplabv3_resnet50 \
#          --aux-loss

# Runing object detection

python3 backbone_train.py -a resnet50_divnormei --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data "/mnt/cube/projects/imagenet_100" --output-dir 'resnet50_divnormei_imagenet_100'

