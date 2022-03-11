#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3

torchrun --nproc_per_node=2 train.py \
         --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset \
         --lr 0.02 \
         --epochs 50 \
         --dataset voc_aug \
         -b 16 \
         --arch deeplabv3 \
         --backbone resnet50 \
         --output deeplabv3_resnet50 \
         --aux-loss