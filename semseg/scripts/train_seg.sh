#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 train.py \
         --lr 0.02 \
         --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
         --dataset "voc_aug" \
         -b 32 \
         --epochs 50 \
         --arch "deeplabv3" \
         --backbone "resnet50_divnormei" \
         --output-dir "/home/AD/rraina/segmentation_benchmark/semseg/results/"\
         --aux-loss \
         --amp \