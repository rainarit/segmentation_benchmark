#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=1 \
         train.py \
         --lr 0.02 \
         --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
         --dataset "voc_aug" \
         -b 16 \
         --epochs 50 \
         --arch "deeplabv3" \
         --backbone "resnet50_eidivnorm" \
         --output-dir "/home/AD/rraina/segmentation_benchmark/semseg/results/"\
         --aux-loss \
         --amp \