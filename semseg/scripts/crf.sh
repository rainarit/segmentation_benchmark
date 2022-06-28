#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3

python3 crf.py \
         --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
         --dataset "voc_aug" \
         --arch "deeplabv3" \
         --backbone "resnet50" \
         --output-dir "/home/AD/rraina/segmentation_benchmark/semseg/results/"\
         --seed-dir "ukpkfw"\