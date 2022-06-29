#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 eval_jitter.py \
         --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
         --dataset "voc_aug" \
         --arch "deeplabv3" \
         --jitter "contrast" \
         --level-min 0.0 \
         --level-max 1.0 \
         --backbone "resnet50_divnorm" \
         --output-dir "/home/AD/rraina/segmentation_benchmark/semseg/results/"\
         --resume "/home/AD/rraina/segmentation_benchmark/semseg/results/resnet50_deeplabv3/ukpkfw/model_best.pth" \
         --aux-loss \