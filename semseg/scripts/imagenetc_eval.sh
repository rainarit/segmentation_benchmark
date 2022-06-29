#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3

model_vgg9_divnorm_mini="vgg9_divnorm_mini"
model_vgg9_divnorm_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/rkbesy/checkpoint_best_vgg9_divnorm_mini.pth"

model=${model_vgg9_divnorm_mini}
path=${model_vgg9_divnorm_mini_path}

for i in $(seq 1 2 5); do \
    pertub="brightness"
    python3 eval_imagenetc.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/ \
                            --pertubation ${pertub} \
                            --level ${i} \
                            --num-classes 100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:7009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
done