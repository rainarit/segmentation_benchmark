#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python3 foolbox_attack.py --arch vgg9_dalernn_mini \
                          --data /mnt/cube/projects/imagenet_100/val/ \
                          --attack LinfFastGradientAttack \
                          --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/vgg9_dalernn_mini_foolbox_imagenet_100 \
                          --checkpoint /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/uszrla/checkpoint_best_vgg9_dalernn_mini.pth \
                          --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \

