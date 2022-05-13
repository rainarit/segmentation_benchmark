#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3

model_vgg9_divnorm_mini="vgg9_divnorm_mini"
model_vgg9_divnorm_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/rkbesy/checkpoint_best_vgg9_divnorm_mini.pth"

model_vgg9_eidivnorm_mini="vgg9_eidivnorm_mini"
model_vgg9_eidivnorm_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/sxphdx/checkpoint_best_vgg9_eidivnorm_mini.pth"

model_vgg9_dalernn_mini="vgg9_dalernn_mini"
model_vgg9_dalernn_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/uszrla/checkpoint_best_vgg9_dalernn_mini.pth"

model_vgg9_ctrl="vgg9_ctrl"
model_vgg9_ctrl_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/dlkoiv/checkpoint_best_vgg9_ctrl.pth"

model=${model_vgg9_dalernn_mini}
path=${model_vgg9_dalernn_mini_path}

python3 foolbox_attack.py --arch ${model} \
                           --data /mnt/cube/projects/imagenet_100/val/ \
                           --attack LinfFastGradientAttack \
                           --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_foolbox_imagenet_100 \
                           --checkpoint ${path} \
                           --dist-url 'tcp://127.0.0.1:19008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
