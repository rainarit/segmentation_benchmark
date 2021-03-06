#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

# Running segmentation

# torchrun --nproc_per_node=2 train.py \
#          --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset \
#          --lr 0.02 \
#          --epochs 50 \
#          --dataset voc_aug \
#          -b 16 \
#          --arch deeplabv3 \
#          --backbone resnet50_eidivnorm \
#          --output deeplabv3_eidivnorm_resnet50 \
#          --aux-loss

torchrun --nproc_per_node=1 \
         train.py \
         --lr 0.02 \
         --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
         --dataset "voc_aug" \
         -b 16 \
         --epochs 50 \
         --arch "deeplabv3" \
         --backbone "resnet50" \
         --output-dir "/home/AD/rraina/segmentation_benchmark/semseg/results/"\
         --aux-loss \
         --amp \
         --master_port 47720 \

# Running object detection

#python3 backbone_train.py -a resnet50_divnormei --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --data "/mnt/cube/projects/imagenet_100" --output-dir 'resnet50_divnormei_imagenet_100'