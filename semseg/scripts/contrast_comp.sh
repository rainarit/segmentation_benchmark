#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

for i in $(seq 7.6 0.1 20.0); do \
    name="resnet_divnormresnet50_divnorm_after_conv1_groups=1_contrast${i}"
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19020 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet_divnormresnet50_divnorm_before_maxpool_groups=1/checkpoint_49.pth --output $name --aux-loss --test-only; \
done