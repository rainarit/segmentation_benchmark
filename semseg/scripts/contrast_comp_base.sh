#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

for i in $(seq 5.2 0.1 20); do \
    name="resnet50_contrast${i}_test"
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19030 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet50/checkpoint_49.pth --output $name --aux-loss --test-only; \
done