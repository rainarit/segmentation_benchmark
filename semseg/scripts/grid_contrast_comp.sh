#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

for i in $(seq 10 2 200); do \
    contrast=5.0
    name="resnet50_contrast5.0_random${i}x${i}block_test"
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19000 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $contrast --grid-size $i --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet50/checkpoint_49.pth --output $name --aux-loss --test-only; \
done