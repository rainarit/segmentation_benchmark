#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 19020 \
        --use_env eval.py \
        --lr 0.02 \
        --epochs 50 \
        --dataset voc_aug \
        --model "deeplabv3" \
        --backbone "resnet50_eidivnorm" \
        --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
        --resume "/home/AD/rraina/segmentation_benchmark/semseg/outputs/deeplabv3_eidivnorm_resnet50/epoch_49/model_49.pth" \
        --aux-loss
