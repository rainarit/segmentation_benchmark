#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 10213 \
        --use_env jitter_eval.py \
        --lr 0.02 \
        --dataset voc_aug --aux-loss\
        --seed 4299 \
        --model deeplabv3 \
        --backbone resnet50_eidivnorm \
        --output "deeplabv3_eidivnorm_resnet50_occlusion_(${low},${high})" \
        --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
        --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/outputs/deeplabv3_eidivnorm_resnet50/epoch_47/model_47.pth"