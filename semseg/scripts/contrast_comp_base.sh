#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

for i in $(seq 1 1 20); do \
    contrast1=$i
    if [ $i -eq 1 ]; then
        contrast2=$i
    else
        contrast2=$((contrast1-1))
    fi

    echo $contrast2,$contrast1
    python3 -m torch.distributed.launch \
            --nproc_per_node=2 \
            --master_port 10213 \
            --use_env jitter_eval.py \
            --lr 0.02 \
            --dataset voc_aug --aux-loss\
            --sigma $contrast1 \
            --model deeplabv3 \
            --backbone resnet50_eidivnorm \
            --output "deeplabv3_eidivnorm_resnet50_sigma_(${contrast2},${contrast1})" \
            --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
            --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/outputs/deeplabv3_eidivnorm_resnet50/epoch_49/model_49.pth"

    python3 -m torch.distributed.launch \
            --nproc_per_node=2 \
            --master_port 10213 \
            --use_env jitter_eval.py \
            --lr 0.02 \
            --dataset voc_aug --aux-loss\
            --sigma $contrast1 \
            --model deeplabv3 \
            --backbone resnet50 \
            --output "deeplabv3_resnet50_sigma_(${contrast2},${contrast1})" \
            --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
            --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/outputs/deeplabv3_resnet50/epoch_49/model_49.pth"
done