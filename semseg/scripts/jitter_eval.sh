#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

for i in $(seq 1 1 20); do \
    max=$i
    if [ $i -eq 1 ]; then
        min=$i
    else
        min=$((max-1))
    fi

    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19020 --use_env eval_jitter.py \
         --data-path "/home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset" \
         --dataset "voc_aug" \
         --arch "deeplabv3" \
         --jitter "contrast" \
         --contrast-min $min \
         --contrast-max $max \
         --backbone "resnet50_eidivnorm" \
         --output-dir "/home/AD/rraina/segmentation_benchmark/semseg/results/"\
         --resume "/home/AD/rraina/segmentation_benchmark/semseg/results/resnet50_eidivnorm_deeplabv3/jtioiu/model_best.pth" \
         --aux-loss \


done