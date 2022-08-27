#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 20220 \
            --use_env train.py \
            --lr 0.02 \
            --data-path "/mnt/sphere/projects/VOC_semseg/dataset" \
            --dataset "voc_aug" \
            -b 16 \
            --epochs 50 \
            --divnorm-fsize 3 \
            --arch "deeplabv3" \
            --backbone "resnet50_dalernn" \
            --output-dir "/home/vveeraba/src/segmentation_benchmark/semseg/results/"\
            --aux-loss \
            --amp \

#--divnorm-fsize 5 for divnormei/eidivnorm
