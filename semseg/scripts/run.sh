#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 19020 \
        --use_env train.py \
        --lr 0.02 \
        --epochs 50 \
        --dataset voc_aug \
        -b 32 \
        --model deeplabv3 \
        --backbone "resnet50_eidivnorm" \
        --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset \
        --output "resnet50_eidivnorm_after_conv1_groups1" \
        --aux-loss

#python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 19008 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 32 --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output-dir resnet_divnormresnet50_divnorm_after_maxpool_after_layer1_groups=1_after_layer2_groups=1 --aux-loss
#python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 19008 --use_env eval_iou.py --lr 0.02 --epochs 50 --dataset voc_aug -b 32 --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output-dir resnet_divnormresnet50_divnorm_after_maxpool_after_layer1_groups=1_after_layer2_groups=1 --aux-loss

# # Evaluate Model
# python3 -m torch.distributed.launch --nproc_per_node=1"resnet50_divnormei_after_conv1_groups1""resnet50_divnormei_after_conv1_groups1""resnet50_divnormei_after_conv1_groups1" --use_env eval.py --lr 0.02 --dataset voc --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss

# # Prostprocessing CRF
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet50_crf
