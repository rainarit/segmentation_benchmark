#!/usr/bin/env bash

# # FCN_ResNet101
# # Train model
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 0.02 --dataset voc -b 8 --model fcn --backbone resnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet101 --aux-loss

# # Evaluate Model
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model fcn --backbone resnet101 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss

# # Prostprocessing CRF
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model fcn --backbone resnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet101_crf

# DeepLabV3_ResNet50
# Train model
export CUDA_VISIBLE_DEVICES=1
#pkill bench
#python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 19090 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 32 --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output-dir resnet_50_test --aux-loss --test-only
#pkill bench
#python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 19008 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 32 --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output-dir resnet_divnormresnet50_divnorm_after_maxpool_after_layer1_groups=1_after_layer2_groups=1 --aux-loss
#python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 19008 --use_env eval_iou.py --lr 0.02 --epochs 50 --dataset voc_aug -b 32 --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output-dir resnet_divnormresnet50_divnorm_after_maxpool_after_layer1_groups=1_after_layer2_groups=1 --aux-loss

# # Evaluate Model
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss

# # Prostprocessing CRF
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet50_crf
