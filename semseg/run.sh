#!/usr/bin/env bash

trap "set +x; sleep 10; set -x" DEBUG

# FCN_ResNet101
# Train model
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 0.02 --dataset voc -b 8 --model fcn --backbone resnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet101 --aux-loss

# Evaluate Model
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model fcn --backbone resnet101 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss

# Prostprocessing CRF
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model fcn --backbone resnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet101_crf

# DeepLabV3_ResNet50
# Train model
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 0.02 --dataset voc -b 8 --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet50 --aux-loss

# Evaluate Model
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss

# Prostprocessing CRF
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet50_crf
