#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
pkill -f bench
python3 /mnt/cube/home/rraina/segmentation_benchmark/semseg/backbone_train_imagenetc.py -a resnet50 --dist-url 'tcp://127.0.0.1:6006' --dist-backend 'gloo' --multiprocessing-distributed --world-size 1 --rank 0 --train_data /home/AD/rraina/segmentation_benchmark/ImageNet/ --val_data /home/AD/rraina/segmentation_benchmark/ImageNetC/contrast/1/  -e
pkill -f bench