#!/bin/bash
pkill bench
pkill -f "bench"
#python3 -m torch.distributed.launch --nproc_per_node=4 --use_env /mnt/cube/home/rraina/segmentation_benchmark/semseg/backbone_val.py /mnt/cube/home/rraina/segmentation_benchmark/ImageNet -e --pretrained
python3 /mnt/cube/home/rraina/segmentation_benchmark/semseg/backbone_val.py -a resnet101_divnorm --dist-url 'tcp://127.0.0.1:6006' --dist-backend 'gloo' --multiprocessing-distributed --world-size 1 --rank 0  -e --pretrained /mnt/cube/home/rraina/segmentation_benchmark/ImageNet
pkill -f "bench"
pkill bench