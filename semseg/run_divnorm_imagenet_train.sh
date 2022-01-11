#!/bin/bash
python backbone_train.py -a $1 --dist-url 'tcp://localhost:58472' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /mnt/cube/home/rraina/segmentation_benchmark/ImageNet --epochs 90 --batch-size 2048 --inplanes 64
