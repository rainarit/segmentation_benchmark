#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

for i in $(seq 1 1 5); do \
    pertub="defocus_blur"
    python3 imagenetc_eval.py --arch vgg9_ctrl \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation vgg9_ctrl_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/vgg9_ctrl_imagenetc_${pertub}${i}_100 \
                            --checkpoint /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/dlkoiv/checkpoint_best_vgg9_ctrl.pth \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="elastic_transform"
    python3 imagenetc_eval.py --arch vgg9_ctrl \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/contrast/${i}/ \
                            --pertubation vgg9_ctrl_imagenetc_contrast${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/vgg9_ctrl_imagenetc_contrast${i}_100 \
                            --checkpoint /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/dlkoiv/checkpoint_best_vgg9_ctrl.pth \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="fog"
    python3 imagenetc_eval.py --arch vgg9_ctrl \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/gaussian_blur/${i}/ \
                            --pertubation vgg9_ctrl_imagenetc_gaussian_blur${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/vgg9_ctrl_imagenetc_gaussian_blur${i}_100 \
                            --checkpoint /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/dlkoiv/checkpoint_best_vgg9_ctrl.pth \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
done