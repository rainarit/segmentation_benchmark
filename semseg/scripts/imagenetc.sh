#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3

model_vgg9_divnorm_mini="vgg9_divnorm_mini"
model_vgg9_divnorm_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/rkbesy/checkpoint_best_vgg9_divnorm_mini.pth"

model_vgg9_eidivnorm_mini="vgg9_eidivnorm_mini"
model_vgg9_eidivnorm_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/sxphdx/checkpoint_best_vgg9_eidivnorm_mini.pth"

model_vgg9_dalernn_mini="vgg9_dalernn_mini"
model_vgg9_dalernn_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/uszrla/checkpoint_best_vgg9_dalernn_mini.pth"

model_vgg9_ctrl="vgg9_ctrl"
model_vgg9_ctrl_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/dlkoiv/checkpoint_best_vgg9_ctrl.pth"


model=${model_vgg9_dalernn_mini}
path=${model_vgg9_dalernn_mini_path}
for i in $(seq 1 2 5); do \
    pertub="brightness"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:7009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="defocus_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:7009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="contrast"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:7009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="gaussian_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:7009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="elastic_transform"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="fog"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="frost"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="gaussian_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="glass_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="impulse_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="jpeg_compression"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="motion_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="pixelate"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="saturate"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="shot_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="snow"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="spatter"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="speckle_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="zoom_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${path} \
                            --dist-url 'tcp://127.0.0.1:9008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

done