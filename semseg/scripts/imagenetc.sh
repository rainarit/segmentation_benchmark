#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

model="vgg9_divnorm_mini"
model_vgg9_divnorm_mini="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/zgkbjl/checkpoint_best_vgg9_divnorm_mini.pth"

for i in $(seq 1 1 5); do \
    pertub="defocus_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    break

    pertub="elastic_transform"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="fog"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="frost"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="gaussian_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="glass_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="impulse_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="jpeg_compression"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="motion_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="pixelate"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="saturate"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="shot_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="snow"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \
    
    pertub="spatter"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="speckle_noise"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

    pertub="zoom_blur"
    python3 imagenetc_eval.py --arch ${model} \
                            --data /home/AD/rraina/segmentation_benchmark/ImageNetC/${pertub}/${i}/ \
                            --pertubation ${model}_imagenetc_${pertub}${i}_100 \
                            --num-classes 100 \
                            --output /home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenetc_${pertub}${i}_100 \
                            --checkpoint ${model_vgg9_divnorm_mini} \
                            --dist-url 'tcp://127.0.0.1:8008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0; \

done