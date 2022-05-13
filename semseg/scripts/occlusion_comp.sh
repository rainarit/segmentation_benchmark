#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

model_vgg9_eidivnorm_mini="vgg9_eidivnorm_mini"
model_vgg9_eidivnorm_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/sxphdx/checkpoint_best_vgg9_eidivnorm_mini.pth"

model_vgg9_dalernn_mini="vgg9_dalernn_mini"
model_vgg9_dalernn_mini_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/uszrla/checkpoint_best_vgg9_dalernn_mini.pth"

model_vgg9_ctrl="vgg9_ctrl"
model_vgg9_ctrl_path="/home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/dlkoiv/checkpoint_best_vgg9_ctrl.pth"

list_models=($model_vgg9_ctrl $model_vgg9_dalernn_mini $model_vgg9_eidivnorm_mini)
list_checkpoints=($model_vgg9_ctrl_path $model_vgg9_dalernn_mini_path $model_vgg9_eidivnorm_mini_path)

for j in $(seq 2 1 2); do \
    model=${list_models[$j]}
    path=${list_checkpoints[$j]}

    for i in $(seq 0.0 0.1 1); do \
        high=$i
        high=${high/#-./-0.}
        high=${high/#./0.}

        if [ $high == "0.0" ]; then
            low=$i
        else
            low="$(echo "scale=1;$high-0.1" | bc)"
        fi

        output_path=/home/AD/rraina/segmentation_benchmark/semseg/outputs/${model}_imagenet_100_occlusion_${low}_${high}.csv

        python3 /home/AD/rraina/segmentation_benchmark/semseg/backbone_eval_vgg_occlusion.py \
            --data '/mnt/cube/projects/imagenet_100/' \
            --arch $model \
            --batch-size 256 \
            --print-freq 1 \
            --occlude_low $low \
            --occlude_high $high \
            --resume $path \
            --gpu 1 \
            --output $output_path
    done;

done;
