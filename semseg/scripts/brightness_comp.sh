#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

for j in $(seq 2 1 5); do

    for i in $(seq 1 1 20); do 

        contrast1=$i
        if [ $i -eq 1 ]; then
            contrast2=$i
        else
            contrast2=$((contrast1-1))
        fi

        echo $contrast2,$contrast1
    
        python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 10211 \
            --use_env jitter_eval.py \
            --lr 0.02 \
            --dataset voc_aug --aux-loss\
            --brightness $contrast1 \
            --model deeplabv3 \
            --backbone "resnet50_divnorm" \
            --output "resnet50_divnorm_after_conv1_groups1_brightness_${j}(${contrast2},${contrast1})" \
            --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
            --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/output/resnet50_divnorm_after_conv1_groups1/checkpoints/checkpoint_49.pth"

        python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 10213 \
            --use_env jitter_eval.py \
            --lr 0.02 \
            --dataset voc_aug --aux-loss\
            --brightness $contrast1 \
            --model deeplabv3 \
            --backbone "resnet50" \
            --output "resnet50_brightness_${j}(${contrast2},${contrast1})" \
            --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
            --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/output/resnet50/checkpoints/checkpoint_49.pth"

        python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 10213 \
            --use_env jitter_eval.py \
            --lr 0.02 \
            --dataset voc_aug --aux-loss\
            --brightness $contrast1 \
            --model deeplabv3 \
            --backbone "resnet50_divnormei" \
            --output "resnet50_divnormei_after_conv1_groups1_brightness_${j}(${contrast2},${contrast1})" \
            --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
            --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/output/resnet50_divnormei_after_conv1_groups1/checkpoints/checkpoint_49.pth"

    done

done

# for i in $(seq 1 1 20); do 

#     # contrast1=$i
#     # contrast1=${contrast1/#-./-0.}
#     # contrast1=${contrast1/#./0.}


#     # if [ $contrast1 == "-0.5" ]; then
#     #     contrast2=$i
#     # else
#     #     #contrast2="$(echo $contrast1 - $initial | bc)"
#     #     contrast2="$(echo "scale=1;$contrast1-0.1" | bc)"
#     #     contrast2=${contrast2/#-./-0.}
#     #     contrast2=${contrast2/#.0/0.}
#     #     contrast2=${contrast2/#./0.}
#     # fi


#     contrast1=$i
#     if [ $i -eq 1 ]; then
#         contrast2=$i
#     else
#         contrast2=$((contrast1-1))
#     fi

#     echo $contrast2,$contrast1

#     # python3 -m torch.distributed.launch \
#     #         --nproc_per_node=1 \
#     #         --master_port 20210 \
#     #         --use_env jitter_eval.py \
#     #         --lr 0.02 \
#     #         --dataset voc_aug --aux-loss\
#     #         --brightness $contrast1 \
#     #         --model deeplabv3 \
#     #         --backbone "resnet50_eidivnorm" \
#     #         --output "resnet50_eidivnorm_after_conv1_groups1_brightness(${contrast2},${contrast1})" \
#     #         --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
#     #         --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/output/resnet50_eidivnorm_after_conv1_groups1/checkpoints/checkpoint_49.pth"

#     python3 -m torch.distributed.launch \
#             --nproc_per_node=1 \
#             --master_port 20211 \
#             --use_env jitter_eval.py \
#             --lr 0.02 \
#             --dataset voc_aug --aux-loss\
#             --brightness $contrast1 \
#             --model deeplabv3 \
#             --backbone "resnet50_divnorm" \
#             --output "resnet50_divnorm_after_conv1_groups1_brightness(${contrast2},${contrast1})" \
#             --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
#             --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/output/resnet50_divnorm_after_conv1_groups1/checkpoints/checkpoint_49.pth"
    
# done








