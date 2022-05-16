#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

for i in $(seq 0.2 0.1 1); do \
    high=$i
    high=${high/#-./-0.}
    high=${high/#./0.}

    if [ $high == "0.0" ]; then
        low=$i
    else
        low="$(echo "scale=1;$high-0.1" | bc)"
    fi

    echo $low,$high

    python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 10213 \
            --use_env jitter_eval.py \
            --lr 0.02 \
            --dataset voc_aug --aux-loss\
            -ol $high \
            --model deeplabv3 \
            --backbone resnet50_eidivnorm \
            --output "deeplabv3_eidivnorm_resnet50_occlusion_(${low},${high})" \
            --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset  \
            --checkpoint "/home/AD/rraina/segmentation_benchmark/semseg/outputs/deeplabv3_eidivnorm_resnet50/epoch_47/model_47.pth"
done