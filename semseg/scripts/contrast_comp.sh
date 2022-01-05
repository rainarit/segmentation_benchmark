#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# for i in $(seq 5.1 0.1 20.0); do \
#      python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19020 --use_env colorjitter_random_dataloader.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet_divnormresnet50_divnorm_before_maxpool_groups=1/checkpoint_49.pth --aux-loss --test-only; \
# done

# for i in $(seq 5.4 0.1 20.0); do \
#     name="resnet_divnormresnet50_divnorm_after_conv1_groups=1_after_layer1_groups=1_after_layer2_groups=1_contrast${i}"
#     python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19020 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet_divnormresnet50_divnorm_before_maxpool_groups=1_after_layer1_groups=1_after_layer2_groups=1/checkpoint_49.pth --output $name --aux-loss --test-only; \
# done

for i in $(seq 20.0 1.0 20.0); do \
    name1="resnet_divnormresnet50_divnorm_after_conv1_groups=1_contrast(1.0,${i})"
    name2="resnet50_contrast(1.0,${i})"
    dataloader="/home/AD/rraina/segmentation_benchmark/semseg/dataloaders/dataloader_test_contrast(1.0,${i}).pth"
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 16010 --use_env colorjitter_random_dataloader.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet_divnormresnet50_divnorm_before_maxpool_groups=1/checkpoint_49.pth --aux-loss --test-only; \
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 16020 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --dataloader $dataloader --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet_divnormresnet50_divnorm_before_maxpool_groups=1/checkpoint_49.pth --output $name1 --aux-loss --test-only; \
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 16030 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --contrast $i --dataloader $dataloader --model deeplabv3 --backbone resnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --resume /home/AD/rraina/segmentation_benchmark/semseg/output/resnet50/checkpoint_49.pth --output $name2 --aux-loss --test-only; \
done