export CUDA_VISIBLE_DEVICES=2,3
pkill -f bench
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19020 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --model deeplabv3 --backbone resnet50_divnorm --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output resnet50_divnorm_after_conv1_groups1 --aux-loss
pkill -f bench