export CUDA_VISIBLE_DEVICES=2
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 19010 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 32 --model deeplabv3 --backbone resnet50_divnorm --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output resnet50_divnorm_after_conv1_groups1 --aux-loss
