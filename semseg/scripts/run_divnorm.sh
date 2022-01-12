# FCN_ResNet101_DivNorm
# Train model
# pkill bench
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 0.02 --dataset voc -b 8 --model fcn --backbone resnet_divnormresnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet_divnormresnet101 --aux-loss
# pkill bench
# # Evaluate Model
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model fcn --backbone resnet_divnormresnet101 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss
# pkill bench
# # Prostprocessing CRF
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model fcn --backbone resnet_divnormresnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet_divnormresnet101_crf
# pkill bench

# # FCN_ResNet50_DivNorm
# # Train model
# pkill bench
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 0.02 --dataset voc -b 8 --model fcn --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet_divnormresnet50 --aux-loss
# pkill bench
# # Evaluate Model
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model fcn --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss
# pkill bench
# # Prostprocessing CRF
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model fcn --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir fcn_resnet_divnormresnet50_crf
# pkill bench

# # DeepLabV3_ResNet50_DivNorm
# # Train model
#pkill bench
#python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset voc_aug -b 16 --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --tensorboard-dir deeplabv3_resnet_divnormresnet50 --aux-loss
#pkill bench

export CUDA_VISIBLE_DEVICES=2,3

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 19020 --use_env train.py --lr 0.02 --epochs 50 --dataset voc_aug -b 16 --model deeplabv3 --backbone resnet50_divnorm --data-path /home/AD/rraina/segmentation_benchmark/benchmark_RELEASE/dataset --output resnet50_divnorm_after_conv1_groups1 --aux-loss


# --resume /home/AD/rraina/segmentation_benchmark/semseg/resnet_divnormresnet50_divnorm_after_maxpool_after_layer1_after_layer2_groups=1/best_checkpoint.pth --test-only
# # Evaluate Model
# pkill -f bench
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss
# pkill bench
# # Prostprocessing CRF
# pkill bench
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet_divnormresnet50_crf
# pkill bench

# # DeepLabV3_ResNet101_DivNorm
# # Train model
# pkill bench
# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --lr 0.02 --dataset voc -b 8 --model deeplabv3 --backbone resnet_divnormresnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet_divnormresnet101 --aux-loss
# pkill bench
# # Evaluate Model
# pkill bench
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --lr 0.02 --dataset voc --model deeplabv3 --backbone resnet_divnormresnet101 --data-path /home/AD/rraina/segmentation_benchmark --aux-loss
# pkill bench
# # Prostprocessing CRF
# pkill bench
# python3 -m torch.distributed.launch --nproc_per_node=1 --use_env crf.py --dataset voc --model deeplabv3 --backbone resnet_divnormresnet101 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet_divnormresnet101_crf
# pkill bench