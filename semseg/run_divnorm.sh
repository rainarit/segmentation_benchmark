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
export CUDA_VISIBLE_DEVICES=0,1
pkill bench
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.032659863 --epoch 80 --dataset voc -b 16 --model deeplabv3 --backbone resnet_divnormresnet50 --data-path /home/AD/rraina/segmentation_benchmark --tensorboard-dir deeplabv3_resnet_divnormresnet50 --aux-loss
pkill bench
# # Evaluate Model
# pkill bench
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