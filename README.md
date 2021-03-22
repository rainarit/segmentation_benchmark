# Image Segmentation Benchmark
## Prerequisites
Install all needed packages/libraries
```bash
pip install -r requirements.txt
```

## Dataset Used:
#### PASCAL Visual Object Classes
* Total number of classes: 20
* The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations.

## Models To Use:
* *fcn_resnet50* 
* *fcn_resnet101*
* *deeplabv3_resnet50*
* *deeplabv3_resnet101*
* *deeplabv3_mobilenet_v3_large*
* *lraspp_mobilenet_v3_large*
           

## How to Run:
### Generic Experiment:
```bash
python3 bench_PASCALVOC.py --model \
                           --pretrained \
                           --device \
                           --batch_size \
                           --workers \
```
### FCN-ResNet101 Example:
```bash
python3 bench_PASCALVOC.py --model fcn_resnet101 \
                           --pretrained True \
                           --device 'cuda' \
                           --batch_size 4 \
                           --workers 4
Selected Model: fcn_resnet101
Selected Pre-trained = True
------------------------------------------------------------------------------------
Downloaded fcn_resnet101 successfully!
------------------------------------------------------------------------------------
Downloading PASCAL VOC 2012 Validation Set
Downloaded PASCAL VOC 2012 Validation Set successfully!
------------------------------------------------------------------------------------
Training Model
Evaluating Model on Validation Set
---------------------Setting model to evaluation mode
---------------------Generating Confusion Matrix
---------------------Generating MetricLogger
  0%|                                                                                                                                  | 0/363 [00:00<?, ?it/s]Test:  [  0/363]  eta: 0:04:36    time: 0.7614  data: 0.3724  max mem: 334
 28%|█████████████████████████████████                                                                                       | 100/363 [00:05<00:12, 21.06it/s]Test:  [100/363]  eta: 0:00:14    time: 0.0484  data: 0.0014  max mem: 380
 55%|█████████████████████████████████████████████████████████████████▊                                                      | 199/363 [00:10<00:09, 17.67it/s]Test:  [200/363]  eta: 0:00:08    time: 0.0513  data: 0.0015  max mem: 395
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████▊                     | 299/363 [00:15<00:02, 21.38it/s]Test:  [300/363]  eta: 0:00:03    time: 0.0482  data: 0.0015  max mem: 395
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:18<00:00, 19.24it/s]
Test: Total time: 0:00:18
global correct: 90.7
average row correct: ['97.1', '77.2', '69.0', '79.5', '67.7', '70.0', '83.0', '77.0', '77.4', '41.6', '83.3', '43.7', '72.2', '82.5', '76.0', '83.3', '66.3', '81.3', '35.5', '76.9', '56.8']
IoU: ['89.8', '70.8', '31.8', '70.6', '58.9', '65.6', '80.6', '70.9', '71.1', '32.8', '78.0', '41.8', '62.3', '72.9', '63.1', '73.1', '55.8', '71.2', '31.3', '72.9', '52.5']
mean IoU: 62.8
```

## Results:
- *All models have been pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.*

| Model                        | Mean IoU | Global Accuracy | Class Accuracies                                                                                                                                                 | Class IoUs                                                                                                                                                       |
| ---------------------------- |:--------:|:---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| FCN-ResNet50                 | 72.8     | 00.0            |                                                                                                                                                                  |                                                                                                                                                                  |
| FCN-ResNet101                | 72.8     |   94.0          | ['97.2', '89.9', '86.4', '86.4', '82.1', '55.2', '91.7', '74.1', '92.0', '63.1', '91.2', '75.4', '86.4', '87.8', '83.3', '95.5', '65.7', '82.8', '72.7', '86.8'] | ['93.7', '88.3', '39.9', '82.4', '72.0', '52.0', '90.0', '68.6', '87.7', '42.7', '79.1', '61.5', '78.8', '77.7', '78.7', '89.4', '55.9', '79.4', '53.2', '84.0'] |
| Deeplabv3-ResNet50           | 75.6     |   94.4.         | ['96.3', '92.1', '88.2', '95.5', '83.8', '62.4', '98.8', '74.3', '94.7', '67.2', '91.5', '70.4', '92.3', '92.1', '95.0', '95.3', '74.2', '93.9', '79.4', '93.3'] | ['93.7', '90.0', '41.2', '87.1', '70.0', '56.8', '94.3', '65.5', '92.1', '46.8', '86.8', '53.5', '85.9', '81.0', '87.6', '88.8', '63.2', '88.5', '54.0', '85.6'] | 
| Deeplabv3-ResNet101          | 77.8     |   95.0          | -----:|
| DeepLabV3 MobileNet V3 Large | are neat |    $1           | -----:|
| LRASPP MobileNet V3 Large    | are neat |    $1           | -----:|


## License ![GitHub](https://img.shields.io/github/license/rainarit/segmentation-benchmark)

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/rainarit/segmentation-benchmark/blob/main/LICENSE) file for details
