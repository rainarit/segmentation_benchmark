# Image Segmentation Benchmark
## Prerequisites
Install Git Repository
```bash
git clone https://github.com/rainarit/segmentation-benchmark.git
cd segmentation-benchmark
```

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
Evaluating Model on Validation Set
---------------------Setting model to evaluation mode
---------------------Generating Confusion Matrix
---------------------Generating MetricLogger
  0%|                                                                                                                                                                                  | 0/91 [00:00<?, ?it/s]
Test:  [ 0/91]  eta: 0:03:04    time: 2.0292  data: 1.4526  max mem: 776
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:18<00:00,  4.90it/s]
Test: Total time: 0:00:18
global correct: 90.9
average row correct: ['97.3', '74.6', '69.0', '79.3', '66.7', '72.6', '84.1', '77.6', '77.0', '40.0', '82.7', '39.0', '73.5', '82.9', '75.7', '82.7', '65.8', '80.4', '35.3', '76.3']
IoU: ['90.1', '68.8', '31.9', '70.4', '59.7', '67.8', '82.3', '71.9', '71.2', '32.4', '77.8', '37.6', '63.0', '74.0', '62.8', '72.9', '55.7', '70.9', '31.8', '72.5']
mean IoU: 63.3
```

## Results:
- *All models have been pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.*



## License ![GitHub](https://img.shields.io/github/license/rainarit/segmentation-benchmark)

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/rainarit/segmentation-benchmark/blob/main/LICENSE) file for details
