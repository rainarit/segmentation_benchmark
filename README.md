Status: Active (updates are to be expected)

# Image Segmentation Benchmark Documentation
## Prerequisites
#### Install Git Repository
```bash
git clone https://github.com/rainarit/segmentation_benchmark.git
cd segmentation_benchmark
```
#### Create Conda Environment
The code in the tutorial has been written using Python 3; many of the dependencies may not be available for Python 2.7.

We **strongly recommend** using the Anaconda Python distribution. You can install either the full [anaconda distribution](https://www.continuum.io/downloads) (very extensive, but large) or [miniconda](https://conda.io/miniconda.html) (much smaller, only essential packages).

Create a new conda environment with the use of the package-list.txt
```bash
conda create --name <env> --file package-list.txt
```

## Run Training Script(s):
#### Download COCO Dataset:
```bash
./coco_download.sh
```
#### Run Training Script:
```bash
./train_fcn_resnet50_coco.sh
```

## License ![GitHub](https://img.shields.io/github/license/rainarit/segmentation_benchmark)

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/rainarit/segmentation-benchmark/blob/main/LICENSE) file for details


