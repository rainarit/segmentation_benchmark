"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation

datasets = {
    'coco': COCOSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)