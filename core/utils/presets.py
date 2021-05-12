from .transforms import *

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
        trans.extend([
            RandomCrop(crop_size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Compose([
            RandomResize(base_size, base_size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)