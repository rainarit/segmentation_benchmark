import transforms as T

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            #T.ColorJitter(contrast=(0,10)),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        aug_img, target = self.transforms(img, target)

        return aug_img, target
        #return aug_img, img

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.ColorJitter(brightness=0, contrast=(0, 0.8), saturation=0, hue=0),
        ])

    def __call__(self, img, target):
        aug_img, target = self.transforms(img, target)
        
        return aug_img, target