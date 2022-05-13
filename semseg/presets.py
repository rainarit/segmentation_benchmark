import torch
import transforms as T

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), contrast=1, brightness=1, sigma=1):
        self.contrast_initial = contrast
        self.contrast_final = contrast
        if (contrast == 1):
            self.contrast_final = contrast
        else:
            self.contrast_final = self.contrast_initial-1

        self.brightness_initial = brightness
        self.brightness_final = brightness
        if (brightness == 1):
            self.brightness_final = brightness
        else:
            self.brightness_final = self.brightness_initial-1

        self.sigma_initial = sigma
        self.sigma_final = sigma
        if (sigma == 1):
            self.sigma_final = sigma
        else:
            self.sigma_final = self.sigma_initial-1

        print("Contrast: ({}, {})".format(self.contrast_final, self.contrast_initial))
        print("Brightness: ({}, {})".format(self.brightness_final, self.brightness_initial))
        print("Sigma: ({}, {})".format(self.sigma_final, self.sigma_initial))


        self.transforms = T.Compose(
            [
                T.RandomResize(base_size, base_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.ColorJitter(contrast=(self.contrast_final,self.contrast_initial), brightness=(self.brightness_final, self.brightness_initial)),
                #T.GaussianBlur(kernel_size=19, sigma=(self.sigma_final, self.sigma_initial)),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)