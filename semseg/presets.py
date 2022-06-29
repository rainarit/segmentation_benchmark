import torch
import transforms as T

class SegmentationPresetTrain:
    def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
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
    def __init__(self, *, base_size, 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225), 
                 jitter=False,
                 contrast_min=1.0,
                 contrast_max=1.0,
                 brightness_min=1.0,
                 brightness_max=1.0,
                 occlude_min=0.0, 
                 occlude_max=0.0):

        self.transforms = T.Compose(
            [
                T.RandomResize(base_size, base_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        if jitter:
            transformations = [
                 T.RandomResize(base_size, base_size),
                 T.PILToTensor(),
                 T.ConvertImageDtype(torch.float),
                 T.ColorJitter(contrast=(contrast_min, contrast_max), brightness=(brightness_min, brightness_max))
            ]
            if occlude_max > 0.0:
                transformations.append(
                    T.Occlude(occlude_min, occlude_max, 0.)
                )
            transformations.append(T.Normalize(mean=mean, std=std))
            self.transforms = T.Compose(transformations)   


    def __call__(self, img, target):
        return self.transforms(img, target)

# class SegmentationPresetTrain:
#     def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         min_size = int(0.5 * base_size)
#         max_size = int(2.0 * base_size)

#         trans = [T.RandomResize(min_size, max_size)]
#         if hflip_prob > 0:
#             trans.append(T.RandomHorizontalFlip(hflip_prob))
#         trans.extend(
#             [
#                 T.RandomCrop(crop_size),
#                 T.PILToTensor(),
#                 T.ConvertImageDtype(torch.float),
#                 T.Normalize(mean=mean, std=std),
#             ]
#         )
#         self.transforms = T.Compose(trans)

#     def __call__(self, img, target):
#         return self.transforms(img, target)


# class SegmentationPresetEval:
#     def __init__(self, base_size, crop_size=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), contrast=1, brightness=1, sigma=1, occlude_low=0, occlude_high=0, jitter=False, blur=False, occlude=False):
#         self.contrast_initial = contrast
#         self.contrast_final = contrast
#         if (contrast == 1):
#             self.contrast_final = contrast
#         else:
#             self.contrast_final = self.contrast_initial-1

#         self.brightness_initial = brightness
#         self.brightness_final = brightness
#         if (brightness == 1):
#             self.brightness_final = brightness
#         else:
#             self.brightness_final = self.brightness_initial-1

#         self.sigma_initial = sigma
#         self.sigma_final = sigma
#         if (sigma == 1):
#             self.sigma_final = sigma
#         else:
#             self.sigma_final = self.sigma_initial-1

#         self.occlusion_initial = occlude_low
#         self.occlusion_final = occlude_high
#         if crop_size:
#             transformations = [T.Resize(base_size), T.CenterCrop(crop_size), T.ToTensor()]
#         else:
#             transformations = [T.Resize(base_size), T.ToTensor()]
#         #transformations = [T.RandomResize(base_size, base_size), T.CenterCrop(crop_size), T.ToTensor()]
#         if jitter:
#             print('Using ColorJitter')
#             print("Contrast: ({}, {})".format(self.contrast_final, self.contrast_initial))
#             print("Brightness: ({}, {})".format(self.brightness_final, self.brightness_initial))
#             transformations.append(T.ColorJitter(contrast=(self.contrast_final,self.contrast_initial), brightness=(self.brightness_final, self.brightness_initial)))
#         if blur:
#             print('Using Gaussian Blur')
#             print("Sigma: ({}, {})".format(self.sigma_final, self.sigma_initial))
#             transformations.append(T.GaussianBlur(kernel_size=19, sigma=(self.sigma_final, self.sigma_initial)))
#         if occlude:
#             print('Using Occlusion')
#             print("Occlusion: ({}, {})".format(self.occlusion_initial, self.occlusion_final))
#             transformations.append(T.Occlude(self.occlusion_initial, self.occlusion_final, 0.),)
#         transformations.append(T.Normalize(mean=mean, std=std))
#         self.transforms = T.Compose(transformations)

#     def __call__(self, img, target):
#         return self.transforms(img, target)