from torchbench.semantic_segmentation import PASCALVOC
from torchbench.semantic_segmentation.transforms import (
    Normalize,
    Resize,
    ToTensor,
    Compose,
)
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as transforms
import PIL

def model_output_function(output, labels):
    return output['out'].argmax(1).flatten(), target.flatten()

def seg_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

model = fcn_resnet101(num_classes=21, pretrained=True)

normalize = Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

PASCALVOC.benchmark(batch_size=32,
    model=model,
    transforms=my_transforms,
    model_output_transform=model_output_function,
    collate_fn=seg_collate_fn,
    paper_model_name='FCN ResNet-101',
    paper_arxiv_id='1605.06211')