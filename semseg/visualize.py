import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

def main(args):
    image = np.load(args.image_path)
    prediction = torch.from_numpy(np.load(args.prediction_path)[0]).argmax(0)
    target = torch.from_numpy(np.load(args.target_path)[0])

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # semantic segmentation original
    original_image = get_original_image(image)

    # semantic segmentation target
    target_mask = get_target_mask(target)

    # semantic segmentation prediction
    prediction_mask = get_prediction_mask(prediction, colors)

    # semantic segmentation blended image
    blend_image = get_blend_mask(original_image, prediction_mask)

    return original_image, target_mask, prediction_mask, blend_image

def get_original_image(image):
    original_image = np.transpose(image*255, (1,2,0))
    original_image = original_image.astype(np.uint8)
    original_image = Image.fromarray(original_image, 'RGB')
    return original_image

def get_target_mask(target, colors):
    target_mask = Image.fromarray(target.byte().cpu().numpy())
    target_mask.putpalette(colors)
    return

def get_prediction_mask(prediction, colors):
    prediction_mask = Image.fromarray(prediction.byte().cpu().numpy())
    prediction_mask.putpalette(colors)
    return

def get_blend_mask(background, overlay):
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    blend_image = Image.blend(background, overlay, 0.5)
    return blend_image

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Segmentation Masks/Results", add_help=add_help)

    parser.add_argument("--image-path", default="", type=str, help="image path")
    parser.add_argument("--target-path", default="", type=str, help="target path")
    parser.add_argument("--prediction-path", default="", type=str, help="prediction path")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)