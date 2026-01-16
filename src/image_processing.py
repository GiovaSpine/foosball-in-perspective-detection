import os
from torch import Tensor
from torchvision import transforms
from PIL import Image, ImageOps


def _pad_image(image: Image) -> Image:
    '''
    Apply a black padding to the provided image.

    Parameters:
    image (Image): The image to apply the padding to

    Returns:
    ImageOps: The padded image
    '''
    # calculate the padding to make the original image a square
    max_side = max(image.size)
    delta_w = max_side - image.width
    delta_h = max_side - image.height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

    # apply the padding
    padded_img = ImageOps.expand(image, padding, fill=(0, 0, 0))

    return padded_img


def process_image_for_ViT(image_path: str) -> Tensor:
    '''
    Given the path of an image, it applies the padding, scales the padded image
    to the input dimension of the model ViT and converts the image to a tensor.

    Parameters:
    img_path (string): The path of the image

    Returns:
    Tensor: The converted padded image to a tensor
    '''

    if not os.path.exists(image_path):
        raise ValueError(f"The following provided path doesn't exist: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    padded_img = _pad_image(img)

    SIZE = 224
    resized_img = padded_img.resize((SIZE, SIZE))
    
    # transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    return transform(resized_img)
