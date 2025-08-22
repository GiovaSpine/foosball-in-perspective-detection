import os
from functools import wraps
from torch import Tensor
from torchvision import transforms
from PIL import Image, ImageOps




def validate_image_path(func):
    """
    Decorator to validate that the first argument `img_path` exists
    before executing the wrapped function.
    """
    @wraps(func)
    def wrapper(img_path, *args, **kwargs):
        if not os.path.exists(img_path):
            raise ValueError(f"The following provided path doesn't exist: {img_path}")
        return func(img_path, *args, **kwargs)
    return wrapper



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


@validate_image_path
def process_image_for_ViT(image_path: str) -> Tensor:
    '''
    Given the path of an image, it applies the padding, scales the padded image
    to the input dimension of the model ViT and converts the image to a tensor.

    Parameters:
    img_path (string): The path of the image

    Returns:
    Tensor: The converted padded image to a tensor
    '''
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


@validate_image_path
def process_image_for_YOLO(image_path: str) -> Tensor:
    '''
    Given the path of an image, it applies the padding, scales the padded image
    to the input dimension of the model YOLO and converts the image to a tensor.

    Parameter:
    img_path (string): The path of the image

    Returns:
    Tensor: The converted padded image to a tensor
    '''
    img = Image.open(image_path).convert("RGB")
    padded_img = _pad_image(img)

    SIZE = 640
    resized_img = padded_img.resize((SIZE, SIZE))
    
    # transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform(resized_img)