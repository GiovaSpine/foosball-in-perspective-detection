import os
from torch import Tensor
from torchvision import transforms
from PIL import Image, ImageOps



def check_img_path(img_path):
    if not os.path.exists(img_path):
        raise ValueError(f"The following provided path doesn't exists: {img_path}")
    

# decorator...
    


def _pad_image(img):
    '''

    '''
    # calculate the padding to make the original image a square
    max_side = max(img.size)
    delta_w = max_side - img.width
    delta_h = max_side - img.height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

    # apply the padding
    padded_img = ImageOps.expand(img, padding, fill=(0, 0, 0))

    return padded_img

        

def process_image_for_ViT(img_path) -> Tensor:
    '''
    Given the path of an image, it applies the padding, scales the padded image to the input dimension of the model ViT
    and converts the image to a tensor.

    Parameters:
    img_path (string): The path of the image

    Returns:
    Tensor: The converted padded image to a tensor
    '''
    img = Image.open(img_path).convert("RGB")
    padded_img = _pad_image(img)

    SIZE = 224
    resized_img = padded_img.resize((SIZE, SIZE))
    
    # transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    return transform(resized_img).unsqueeze(0)  # batch of a single image



def process_image_for_YOLO(img_path) -> Tensor:
    '''
    Given the path of an image, it applies the padding, scales the padded image to the input dimension of the model YOLO
    and converts the image to a tensor.

    Parameter:
    img_path (string): The path of the image

    Returns:
    Tensor: The converted padded image to a tensor
    '''
    img = Image.open(img_path).convert("RGB")
    padded_img = _pad_image(img)

    SIZE = 640
    resized_img = padded_img.resize((SIZE, SIZE))
    
    # transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform(resized_img).unsqueeze(0)  # batch of a single image