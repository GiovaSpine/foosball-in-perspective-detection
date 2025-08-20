import os
import torch
from torchvision import models
from image_processing import process_image_for_ViT
from config import *


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# loading of the pretrained model
vit_model = models.vit_b_16(pretrained=True)
vit_model.eval()  # validation mode
vit_model.heads = torch.nn.Identity()  # removes the classification head
vit_model.to(device)


# feature extraction
def extract_features(images_path: str, features_path: str) -> None:
    '''
    Extract the features from images using a pretrained ViT model, and save those features
    in a specifed path as pt files

    Parameters:
    images_path (str): The path of the images from which the features will be extracted
    features_path (str): The path where the features will be saved

    Returns:
    None
    '''
    if not os.path.exists(images_path) or not os.path.exists(features_path):
        error_message = ""
        if not os.path.exists(images_path):
            error_message += f"The following path doesn't exists: {images_path}\n"
        if not os.path.exists(features_path):
            error_message += f"The following path doesn't exists: {features_path}\n"    
        raise ValueError(error_message)
    
    for img_name in os.listdir(images_path):
        if not any(img_name.lower().endswith(ext) for ext in IMAGES_EXTENSIONS):
            continue  # ignore other types of files

        img_path = os.path.join(images_path, img_name)
        
        img_tensor = process_image_for_ViT(img_path).to(device)  # move tensor to GPU
        
        with torch.no_grad():
            features = vit_model(img_tensor)  # feature extraction

        feature_path = os.path.join(features_path, os.path.splitext(img_name)[0] + ".pt")
        torch.save(features.cpu(), feature_path)  # save to CPU
        print(f"Features saved: {feature_path}")

    print(f"Done extracting features for {images_path}")


extract_features(IMAGES_DIRECTORY, FEATURES_DIRECTORY)