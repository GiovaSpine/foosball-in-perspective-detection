import os
import argparse
import torch
from torchvision import models
from image_processing import process_image_for_ViT
from config import *



# feature extraction
def extract_features(images_path: str, features_path: str) -> None:
    '''
    Extract the features from images using a pretrained ViT model, and save those features
    in a specifed path as pt files.

    Parameters:
    images_path (str): The path of the images from which the features will be extracted
    features_path (str): The path where the features will be saved

    Returns:
    None
    '''
    if not os.path.exists(images_path) or not os.path.exists(features_path):
        error_message = "\n"
        if not os.path.exists(images_path):
            error_message += f"images_path provided doesn't exists: {images_path}\n"
        if not os.path.exists(features_path):
            error_message += f"features_path provided doesn't exists: {features_path}\n"    
        raise ValueError(error_message)
    

    for image_name in os.listdir(images_path):
        if not any(image_name.lower().endswith(ext) for ext in IMAGES_DATA_EXTENSIONS):
            continue  # ignore other types of files

        image_path = os.path.join(images_path, image_name)
        
        image_tensor = process_image_for_ViT(image_path)  # process the image for ViT
        image_tensor = image_tensor.unsqueeze(0)  # batch of a single image
        image_tensor = image_tensor.to(device)  # move tensor to GPU
        
        with torch.no_grad():
            features = vit_model(image_tensor)  # feature extraction

        feature_path = os.path.join(features_path, os.path.splitext(image_name)[0] + ".pt")
        torch.save(features.cpu(), feature_path)  # save to CPU
        print(f"Feature saved: {feature_path}")

    print(f"Done extracting features for {images_path}")

# =============================================================================

def execute_features_extraction(dataset: str) -> None:
    '''
    Starts the features extraction for the desired dataset.
    
    Parameters:
    dataset: {DEFAULT, ADDED, AUGMENTED}

    Returns:
    None
    '''
    # device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # loading of the pretrained model
    global vit_model
    vit_model = models.vit_b_16(pretrained=True)
    vit_model.eval()  # validation mode
    vit_model.heads = torch.nn.Identity()  # removes the classification head
    vit_model.to(device)
    
    if dataset not in ["DEFAULT", "ADDED", "AUGMENTED"]:
        raise ValueError(f"Error: not valid dataset: {dataset} not in  [DEFAULT, ADDED, AUGMENTED]")
    
    print(f"Extracting features for the dataset {dataset}...")

    if dataset == "DEFAULT":
        extract_features(IMAGES_DATA_DIRECTORY, FEATURES_DIRECTORY)
    elif dataset == "ADDED":
        extract_features(IMAGES_DATA_DIRECTORY, FEATURES_DIRECTORY)
        extract_features(ADDED_IMAGES_DATA_DIRECTORY, ADDED_FEATURES_DIRECTORY)
    else:
        extract_features(AUGMENTED_IMAGES_DATA_DIRECTORY, AUGMENTED_FEATURES_DIRECTORY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts the clustering fot the desired dataset."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["DEFAULT", "ADDED", "AUGMENTED"],
        help="The dataset of images (DEFAULT, ADDED, AUGMENTED)",
    )
    args = parser.parse_args()
    execute_features_extraction(args.dataset)
