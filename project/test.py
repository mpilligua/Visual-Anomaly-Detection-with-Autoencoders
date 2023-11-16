import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import os

from dataset import create_dataloader, create_test_dataloader
from utils import LoadConfig, load_model, get_weights


"""
This script is used to pass the test images of the cropped patches of negative patients through the autoencoder
and save the output images.
"""

def get_loader(train = True):
    """
    Create the data loader for the training and validation sets.

    Args:
        train (bool): Whether to create the data loader for the training or validation set.

    Returns:
        DataLoader: The data loader for the training or validation set.
    """

    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader = create_dataloader(config['input_dir'], config['datasets']['neg_samples_dir'], transform, config['batch_size'], shuffle=True)
    return train_loader, val_loader


def validation(model, loader, criterion):
    """
    Perform the validation step.

    Args:
        model (nn.Module): The model to use for validation.
        loader (DataLoader): The data loader to use for validation.
        criterion (nn.Module): The loss function to use for validation.

    Returns:
        float: The validation loss.
    """

    model.eval()
    loss = 0
    with torch.no_grad():
        for img in tqdm(loader):
            img = img.to(device)
            outputs = model(img)

            val_loss = criterion(outputs, img)
            loss += val_loss.item()
        return loss / len(loader), outputs

if __name__ == '__main__':
    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set the weights path and create the output directory
    weights_path = config["weights_dir"]
    out_dir = config["root_dir"] + "/test_images/"

    # load parameters from config
    batch_size = config["datasets"]["test"]["batch_size"]
    path_folder = config["datasets"]["test"]["path_folder"]
    path_pickle = config["datasets"]["test"]["path_pickle"]

    # create the model and load the weights
    model = load_model(config['model_name']).to(device)
    path = get_weights(weights_path)
    model.load_state_dict(torch.load(path))  

    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])

    # create the data loader
    test_loader = create_test_dataloader(path_folder, path_pickle, transform, batch_size, shuffle=True)

    # iterate over the data loader and save the images in the out_dir
    for i, (img, class_, name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img = img.to(device)
        outputs = model(img)

        for j in range(len(outputs)):
            n = name[j]
            if not n.endswith(".png"):
                n = n + ".png"
            cl = class_[j]

            if not os.path.exists(out_dir + str("/".join(n.split("/")[:-1]))):
                os.makedirs(out_dir + str("/".join(n.split("/")[:-1])),exist_ok=True)

            save_image(outputs[j], out_dir + n)
        
