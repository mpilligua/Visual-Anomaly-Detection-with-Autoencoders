import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import os

from dataset import create_CroppedPatches_loader, DatasetCroppedPatchesTest
from utils import LoadConfig, load_model, createDir, get_weights

"""
This script is used to pass the all images of the cropped patches through the autoencoder
and save the output images.
"""

if __name__ == '__main__':
    # load the inline arguments and the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run4')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the data loader for the test set
    weights_path = config["weights_dir"]
    out_dir = config["root_dir"] + "/CroppedPatches/"
    createDir(out_dir)

    batch_size = config["datasets"]["test"]["batch_size"]

    # load the model and the weights
    model = load_model(config['model_name']).to(device)
    path = get_weights(weights_path)
    model.load_state_dict(torch.load(path))

    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])

    # create the data loader
    test_loader = create_CroppedPatches_loader(transform, batch_size, shuffle=False)
    
    # iterate over the data loader and save the images in the out_dir
    for i, (img, name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img = img.to(device)
        outputs = model(img)
        outputs = outputs.cpu()

        for j in range(len(outputs)):
            n = name[j]
            if not n.endswith(".png"):
                n = n + ".png"

            if not os.path.exists(out_dir + str("/".join(n.split("/")[:-1]))):
                os.makedirs(out_dir + str("/".join(n.split("/")[:-1])),exist_ok=True)

            save_image(outputs[j], out_dir + n)
        
