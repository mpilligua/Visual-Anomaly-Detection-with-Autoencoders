import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from dataset import create_dataloader_predicted
from utils import LoadConfig, load_model, createDir, get_weights

"""
This script is used to visualize some images of the dataset of each class model
before and after passing through the autoencoder.
"""

if __name__ == '__main__':
    # load the inline arguments and the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run6')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set the weights path and create the output directory
    weights_path = config["weights_dir"]
    out_dir = config["root_dir"] + "/visualization/"
    createDir(out_dir)

    batch_size = config["datasets"]["test"]["batch_size"]

    # create the model and load the weights
    model = load_model(config['model_name']).to(device)
    path = get_weights(weights_path)
    model.load_state_dict(torch.load(path))

    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])

    path = "/fhome/gia07/project/Train_test_splits/test_data.pkl"

    # create the data loader
    test_loader = create_dataloader_predicted(path, transform = None, batch_size = 1, shuffle=False)
    
    ImgxClass = {}
    PredxClass = {}    
    # iterate over the data loader and save the images in a dictionary
    for i, (img, pred, label, img_path) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img = img.to(device)
        pred = pred.to(device)
        label = label.to(device)
        
        if label.item() not in ImgxClass:
            ImgxClass[label.item()] = []
        
        if label.item() not in PredxClass:
            PredxClass[label.item()] = []

        ImgxClass[label.item()].append(img)
        PredxClass[label.item()].append(pred)
        

    # create a 5x5 grid with the images of each class
    for key in ImgxClass:
        fig, ax = plt.subplots(5, 5, figsize=(10, 10))
        ax = ax.flatten()

        for i in range(25):
            ax[i].axis('off')
            try: 
                ax[i].imshow(ImgxClass[key][i].cpu().detach().numpy().squeeze(0)[:, :, ::-1])
            except IndexError:
                continue

        fig.suptitle("Class {} original".format(key), fontsize=22)
        plt.savefig(out_dir + "class_{}_original.png".format(key))

        fig, ax = plt.subplots(5, 5, figsize=(10, 10))

        ax = ax.flatten()

        for i in range(25):
            ax[i].axis('off')
            try: 
                ax[i].imshow(PredxClass[key][i].cpu().detach().numpy().squeeze(0)[:, :, ::-1])
            except IndexError:
                continue

        fig.suptitle("Class {} predicted".format(key), fontsize=22)
        plt.savefig(out_dir + "class_{}_predicted.png".format(key))

    


    