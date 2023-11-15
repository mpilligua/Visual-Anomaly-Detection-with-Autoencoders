import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from dataset import create_dataloader, create_test_dataloader, create_dataloader_predicted
from utils import LoadConfig, load_model, get_optimer, createDir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run6')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_model(config['model_name']).to(device)

    path2load = None
    weights_path = config["weights_dir"]
    out_dir = config["root_dir"] + "/visualization/"
    createDir(out_dir)

    batch_size = config["datasets"]["test"]["batch_size"]

    for path in os.listdir(weights_path):
        # print(path, path[-4:])
        if path[-4:] == ".pth":
            if path2load == None: 
                path2load = path
            elif int(path.split("epoch_")[1].split(".")[0]) > int(path2load.split("epoch_")[1].split(".")[0]):        
                path2load = path
    model.load_state_dict(torch.load(weights_path + path2load))  

    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])

    path = "/fhome/gia07/project/Train_test_splits/test_data.pkl"

    test_loader = create_dataloader_predicted(path, transform = None, batch_size = 1, shuffle=False)
    
    # crete a subplot with 5 x 5 images for each class

    ImgxClass = {}
    PredxClass = {}    
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
        
    # print(ImgxClass[ImgxClass.keys()[0]][0].shape)
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

    


    