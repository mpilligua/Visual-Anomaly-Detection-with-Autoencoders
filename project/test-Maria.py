import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import os

from dataset import create_dataloader, create_test_dataloader
from utils import LoadConfig, load_model, get_optimer



def get_loader(train = True):
    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader = create_dataloader(config['input_dir'], config['datasets']['neg_samples_dir'], transform, config['batch_size'], shuffle=True)
    return train_loader, val_loader

def validation(model, loader, criterion):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_model(config['model_name']).to(device)

    path2load = None
    weights_path = config["weights_dir"]
    out_dir = config["root_dir"] + "/test_images/"

    batch_size = config["datasets"]["test"]["batch_size"]

    path_folder = config["datasets"]["test"]["path_folder"]
    path_pickle = config["datasets"]["test"]["path_pickle"]

    for path in os.listdir(weights_path):
        if path[-4:] == ".pth":
            if path2load == None: 
                path2load = path
            elif int(path.split("epoch_")[1].split(".")[0]) > int(path2load.split("epoch_")[1].split(".")[0]):        
                path2load = path
                
    model.load_state_dict(torch.load(weights_path + path2load))  

    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])

    test_loader = create_test_dataloader(path_folder, path_pickle, transform, batch_size, shuffle=True)

    for i, (img, class_, name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img = img.to(device)
        outputs = model(img)

        #save the images in the out_dir
        for j in range(len(outputs)):
            n = name[j]
            if not n.endswith(".png"):
                n = n + ".png"
            cl = class_[j]

            if not os.path.exists(out_dir + str("/".join(n.split("/")[:-1]))):
                os.makedirs(out_dir + str("/".join(n.split("/")[:-1])),exist_ok=True)

            save_image(outputs[j], out_dir + n)
        
