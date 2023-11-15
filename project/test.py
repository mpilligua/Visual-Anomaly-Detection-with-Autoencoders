import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
import wandb
from tqdm import tqdm

from dataset import create_dataloader
from utils import LoadConfig, load_model, get_optimer


def get_loader():
    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader = create_dataloader(config['input_dir'], config['datasets']['neg_samples_dir'], transform, config['batch_size'], shuffle=True)
    return train_loader, val_loader

from PIL import Image

def get_output_images(model, loader, save_dir):
    with torch.no_grad():
        for i, img in enumerate(tqdm(loader)):
            img = img.to(device)
            outputs = model(img)
            # Save the images to the output folder
            for j in range(outputs.size(0)):
                output_img = outputs[j].permute(1, 2, 0).cpu().numpy()
                output_img = (output_img * 255).astype('uint8')
                output_img = Image.fromarray(output_img)
                output_img.save(f'{save_dir}/output_{i * loader.batch_size + j}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_loader(train = True)
    model = load_model(config['model_name']).to(device)

    if config["network"]["checkpoint"] != None: 
        model.load_state_dict(torch.load(config["network"]["checkpoint"]))
    else:
        path = None
        for path in os.listdir("/fhome/gia07/project/runs/" + args.test_name + "/weights"):
            if path.endswith(".pth"):
                if path == None: 
                    path2load = path
                elif int(path.split("epoch_")[1].split(".")) > int(path2load.split("epoch_")[1].split(".")):        
                    path2load = path
        PATH2LOAD = "/fhome/gia07/project/runs/" + args.test_name + "/weights/" + path2load
        model.load_state_dict(torch.load(PATH2LOAD))  
    get_output_images(model, val_loader,PATH2LOAD)

from PIL import Image

def get_output_images(model, loader, save_dir):
    with torch.no_grad():
        for i, img in enumerate(tqdm(loader)):
            img = img.to(device)
            outputs = model(img)
            # Save the images to the output folder
            for j in range(outputs.size(0)):
                output_img = outputs[j].permute(1, 2, 0).cpu().numpy()
                output_img = (output_img * 255).astype('uint8')
                output_img = Image.fromarray(output_img)
                output_img.save(f'{save_dir}/output_{i * loader.batch_size + j}.png')

