import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import os

from utils import LoadConfig, load_model, createDir, get_weights
from PIL import Image

if __name__ == '__main__':
    # load the inline arguments and the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1')
    parser.add_argument('--img_path', type=str, default='/fhome/mapsiv/QuironHelico/CroppedPatches/B22-26_1/00001.png')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the data loader for the test set
    weights_path = config["weights_dir"]
    out_dir = config["root_dir"] + "test_1_image/"
    createDir(out_dir)

    # load parameters from config
    batch_size = config["datasets"]["test"]["batch_size"]
    path_folder = config["datasets"]["test"]["path_folder"]
    path_pickle = config["datasets"]["test"]["path_pickle"]

    # load the model and the weights
    model = load_model(config['model_name']).to(device)
    path = get_weights(weights_path)
    model.load_state_dict(torch.load(path))
    
    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])


    # open the image, pass it through the model and save the output
    img_path = args.img_path
    img = Image.open(img_path).convert('RGB')
    img = transform(img)

    img = img.to(device)
    outputs = model(img)
    
    n = img_path.split("/")[-1]
    save_image(outputs, out_dir + n)
        
