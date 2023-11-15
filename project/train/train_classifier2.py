import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm

from dataset import *
from utils import *
from models.autencoder import unet_cnn



def get_loader(train = True):
    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader, test_loader = create_annotated_loader(transform, config["datasets"]["train"]['batch_size'], config["datasets"]["val"]['batch_size'], config["datasets"]["test"]['batch_size'])
    return train_loader, val_loader

def train(model, loader, optimizer, criterion, epoch):
    loss = 0
    model.train()

    for i, (img, targets, name) in tqdm(enumerate(loader), total=len(loader)):
        # load it to the active device
        img = img.to(device)
        outputs = model(img)

        targets = targets.to(device)

        train_loss = criterion(outputs, targets)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        wandb.log({"Train Loss": train_loss.item()}, step = len(loader)*len(img)*(epoch)+i*len(img))

    # compute the epoch training loss
    return loss / len(loader)

def validation(model, loader, criterion, epoch):
    model.eval()
    loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for img, targets, name in tqdm(loader):
            img = img.to(device)
            outputs = model(img)

            targets = targets.to(device)

            val_loss = criterion(outputs, targets)
            loss += val_loss.item()
            correct += (outputs.argmax(1) == targets.argmax(1)).sum().item() 
            total += len(targets) 
    
        return loss / len(loader), correct/total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run3')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)

    config["weights_dir"] = config["weights_dir"] + "/UNET_CNN/"
    createDir(config["weights_dir"])

    config["output_dir"] = config["output_dir"] + "/UNET_CNN/"
    createDir(config["output_dir"])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    with wandb.init(project='UNET_CNN', config=config, name=args.test_name) as run:

        train_loader, val_loader = get_loader(train = True)
        unet = load_model(config['model_name']).to(device)

        if config["network"]["checkpoint"] != None: 
            unet.load_state_dict(torch.load(config["network"]["checkpoint"]))
            print("Load model from checkpoint {}".format(config["network"]["checkpoint"]))
        else: 
            path = get_weights(config["weights_dir"])
            unet.load_state_dict(torch.load(path))

        model = unet_cnn(unet).to(device)
        # print(model.state_dict())

        wandb.watch(model)
        optimizer = get_optimer(config['optimizer_name'], model, lr = config['lr'])
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        for epoch in range(config['epochs']):
            print("Epoch {}/{}".format(epoch, config['epochs']))
            print("Training")
            train_loss = train(model, train_loader, optimizer, criterion, epoch=epoch)
            print("Validation")
            val_loss, val_accuraccy = validation(model, val_loader, criterion, epoch = epoch)
            print("epoch : {}| Train loss = {:.6f}| Val loss = {:.6f}".format(epoch, train_loss, val_loss))
            wandb.log({"Epoch Train Loss": train_loss, "Epoch Validation Loss": val_loss, "epoch":epoch+1, "Accuracy Validation": val_accuraccy}, step=(epoch+1)*len(train_loader)*config["datasets"]["train"]['batch_size'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best validation loss")
                torch.save(model.state_dict(), f'{config["weights_dir"]}/UNet_cnn_epoch_{epoch}.pth')
    
