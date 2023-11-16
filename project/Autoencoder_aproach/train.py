import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm

from dataset import create_dataloader
from utils import LoadConfig, load_model, get_optimer



def get_loader(train = True):
    """
    Create dataloader for training and validation with augmentation
    """
    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader = create_dataloader(config['input_dir'], config['datasets']['neg_samples_dir'], transform, config["datasets"]["train"]['batch_size'], shuffle=True)
    return train_loader, val_loader

def train(model, loader, optimizer, criterion, epoch):
    loss = 0
    model.train()

    for i, (img, name) in tqdm(enumerate(loader), total=len(loader)):
        # load it to the active device
        img = img.to(device)
        outputs = model(img)

        train_loss = criterion(outputs, img)
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
        for img, name in tqdm(loader):
            img = img.to(device)
            outputs = model(img)

            val_loss = criterion(outputs, img)
            loss += val_loss.item()
    
        # save the last batch output of every epoch
        for j in range(outputs.size(0)):
            save_image(outputs[j], f'{config["output_dir"]}/{epoch}_{j}.png')
                    
        return loss / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run3')
    args = parser.parse_args()

    # Load config file
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Initialize wandb
    with wandb.init(project='test', config=config, name=args.test_name) as run:

        # Get dataloader
        train_loader, val_loader = get_loader(train = True)
        # Load model specified in config file
        model = load_model(config['model_name']).to(device)

        # Load checkpoint if specified
        if config["network"]["checkpoint"] != None: 
            model.load_state_dict(torch.load(config["network"]["checkpoint"]))
            print("Load model from checkpoint {}".format(config["network"]["checkpoint"]))

        wandb.watch(model)
        # Get optimizer and loss function
        optimizer = get_optimer(config['optimizer_name'], model, lr = config['lr'])
        criterion = nn.MSELoss()

        # Start training
        best_val_loss = float('inf')
        for epoch in range(config['epochs']):
            print("Epoch {}/{}".format(epoch, config['epochs']))
            # Train and validate for one epoch
            print("Training")
            train_loss = train(model, train_loader, optimizer, criterion, epoch=epoch)
            print("Validation")
            val_loss = validation(model, val_loader, criterion, epoch = epoch)
            print("epoch : {}| Train loss = {:.6f}| Val loss = {:.6f}".format(epoch, train_loss, val_loss))
            wandb.log({"Epoch Train Loss": train_loss, "Epoch Validation Loss": val_loss, "epoch":epoch+1}, step=(epoch+1)*len(train_loader)*config["datasets"]["train"]['batch_size'])
            # Save model if validation loss is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best validation loss")
                torch.save(model.state_dict(), f'{config["weights_dir"]}/CNN_autoencoder_epoch_{epoch}.pth')
    
