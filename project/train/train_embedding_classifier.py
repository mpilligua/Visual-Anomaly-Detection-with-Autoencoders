import torch
import torch.nn as nn
import argparse
from torchvision.transforms import v2 as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm

from dataset import create_annotated_loader
from utils import LoadConfig, load_model, get_optimer, get_weights, createDir



def get_loader(train = True):
    transform_train = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                        T.ToTensor(),
                        T.ElasticTransform(),
                        T.RandomEqualize(),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])
    transform_val = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                           ])

    train_loader, val_loader, test_loader = create_annotated_loader(transform_train, config["datasets"]["train"]['batch_size'], config["datasets"]["val"]['batch_size'], config["datasets"]["test"]['batch_size'])
    return train_loader, val_loader

def train(encoder, model, loader, optimizer, criterion, epoch):
    loss = 0
    model.train()
    encoder.eval()

    for i, (img, targets, _) in tqdm(enumerate(loader), total=len(loader)):
        img = img.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        _, embs  = encoder(img, return_embedding=True)
        
        embs = embs.view(embs.size(0), -1)  # flatten the vector embs
        
        targets_hot_v = torch.zeros(embs.size(0), 3).to(device)
        
        for j, v in enumerate(targets): 
            if v == 0: 
                targets_hot_v[j][1] = 1
            elif v == 1:
                targets_hot_v[j][2] = 1
            else:
                targets_hot_v[j][0] = 1

        outputs = model(embs)
        
        try:
            print(targets_hot_v[0])
            print(outputs.shape, targets_hot_v.shape)
        except:
            print("Error")
            print(outputs, targets_hot_v)
            exit()
        train_loss = criterion(outputs, targets_hot_v)
        train_loss.backward()
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        wandb.log({"Train Loss": train_loss.item()}, step = len(loader)*len(img)*(epoch)+i*len(img))

    # compute the epoch training loss
    return loss / len(loader)

def validation(encoder, model, loader, criterion, epoch):
    model.eval()
    encoder.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (img, targets, _) in tqdm(enumerate(loader), total=len(loader)):
            img = img.to(device)
            targets = targets.to(device)
            _, embs = encoder(img, return_embedding=True)
            embs = embs.view(embs.size(0), -1)  # flatten the vector embs
        
            targets_hot_v = torch.zeros(embs.size(0), 3).to(device)
            
            for j, v in enumerate(targets): 
                if v == 0: 
                    targets_hot_v[j][1] = 1
                elif v == 1:
                    targets_hot_v[j][2] = 1
                else:
                    targets_hot_v[j][0] = 1

            outputs = model(embs)
            
            try:
                print(targets_hot_v[0])
                print(outputs.shape, targets_hot_v.shape)
            except:
                print("Error")
                print(outputs, targets_hot_v)
                exit()
            val_loss = criterion(outputs, targets_hot_v)
            loss += val_loss.item()
            correct += (outputs.argmax(1) == targets_hot_v.argmax(1)).sum().item()
            total += len(targets_hot_v)  
        
    return loss / len(loader), correct/total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run2')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    with wandb.init(project='CLF_embedding', config=config, name=args.test_name) as run:
        train_loader, val_loader = get_loader(train = True)
        encoder = load_model(config['model_name']).to(device)

        if config["network"]["checkpoint"] != None: 
            encoder.load_state_dict(torch.load(config["network"]["checkpoint"]))
            print("Load model from checkpoint {}".format(config["network"]["checkpoint"]))
        else: 
            path = get_weights(config["weights_dir"])
            encoder.load_state_dict(torch.load(path))

        model = load_model(config['classifier_name']).to(device)

        config["weights_dir"] = config["weights_dir"] + "CLF/"
        config["output_dir"] = config["output_dir"] + "CLF/"
        createDir(config["weights_dir"])
        createDir(config["output_dir"])


        wandb.watch(model)
        optimizer = get_optimer(config['optimizer_name'], model, lr = config['lr'])
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        
        for epoch in range(config['epochs']):
            print("Epoch {}/{}".format(epoch, config['epochs']))
            print("Training")
            train_loss = train(encoder, model, train_loader, optimizer, criterion, epoch=epoch)
            print("Validation")
            val_loss, val_acc = validation(encoder, model, val_loader, criterion, epoch = epoch)
            print("epoch : {}| Train loss = {:.6f}| Val loss = {:.6f}".format(epoch, train_loss, val_loss))
            wandb.log({"Epoch Train Loss": train_loss, "Epoch Validation Loss": val_loss, "accuracy validation":val_acc, "epoch":epoch+1}, step=(epoch+1)*len(train_loader)*config["datasets"]["train"]['batch_size'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best validation loss")
                torch.save(model.state_dict(), f'{config["weights_dir"]}/CLS_emb_{epoch}.pth')
    
