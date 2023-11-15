import torch
import torch.nn as nn
import argparse
from torchvision.transforms import v2 as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm

from dataset import *
from utils import *

from sklearn.metrics import balanced_accuracy_score


def get_splits(splits):
    """
    Get the training, validation and test data loaders for a given number of splits.

    Args:
        splits (int): The number of splits to use.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test data loaders.
    """
    transform_train = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                        T.ElasticTransform(),
                        T.RandomEqualize(),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])
    
    transform_val = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                           ])

    train_dict_path = "/fhome/gia07/project/Train_test_splits/train_data.pkl"
    test_dict_path = "/fhome/gia07/project/Train_test_splits/test_data.pkl"  
    paths = [train_dict_path, test_dict_path]
    transforms = [transform_train, transform_val]                     
    for x in create_dataloader_predicted_CNN(paths, transforms, batch_size=config["datasets"]["train"]["batch_size"], run=args.test_name, shuffle=True, splits=splits, classes = config['classes']):
        yield x

def train(model, loader, optimizer, criterion, epoch):
    """
    Train a given model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The data loader to use for training.
        optimizer (Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss function to use for training.
        epoch (int): The current epoch number.

    Returns:
        float: The average training loss for the epoch.
    """

    loss = 0
    model.train()

    for i, (img, targets, _) in tqdm(enumerate(loader), total=len(loader)):
        img = img.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        targets_one_hot = torch.zeros(targets.size(0), config['classes']).to(device)
        targets_one_hot[:, 0] = (targets == 0).float()
        targets_one_hot[:, 1] = (targets == 2).float()
        train_loss = criterion(outputs, targets_one_hot)
        train_loss.backward()
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        wandb.log({"Train Loss": train_loss.item()}, step = len(loader)*len(img)*(epoch)+i*len(img))

    # compute the epoch training loss
    return loss / len(loader)

def validation(model, loader, criterion):
    """
    Compute the validation loss, accuracy and balanced accuracy for a given model and data loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader to use for evaluation.
        criterion (nn.Module): The loss function to use for evaluation.

    Returns:
        Tuple[float, float, float]: The validation loss, accuracy and balanced accuracy.
    """

    model.eval()
    loss = 0
    correct = 0
    total = 0
    balanced_acc = 0
    with torch.no_grad():
        for i, (img, targets, _) in tqdm(enumerate(loader), total=len(loader)):
            img = img.to(device)
            targets = targets.to(device)
            outputs = model(img)

            targets_one_hot = torch.zeros(targets.size(0), config['classes']).to(device)
            targets_one_hot[:, 0] = (targets == 0).float()
            targets_one_hot[:, 1] = (targets == 2).float()

            val_loss = criterion(outputs, targets_one_hot)
            loss += val_loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            targets = targets.cpu().numpy()
            outputs = outputs.argmax(1).cpu().numpy()
            balanced_acc += balanced_accuracy_score(targets, outputs)
            total += len(targets)  
        
    return loss / len(loader), correct/total, balanced_acc/len(loader)

def get_label_percentage(dataset):
    """"
    Compute the percentage of each label in a given dataset.

    Args:
        dataset (Dataset): The dataset to use.

    Returns:
        List[float]: The percentage of each label in the dataset.
    """
    label_counts = {}
    for _, label, _ in dataset:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    total_count = sum(label_counts.values())
    label_percentage = [count / total_count for label, count in label_counts.items()]
    return label_percentage

if __name__ == '__main__':
    # Parse command line arguments and load the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run2')
    args = parser.parse_args()
    config = LoadConfig_clf(args.test_name)

    # Set the device and get the root weights directory
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_dir = config['weights_dir']

    splits = 1

    # Iterate over the k-fold splits
    for i, (train_loader, val_loader, test_loader) in enumerate(get_splits(splits)):
        
        # create the split weights directory
        config['weights_dir'] = weights_dir + f'_{i+1}_{splits}/'
        createDir(config['weights_dir'])
        
        # create the wandb run
        with wandb.init(project=f'CLF_finetunned', config=config, name=args.test_name + f"_{i+1}_{splits}") as run:
            # get the model, if a checkpoint is specified load the weights
            model = load_model(config['model_name'], classes=config["classes"]).to(device)
            if config["network"]["checkpoint"] != None: 
                model.load_state_dict(torch.load(config["network"]["checkpoint"]))
                print("Load model from checkpoint {}".format(config["network"]["checkpoint"]))

            # get the loss function and optimizer
            class_weights = 1 / torch.Tensor(get_label_percentage(train_loader.dataset))
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
            optimizer = get_optimer(config['optimizer_name'], model, lr = config['lr'])
           
            # train the model
            best_val_loss = float('inf')
            for epoch in range(config['epochs']):
                print("Epoch {}/{}".format(epoch, config['epochs']))
                print("Training")
                train_loss = train(model, train_loader, optimizer, criterion, epoch=epoch)
                print("Validation")
                val_loss, val_acc, val_bal_acc = validation(model, val_loader, criterion, epoch = epoch)
                print("epoch : {}| Train loss = {:.6f}| Val loss = {:.6f}".format(epoch, train_loss, val_loss))
                wandb.log({"Epoch Train Loss": train_loss, "Epoch Validation Loss": val_loss, "accuracy validation":val_acc, "epoch":epoch+1, "balanced accuracy validation":val_bal_acc}, step=(epoch+1)*len(train_loader)*config["datasets"]["train"]['batch_size'])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("New best validation loss")
                    torch.save(model.state_dict(), f'{config["weights_dir"]}/CNN_autoencoder_epoch_{epoch}.pth')
    
