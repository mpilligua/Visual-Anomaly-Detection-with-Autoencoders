import torch
import torch.nn as nn
import argparse
from torchvision.transforms import v2 as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import json

from dataset import *
from utils import *



def get_loader():
    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                        #    T.RandomHorizontalFlip(),
                        #    T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader, test_loader = create_annotated_loader(transform, config["datasets"]["train"]['batch_size'], config["datasets"]["val"]['batch_size'], config["datasets"]["test"]['batch_size'])
    return test_loader


def test(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    # Compute the precision and recall for each class
    True_positives = [0, 0, 0]
    False_positives = [0, 0, 0]
    True_negatives = [0, 0, 0]
    False_negatives = [0, 0, 0]

    with torch.no_grad():
        diccionary_preds = {}

        for i, (img, targets, _, pacient) in tqdm(enumerate(loader), total=len(loader)):
            img = img.to(device)
            targets = targets.to(device)
            outputs = model(img)

            test_loss += criterion(outputs, targets).item()
            for o in outputs: 
                if pacient[0] not in diccionary_preds.keys():
                    diccionary_preds[pacient[0]] = [o]
                else:
                    diccionary_preds[pacient[0]].append(o)

            # Compute the true positives, false positives, true negatives and false negatives
            for i in range(3):
                True_positives[i] += torch.sum((outputs.argmax(dim=1) == i) & (targets.argmax(1) == i)).item()
                False_positives[i] += torch.sum((outputs.argmax(dim=1) == i) & (targets.argmax(1) != i)).item()
                True_negatives[i] += torch.sum((outputs.argmax(dim=1) != i) & (targets.argmax(1) != i)).item()
                False_negatives[i] += torch.sum((outputs.argmax(dim=1) != i) & (targets.argmax(1) == i)).item()

    dict_metrics = {}
    for i in range(3):
        if True_positives[i] + False_positives[i] == 0:
            precision = 0
        else:
            precision = True_positives[i] / (True_positives[i] + False_positives[i])
        if True_positives[i] + False_negatives[i] == 0:
            recall = 0
        else:
            recall = True_positives[i] / (True_positives[i] + False_negatives[i])
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        dict_metrics["precision_class_" + str(i)] = precision
        dict_metrics["recall_class_" + str(i)] = recall
        dict_metrics["f1_class_" + str(i)] = f1

    dict_metrics["TP"] = True_positives
    dict_metrics["FP"] = False_positives
    dict_metrics["TN"] = True_negatives
    dict_metrics["FN"] = False_negatives
    dict_metrics["test_loss"] = test_loss / len(loader)

    return dict_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1_test')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = get_loader()
    unet = load_model(config['model_name']).to(device)
    model = unet_cnn(pretrained = unet).to(device)

    if config["network"]["checkpoint"] != None: 
        model.load_state_dict(torch.load(config["network"]["checkpoint"]))
        print("Load model from checkpoint {}".format(config["network"]["checkpoint"]))
    else: 
        path = get_weights(config["weights_dir"] + "/UNET_CNN/")
        model.load_state_dict(torch.load(path))


    metrics = test(model, test_loader)

    print(metrics)

    # Write a json file with the metrics
    with open(config["root_dir"] + "metrics_val.json", 'w') as f:
        json.dump(metrics, f)

