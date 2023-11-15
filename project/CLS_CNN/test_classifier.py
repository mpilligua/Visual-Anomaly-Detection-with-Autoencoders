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

from train_classifier import get_splits

def test(model, loader):
    """
    Test a given model.

    Args:
        model (nn.Module): The model to test.
        loader (DataLoader): The data loader to use for testing.

    Returns:
        dict: A dictionary containing the metrics for the test set.
                TP: True positives  
                FP: False positives
                TN: True negatives
                FN: False negatives
                test_loss: The average test loss.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    
    # Compute the precision and recall for each class
    True_positives = [0, 0, 0]
    False_positives = [0, 0, 0]
    True_negatives = [0, 0, 0]
    False_negatives = [0, 0, 0]

    with torch.no_grad():
        for i, (img, targets, _) in tqdm(enumerate(loader), total=len(loader)):
            img = img.to(device)
            targets = targets.to(device)
            outputs = model(img)

            targets_one_hot = torch.zeros(targets.size(0), config['classes']).to(device)
            targets_one_hot[:, 0] = (targets == 0).float()
            targets_one_hot[:, 1] = (targets == 2).float()

            test_loss += criterion(outputs, targets_one_hot).item()
                                                                                                   
            for i in range(2):
                True_positives[i] += ((outputs.argmax(1) == i) * (targets_one_hot.argmax(1) == i)).sum().item()
                False_positives[i] += ((outputs.argmax(1) == i) * (targets_one_hot.argmax(1) != i)).sum().item()
                True_negatives[i] += ((outputs.argmax(1) != i) * (targets_one_hot.argmax(1) != i)).sum().item()
                False_negatives[i] += ((outputs.argmax(1) != i) * (targets_one_hot.argmax(1) == i)).sum().item()

    dict_metrics = {}
    for i in range(2):
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
    # Parse command line arguments and load the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1_test')
    args = parser.parse_args()
    config = LoadConfig_clf(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(config['model_name'], classes=config["classes"]).to(device)

    weights_dir = config['weights_dir']
    splits = 1
    for i, (train_loader, val_loader, test_loader) in enumerate(get_splits(splits)):
        # config['output_dir'] = out_dir + f'_{i+1}_{splits}/'
        config['weights_dir'] = weights_dir + f'_{i+1}_{splits}/'
        # createDir(config['output_dir'])
        

        print("Load model from weights {}".format(config["weights_dir"]))
        path = get_weights(config["weights_dir"])
        model.load_state_dict(torch.load(path))

        metrics = test(model, train_loader)

        print(metrics)

        # Write a json file with the metrics
        with open(config["root_dir"] + "metrics_train.json", 'w') as f:
            json.dump(metrics, f)

