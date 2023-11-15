import yaml
import os
from models.autencoder import *
from CNN_approach_files.models.classifier import *
import torch


def load_model(model_name, classes=None):
    """
    Load the model with the given name.

    Args:
        model_name (str): The name of the model to load.
        classes (int, optional): The number of classes in the dataset.
                                 Used only for the EFFICIENTNET models. Defaults to None.

    Returns:
        nn.Module: The model.
    """

    if model_name == 'CNN_AUTOENCODER':
        return ConvAE()
    elif model_name == 'UNET':
        return UNet()
    elif model_name == 'UNET_PRECONSTRUCTED':
        return Unet_preconstructed()
    elif model_name == 'UNET_NOT_RESIDUALS':
        return UNet_NotRes()
    elif model_name == 'RESNET50':
        return ResNet50()
    elif model_name == 'RESNET50_1LAYER':
        return ResNet50_1layer()
    elif model_name == 'CLS_emb':
        return MLP_emb(1048576, 100, 3)
    elif model_name == 'UNet_CNN':
        return unet_cnn()
    elif model_name == 'VGG19':
        return VGG19()
    elif model_name == 'VGG13':
        return VGG13()
    elif model_name == 'CONVNEXT':
        return CONVNEXT_BASE()
    elif model_name == 'EFFICIENTNET':
        return EFFICIENTNET(out=classes)
    else:
        raise Exception("Model not found")

def get_optimer(optimizer_name, model, lr):
    """
    Get the optimizer with the given name.

    Args:
        optimizer_name (str): The name of the optimizer to load.
        model (nn.Module): The model to optimize.
        lr (float): The learning rate.

    Returns:
        torch.optim: The optimizer.
    """

    if optimizer_name == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'ADAGRAD':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not found")

def createDir(dir_name):
    """
    Create a directory if it does not exist.

    Args:
        dir_name (str): The name of the directory to create.
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def LoadConfig(test_name):
    """
    Load the configuration file with the given name.

    Args:
        test_name (str): The name of the configuration file.

    Returns:
        dict: The configuration file.
    """

    with open("/fhome/gia07/project/setups/" + test_name + ".yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt[test_name] = test_name
    opt["output_dir"] = "/fhome/gia07/project/runs/" + test_name + "/images/"
    opt["weights_dir"] = "/fhome/gia07/project/runs/" + test_name + "/weights/"
    opt["root_dir"] = "/fhome/gia07/project/runs/" + test_name + "/"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])

    opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt

def LoadConfig_clf(test_name):
    """
    Load the configuration file with the given name.

    Args:
        test_name (str): The name of the configuration file.

    Returns:
        dict: The configuration file.
    """

    with open("/fhome/gia07/project/setups_clf/" + test_name + ".yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt[test_name] = test_name
    opt["output_dir"] = "/fhome/gia07/project/runs_clf/" + test_name + "/images/"
    opt["weights_dir"] = "/fhome/gia07/project/runs_clf/" + test_name + "/weights/"
    opt["root_dir"] = "/fhome/gia07/project/runs_clf/" + test_name + "/"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])

    opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt

def get_weights(weights_path):
    """
    Get the weights with the highest epoch number.

    Args:
        weights_path (str): The path to the weights.

    Returns:
        str: The path to the weights with the highest epoch number.
    """
    
    path2load = None
    for path in os.listdir(weights_path):
        if path[-4:] == ".pth":
            if path2load == None: 
                path2load = path
            elif int(path.split("epoch_")[1].split(".")[0]) > int(path2load.split("epoch_")[1].split(".")[0]):        
                path2load = path
    if path2load == None: 
        print("No weights found")
        return None

    return weights_path + path2load