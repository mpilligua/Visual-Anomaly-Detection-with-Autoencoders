import torch
import pickle
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
from torchvision import transforms as T



class DatasetPred(Dataset):
    def __init__(self, transform=None, name_patients=None, y_patients=None, run="run4", classes = 3):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
            name_patients (list): A list with the names of the patients to use.
            y_patients (list): A list with the labels of the patients to use.
            run (str): The run name.
            classes (int): The number of classes in the dataset (2 or 3)
        """
        
        self.root = "/fhome/mapsiv/QuironHelico/AnnotatedPatches"
        self.csv_annotations = "/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv"
        self.root_pred = f"/fhome/gia07/project/runs/run5/Annotated"
        self.name_patients = name_patients
        self.transform = transform
        self.y = []

        self.img_paths = []
        for patient in self.name_patients:
            folder = patient + "_0"
            try:
                for img in os.listdir(os.path.join(self.root_pred, folder)):
                    if img[-4:] == ".png":
                        
                        # iterate over the csv to get the label
                        for line in open(self.csv_annotations, 'r').readlines()[1:]:
                            line = line[:-1].split(",")
                            if line[0].split(".")[0] == folder and line[0].split(".")[1] == img[:-4]:
                                cl = int(line[1])+1
                                break
                        if classes != 2 or cl != 1:
                            self.y.append(cl)
                            self.img_paths.append(os.path.join(folder, img))
            except FileNotFoundError:
                # print("Folder {} does not exist".format(os.path.join(self.root, folder)))
                continue

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(self.root + "/" + img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.y[idx], img_path



def remove_class(X, y, class_to_remove):
    """ Remove a class from the dataset

    Args:
        X (np.array): The dataset features.
        y (np.array): The dataset labels.
        class_to_remove (int): The class to remove.

    Returns:
        np.array: The dataset features without the class.
        np.array: The dataset labels without the class.
    """

    idx_to_remove = []
    for idx, label in enumerate(y):
        if str(label) == str(class_to_remove):
            idx_to_remove.append(idx)
    X = np.delete(X, idx_to_remove, axis=0)
    y = np.delete(y, idx_to_remove, axis=0)
    return X, y


def create_dataloader_predicted_CNN(paths, transforms, batch_size, shuffle=True, run="run4", splits = 5, classes = 3):
    """ Create a data loader for the dataset.

    Args:
        paths (list): A list with the paths to the train and test dictionaries.
        transforms (list): A list with the transforms to apply to the images.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data or not.
        run (str): The run name.
        splits (int): The number of splits to perform in the dataset.
        classes (int): The number of classes in the dataset.

    Yields:
        DataLoader: The train, validation and test data loaders for each split.
    """
    
    
    [transform_train, transform_val] = transforms
    [path_dict_train, path_dict_val] = paths


    with open(path_dict_val, 'rb') as file:
        pacient_ids, labels = pickle.load(file)

    test_dataset = DatasetPred(transform_val, pacient_ids, labels, run=run, classes = classes)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


    with open(path_dict_train, 'rb') as file:
        pacient_ids, labels = pickle.load(file)

    if splits == 1:
        from sklearn.model_selection import train_test_split
        pacient_ids_train, pacient_ids_val, labels_train, labels_val = train_test_split(pacient_ids, labels, test_size=0.2, random_state=42)
        pacient_ids_split = {"train": pacient_ids_train, "val": pacient_ids_val}
        labels_split = {"train": labels_train, "val": labels_val}

        train_dataset = DatasetPred(transform_train, pacient_ids_split["train"], labels_split["train"], run=run, classes = classes)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        val_dataset = DatasetPred(transform_val, pacient_ids_split["val"], labels_split["val"], run=run, classes = classes)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        yield train_dataloader, val_dataloader, test_dataloader
    
    else: 
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        pacient_ids = np.array(pacient_ids)
        labels = np.array(labels)

        for train_split, val_split in skf.split(pacient_ids, labels):
            pacient_ids_split = {"train": pacient_ids[train_split], "val": pacient_ids[val_split]}
            labels_split = {"train": labels[train_split], "val": labels[val_split]}

            train_dataset = DatasetPred(transform_train, pacient_ids_split["train"], labels_split["train"], run=run)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

            val_dataset = DatasetPred(transform_val, pacient_ids_split["val"], labels_split["val"], run=run)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

            yield train_dataloader, val_dataloader, test_dataloader
    