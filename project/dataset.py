import torch
import pickle
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
from torchvision import transforms as T


class DatasetCroppedPatches(Dataset):
    def __init__(self, img_paths, transform=None):
        """
        Read all images from the folder CroppedPatches.
        
        Args:
            img_paths (list): A list with the paths of the images to use.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Used to train and validate the autoencoder in train.py and test.py
        """

        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, "/".join(img_path.split("/")[-2:])
    
class DatasetCroppedPatchesTest(Dataset):
    def __init__(self, img_paths, name_pacients, y_pacients, transform=None):
        """
        Read all images from the folder CroppedPatches.

        Args:
            img_paths (list): A list with the paths of the images to use.
            name_pacients (list): A list with the names of the patients to use.
            y_pacients (list): A list with the labels of the patients to use.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Used to test the autoencoder in generate_imgs_autoencoder.py
        """

        self.img_paths = img_paths
        self.name_pacients = name_pacients
        self.y = y_pacients
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        pacient = img_path.split("/")[-2]
        for id_pacient, name_pacient in enumerate(self.name_pacients):   
            if pacient.split("_")[0] == name_pacient:
                y = self.y[id_pacient]
        return img, y, "/".join(img_path.split("/")[-2:])

class DatasetAnnotatedPatchesTest(Dataset):
    def __init__(self, transform=None):
        """
        Read all images from the folder AnnotatedPatches.
        
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Used to test the autoencoder only in the annotated patches in test_annotated.py
        """

        self.root = "/fhome/mapsiv/QuironHelico/AnnotatedPatches"
        self.name_pacients = os.listdir(self.root)
        self.csv_annotations = "/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv"
        self.transform = transform

        self.y = []
        self.img_paths = []
        with open(self.csv_annotations, 'r') as file:
            for line in file.readlines()[1:]:
                line = line[:-1].split(",")
                # print(line)
                folder = line[0].split(".")[0]
                patch = line[0].split(".")[1]
                
                if os.path.exists(os.path.join(self.root, folder, patch + ".png")):
                    self.y.append(int(line[1]))
                    self.img_paths.append(os.path.join(folder, patch + ".png"))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(self.root + "/" + img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # set y as a hot vector
        y = np.zeros(3)
        y[self.y[idx]+1] = 1

        # pass y to tensor and to the device
        y = torch.tensor(y, dtype=torch.float32)

        pacient = img_path.split("/")[0]

        return img, y, img_path, pacient
    
class DatasetCroppedPatches_get1000_imgs(Dataset):
    def __init__(self, transform=None):
        """
        Get 1000 random images of each patient
        
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Used in in generate_imgs_autoencoder.py

        """
        self.root = "/fhome/mapsiv/QuironHelico/CroppedPatches"
        self.name_pacients = os.listdir(self.root)
        self.transform = transform

        import random
        random.seed(10)
        self.img_paths = []
        for folder in os.listdir(self.root):
            if not (folder.endswith(".png") or folder.endswith(".csv")):
                imgs = os.listdir(os.path.join(self.root, folder))
                random.shuffle(imgs)
                for img in imgs[:1000]:
                    if img.endswith(".png"):
                        self.img_paths.append(os.path.join(folder, img))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(self.root + "/" + img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, img_path



class DatasetPred(Dataset):
    def __init__(self, transform=None, name_patients=None, y_patients=None, run="run4"):
        """
        Read all images from the folder AnnotatedPatches and the predicted images from the run.

        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
            name_patients (list): A list with the names of the patients to use.
            y_patients (list): A list with the labels of the patients to use.
            run (str): The run name.
        
        Used to train and test the patches classifier classifier_patches.py and classifier_patches_threshold.py
        """

        self.root = "/fhome/mapsiv/QuironHelico/AnnotatedPatches"
        self.csv_annotations = "/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv"
        self.root_pred = f"/fhome/gia07/project/runs/{run}/Annotated"
        self.name_patients = name_patients
        self.transform = transform
        self.y = []

        self.img_paths = []
        for patient in self.name_patients:
            folder = patient + "_0"
            try:
                for img in os.listdir(os.path.join(self.root_pred, folder)):
                    if img[-4:] == ".png":
                        self.img_paths.append(os.path.join(folder, img))
                        
                        for line in open(self.csv_annotations, 'r').readlines()[1:]:
                            line = line[:-1].split(",")
                            if line[0].split(".")[0] == folder and line[0].split(".")[1] == img[:-4]:
                                self.y.append(int(line[1])+1)
                                break
            except FileNotFoundError:
                print("Folder {} does not exist".format(os.path.join(self.root, folder)))
                continue

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(os.path.join(self.root, img_path))
        pred = cv2.imread(os.path.join(self.root_pred, img_path))

        if self.transform:
            img = self.transform(img)
        
        return img, pred, self.y[idx], img_path
    
class DatasetPredCropped(Dataset):
    def __init__(self, transform=None, name_patients=None, y_patients=None, run="run4"):
        """
        Read the 1000 images of each patient from the folder CroppedPatches and the output of the autoencoder.

        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
            name_patients (list): A list with the names of the patients to use.
            y_patients (list): A list with the labels of the patients to use.
            run (str): The run name.
        
        Used to train and test the patients classifier in classifier_patients.py and classifier_patients_threshold.py
        """

        self.root = "/fhome/mapsiv/QuironHelico/CroppedPatches"
        self.root_pred = f"/fhome/gia07/project/runs/{run}/CroppedPatches"
        self.name_pacients = name_patients
        self.transform = transform

        self.img_paths = []
        for folder in os.listdir(self.root_pred):
            if folder.split("_")[0] in self.name_pacients:
                imgs = os.listdir(os.path.join(self.root_pred, folder))
                for img in imgs:
                    if img.endswith(".png"):
                        self.img_paths.append(os.path.join(folder, img))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(os.path.join(self.root, img_path))
        pred = cv2.imread(os.path.join(self.root_pred, img_path))

        if self.transform:
            img = self.transform(img)
        
        return img, pred, img_path



def create_dataloader_predicted(path_dict, transform, batch_size, annotated=True, shuffle=True, run="run4"):
    with open(path_dict, 'rb') as file:
        pacient_ids, labels = pickle.load(file)
    if annotated:
        dataset = DatasetPred(transform, pacient_ids, labels, run=run)
    else:
        dataset = DatasetPredCropped(transform, pacient_ids, labels, run=run)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
    
def create_dataloader(path_folder, path_dict, transform, batch_size, shuffle=True):
    with open(path_dict, 'rb') as file:
        pacient_ids = pickle.load(file)
    
    pacient_ids_train = pacient_ids["train"]
    pacient_ids_val = pacient_ids["val"]

    paths_imgs_train = []
    for folder in pacient_ids_train:
        if os.path.exists(os.path.join(path_folder, folder + "_1")):
            paths_imgs_train.extend([os.path.join(path_folder, folder + "_1", img) for img in os.listdir(os.path.join(path_folder, folder + "_1")) if img.endswith('.png')])
        else:
            print("Folder {} does not exist".format(os.path.join(path_folder, folder + "_1")))
    dataset_train = DatasetCroppedPatches(paths_imgs_train, transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    
    paths_imgs_val = []
    for folder in pacient_ids_val:
        if os.path.exists(os.path.join(path_folder, folder + "_1")):
            paths_imgs_val.extend([os.path.join(path_folder, folder + "_1", img) for img in os.listdir(os.path.join(path_folder, folder + "_1")) if img.endswith('.png')])
        else:
            print("Folder {} does not exist".format(os.path.join(path_folder, folder + "_1")))
    dataset_val = DatasetCroppedPatches(paths_imgs_val, transform)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader_train, dataloader_val

def create_test_dataloader(path_folder, path_pickle, transform, batch_size, shuffle=True):
    with open(path_pickle, 'rb') as file:
        pacient_ids, y_pacients = pickle.load(file)
    
    paths_imgs_test = []
    for folder in pacient_ids:
        if os.path.exists(os.path.join(path_folder, folder + "_1")):
            paths_imgs_test.extend([os.path.join(path_folder, folder + "_1", img) for img in os.listdir(os.path.join(path_folder, folder + "_1")) if img.endswith('.png')])
        else:
            print("Folder {} does not exist".format(os.path.join(path_folder, folder + "_1")))

    dataset_test = DatasetCroppedPatchesTest(paths_imgs_test, pacient_ids, y_pacients, transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=shuffle)

    return dataloader_test

def create_annotated_loader(transform, batch_size_train, batch_size_val, shuffle=True):
    dataset = DatasetAnnotatedPatchesTest(transform)

    train_split = int(0.7*len(dataset))
    val_split = int(0.1*len(dataset))
    test_split = len(dataset) - train_split - val_split

    train, val, test = torch.utils.data.random_split(dataset, [train_split, val_split, test_split], generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size_train, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size_val, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size_val, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader

def create_CroppedPatches_loader(transform, batch_size, shuffle=True):
    dataset = DatasetCroppedPatches_get1000_imgs(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

