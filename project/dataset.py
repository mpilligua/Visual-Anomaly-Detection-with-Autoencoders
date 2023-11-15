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
        Store the path of the image and the label in a list.
        -1 -> no helicobacter
        0 -> don't know
        1 -> helicobacter and inflammation
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
    
class DatasetCroppedPatches_get400_imgs(Dataset):
    def __init__(self, transform=None):
        """
        Read all images from the folder CroppedPatches.
        Store the path of the image and the label in a list.
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
                        
                        # iterate over the csv to get the label
                        for line in open(self.csv_annotations, 'r').readlines()[1:]:
                            line = line[:-1].split(",")
                            # print(line[0].split(".")[0])
                            # print(line[0].split(".")[1])
                            # print(folder)
                            # print(img[:-4])
                            # break
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

class DatasetFinetuneing(Dataset):
    def __init__(self, transform=None, partition="train"):
        self.root = "/fhome/mapsiv/QuironHelico/AnnotatedPatches"
        self.csv_annotations = "/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv"
        save_partition_path = "/fhome/gia07/project/Classifier/Partitions"
        self.transform = transform

        name_pacients = os.listdir(self.root)
        import random
        import pickle
        random.seed(10)
        random.shuffle(name_pacients)
        if partition == "train":
            self.name_pacients = name_pacients[:int(0.7*len(name_pacients))]
        elif partition == "val":
            self.name_pacients = name_pacients[int(0.7*len(name_pacients)):int(0.8*len(name_pacients))]
        elif partition == "test":
            self.name_pacients = name_pacients[int(0.8*len(name_pacients)):]
        else:
            raise ValueError("Partition must be train, val or test")
        
        with open(os.path.join(save_partition_path, partition + ".pickle"), 'wb') as file:
            pickle.dump(self.name_pacients, file)

        self.y = []
        self.img_paths = []
        with open(self.csv_annotations, 'r') as file:
            for line in file.readlines()[1:]:
                line = line[:-1].split(",")
                folder = line[0].split(".")[0]
                patch = line[0].split(".")[1]
                
                if (folder in self.name_pacients) and (os.path.exists(os.path.join(self.root, folder, patch + ".png"))):
                    self.y.append(int(line[1])+1)
                    self.img_paths.append(os.path.join(folder, patch + ".png"))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(self.root + "/" + img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img, self.y[idx], img_path




class DatasetPred2(Dataset):
    def __init__(self, transform=None, name_patients=None, y_patients=None, run="run4", classes = 3):
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
    
class DatasetPredCropped2(Dataset):
    def __init__(self, transform=None, name_patients=None, y_patients=None, run="run4"):                
        self.root = "/fhome/mapsiv/QuironHelico/CroppedPatches"
        self.root_pred = f"/fhome/gia07/project/runs/run5/CroppedPatches"
        self.name_pacients = name_patients
        self.transform = transform

        self.img_paths = []
        for folder in os.listdir(self.root_pred):
            if folder.split("_")[0] in self.name_pacients:
                imgs = os.listdir(os.path.join(self.root, folder))
                for img in imgs:
                    if img.endswith(".png"):
                        self.img_paths.append(os.path.join(folder, img))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(self.root + "/" + img_path).convert('RGB')
        # pred = cv2.imread(os.path.join(self.root_pred, img_path))

        if self.transform:
            img = self.transform(img)
        
        return img, img_path


def create_dataloader_predicted(path_dict, transform, batch_size, annotated=True, shuffle=True, run="run4"):
    with open(path_dict, 'rb') as file:
        pacient_ids, labels = pickle.load(file)
    if annotated:
        dataset = DatasetPred(transform, pacient_ids, labels, run=run)
    else:
        dataset = DatasetPredCropped(transform, pacient_ids, labels, run=run)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
    

def remove_class(X, y, class_to_remove):
    idx_to_remove = []
    for idx, label in enumerate(y):
        if str(label) == str(class_to_remove):
            idx_to_remove.append(idx)
    X = np.delete(X, idx_to_remove, axis=0)
    y = np.delete(y, idx_to_remove, axis=0)
    return X, y

def create_dataloader_predicted_CNN(paths, transforms, batch_size, shuffle=True, run="run4", splits = 5, classes = 3):
    [transform_train, transform_val] = transforms
    [path_dict_train, path_dict_val] = paths


    with open(path_dict_val, 'rb') as file:
        pacient_ids, labels = pickle.load(file)

    test_dataset = DatasetPred2(transform_val, pacient_ids, labels, run=run, classes = classes)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


    with open(path_dict_train, 'rb') as file:
        pacient_ids, labels = pickle.load(file)

    if splits == 1:
        from sklearn.model_selection import train_test_split
        pacient_ids_train, pacient_ids_val, labels_train, labels_val = train_test_split(pacient_ids, labels, test_size=0.2, random_state=42)
        pacient_ids_split = {"train": pacient_ids_train, "val": pacient_ids_val}
        labels_split = {"train": labels_train, "val": labels_val}

        train_dataset = DatasetPred2(transform_train, pacient_ids_split["train"], labels_split["train"], run=run, classes = classes)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        val_dataset = DatasetPred2(transform_val, pacient_ids_split["val"], labels_split["val"], run=run, classes = classes)
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

            train_dataset = DatasetPred2(transform_train, pacient_ids_split["train"], labels_split["train"], run=run)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

            val_dataset = DatasetPred2(transform_val, pacient_ids_split["val"], labels_split["val"], run=run)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

            yield train_dataloader, val_dataloader, test_dataloader
    


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
    dataset = DatasetCroppedPatches_get400_imgs(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    import torchvision.transforms as T
    transf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ]) 
    data = DatasetFinetuneing(transform=transf ,partition="train")
    print(len(data))
    data = DatasetFinetuneing(transform=transf ,partition="val")
    print(len(data))
    data = DatasetFinetuneing(transform=transf ,partition="test")
    print(len(data))
    

