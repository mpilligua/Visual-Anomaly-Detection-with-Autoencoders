from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from joblib import dump, load
import pickle
from sklearn.metrics import classification_report
from dataset import *
from utils import *
from torchvision import transforms as T

def test_model(loader, model):
    y_true = []
    for orginial, label, name in loader:
        y_true.append(label.numpy()[0])
        y_pred = model(orginial) # prediccio del model 
    
    target_names = ['negative_uncertainty', 'positive']
    print(classification_report(y_true, y_pred, target_names=target_names))
    return y_pred, y_true

def classify_all_patches_patients_and_save(model, dataloader, out_dir):
    # create 3 log files for each list
    f_y_pred = open(f"{out_dir}/y_pred2.txt", "w")
    # f_y_true = open(f"{out_dir}/y_true.txt", "w")
    f_list_names = open(f"{out_dir}/list_names2.txt", "w")

    for original, targets, name in tqdm(dataloader):
        f_list_names.write(name[0] + "\n")
        original = original.to(device)
        outputs = model(original) # prediccio del model
        
        # targets_one_hot = torch.zeros(targets.size(0), config['classes']).to(device)
        # targets_one_hot[:, 0] = (targets == 0).float()
        # targets_one_hot[:, 1] = (targets == 2).float()

        # f_y_true.write(str(targets_one_hot.argmax(1).cpu().numpy()[0]) + "\n")
        f_y_pred.write(str(outputs.argmax(1).cpu().numpy()[0]) + "\n")
        del original
        torch.cuda.empty_cache()
    
    # close the files
    f_y_pred.close()
    # f_y_true.close()
    f_list_names.close()

    # read the files
    f_y_pred = open(f"{out_dir}/y_pred2.txt", "r")
    # f_y_true = open(f"{out_dir}/y_true.txt", "r")
    f_list_names = open(f"{out_dir}/list_names2.txt", "r")

    y_pred = f_y_pred.readlines()
    # y_true = f_y_true.readlines()
    list_names = f_list_names.readlines()

    # close the files
    f_y_pred.close()
    # f_y_true.close()
    f_list_names.close()

    dict_patient_patches = {"truth": {}, "prediction": {}}
    for i, element in enumerate(list_names):
        patient = element.split("/")[0].split("_")[0]
        if patient not in dict_patient_patches["prediction"].keys():
            dict_patient_patches["prediction"][patient] = [int(y_pred[i])]
        else:
            dict_patient_patches["prediction"][patient].append(int(y_pred[i]))        

    return dict_patient_patches

def get_splits(splits):
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
    # for x in create_dataloader_predicted_CNN(paths, transforms, batch_size=1, run=args.test_name, shuffle=True, splits=splits, classes = config['classes']):
    #     yield x
    return create_CroppedPatches_loader(transform_val, 1, shuffle=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run2')
    args = parser.parse_args()
    config = LoadConfig_clf(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clf = load_model(config['model_name'], classes=config["classes"]).to(device)

    transform_val = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                           ])

    best_tres = 156
    train_dict_path = "/fhome/gia07/project/Train_test_splits/train_data.pkl"
    test_dict_path = "/fhome/gia07/project/Train_test_splits/test_data.pkl"
    train_loader = create_dataloader_predicted(train_dict_path, transform_val, 1, run="run5", annotated=False, shuffle=True, pil = True)
    test_loader = create_dataloader_predicted(test_dict_path, transform_val, 1, run="run5", annotated=False, shuffle=True, pil = True)
    
    out_dir = f"{config['root_dir']}/Ground_truth_patient_classification"
    weights_dir = config['weights_dir']
    print("Output dir:", out_dir)
    createDir(out_dir)
    
    splits = i = 1
    config['weights_dir'] = weights_dir + f'_{i}_{splits}/'
    
    path = get_weights(config["weights_dir"])
    print("Load model from weights {}".format(path))
    clf.load_state_dict(torch.load(path))

    dict_train = classify_all_patches_patients_and_save(clf, train_loader, out_dir)
    with open(out_dir + "/dict_train_cropped_positive_negative.pkl", 'wb') as file:
        pickle.dump(dict_train, file)

    # dict_test = classify_all_patches_patients_and_save(clf, test_loader, out_dir)
    # with open(out_dir + "/dict_test_cropped_positive_negative.pkl", 'wb') as file:
    #     pickle.dump(dict_test, file)
