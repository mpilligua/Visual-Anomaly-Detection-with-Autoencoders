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
import argparse


def classify_all_patches_patients_and_save(model, dataloader, out_dir):
    """
    Classify all the patches of the patients and save the predictions in a dictionary.

    Args:
        model (nn.Module): The model to use for classification.
        dataloader (DataLoader): The data loader to use for testing.
        out_dir (str): The directory where to save the predictions.

    Returns:
        dict: A dictionary containing the predictions for each patient.
    """

    # create 2 log files that saves the predictions and the names of the patients
    f_y_pred = open(f"{out_dir}/y_pred2.txt", "w")
    f_list_names = open(f"{out_dir}/list_names2.txt", "w")

    for original, _ , name in tqdm(dataloader):
        f_list_names.write(name[0] + "\n")
        original = original.to(device)
        outputs = model(original)
        
        f_y_pred.write(str(outputs.argmax(1).cpu().numpy()[0]) + "\n")
        del original
        torch.cuda.empty_cache()

    f_y_pred.close()
    f_list_names.close()

    # read the files and create a dictionary with the predictions
    f_y_pred = open(f"{out_dir}/y_pred2.txt", "r")
    f_list_names = open(f"{out_dir}/list_names2.txt", "r")

    y_pred = f_y_pred.readlines()
    list_names = f_list_names.readlines()

    f_y_pred.close()
    f_list_names.close()

    dict_patient_patches = {"truth": {}, "prediction": {}}
    for i, element in enumerate(list_names):
        patient = element.split("/")[0].split("_")[0]
        if patient not in dict_patient_patches["prediction"].keys():
            dict_patient_patches["prediction"][patient] = [int(y_pred[i])]
        else:
            dict_patient_patches["prediction"][patient].append(int(y_pred[i]))        

    return dict_patient_patches


if __name__ == '__main__':
    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run2')
    args = parser.parse_args()
    config = LoadConfig_clf(args.test_name)

    # set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load the model
    clf = load_model(config['model_name'], classes=config["classes"]).to(device)

    # create the data loaders
    transform_val = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                           ])

    train_dict_path = "/fhome/gia07/project/Train_test_splits/train_data.pkl"
    test_dict_path = "/fhome/gia07/project/Train_test_splits/test_data.pkl"
    train_loader = create_dataloader_predicted(train_dict_path, transform_val, 1, run="run5", annotated=False, shuffle=True, pil = True)
    test_loader = create_dataloader_predicted(test_dict_path, transform_val, 1, run="run5", annotated=False, shuffle=True, pil = True)
    
    # create the output directory
    out_dir = f"{config['root_dir']}/Ground_truth_patient_classification"
    weights_dir = config['weights_dir']
    createDir(out_dir)
    
    # load the weights
    splits = i = 1
    config['weights_dir'] = weights_dir + f'_{i}_{splits}/'
    path = get_weights(config["weights_dir"])
    print("Load model from weights {}".format(path))
    clf.load_state_dict(torch.load(path))

    # classify the patches and save the predictions
    dict_train = classify_all_patches_patients_and_save(clf, train_loader, out_dir)
    with open(out_dir + "/dict_train_cropped_positive_negative.pkl", 'wb') as file:
        pickle.dump(dict_train, file)

    dict_test = classify_all_patches_patients_and_save(clf, test_loader, out_dir)
    with open(out_dir + "/dict_test_cropped_positive_negative.pkl", 'wb') as file:
        pickle.dump(dict_test, file)
