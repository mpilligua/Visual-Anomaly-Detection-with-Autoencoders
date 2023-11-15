from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

import os
from joblib import dump, load
import pickle
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from metrics import MSE_Red_channel as vectordifference
from metrics import MSE_Red_channel_cropped as vectordifference_Cropped
from dataset import DatasetPred, create_dataloader_predicted

def classify_all_patches_patients_and_save(clf, dataloader, treshold_pos_neg, tresh=155):
    # list_names = []
    # data = (dataloader.dataset)
    # for i in tqdm(range(len(data))):
    #     list_names.append(data[i][2])
    
    X, list_names = vectordifference_Cropped(dataloader, treshold=tresh)
    y_pred = np.where(X > treshold_pos_neg, 1, 0)
        
    dict_patient_patches = {"truth": {}, "prediction": {}}
    for i, element in enumerate(list_names):
        patient = element.split("/")[0].split("_")[0]
        # print(patient)
        if patient not in dict_patient_patches["prediction"].keys():
            #dict_patient_patches["truth"][patient] = [y_true[i]]
            dict_patient_patches["prediction"][patient] = [y_pred[i]]
        else:
            #dict_patient_patches["truth"][patient].append(y_true[i])
            dict_patient_patches["prediction"][patient].append(y_pred[i])        

    return dict_patient_patches


if __name__ == '__main__':
    name_model = "tree_positive_other"
    run = "run5"
    train_loader_annotated = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/train_data.pkl", None, 1, shuffle=True, run=run)
    test_loader_annotated = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/test_data.pkl", None, 1, shuffle=True, run=run)
    
    X_train, y_train3 = vectordifference(train_loader_annotated, treshold=156)
    X_train, y_train = X_train[y_train3 != 1], y_train3[y_train3 != 1]
    X_test, y_test3 = vectordifference(test_loader_annotated, treshold=156)
    X_test, y_test = X_test[y_test3 != 1], y_test3[y_test3 != 1]

    # np.where(condition, x, y) -> if condition is true, return x, else return y
    y_train = np.where(y_train <= 1, 0, 1)
    y_test = np.where(y_test <= 1, 0, 1)


    # K-folds cross validation
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=10)
    dict_splits_results = {}

    best_thresholds = []
    for i, (train_index, test_index) in enumerate(kf.split(X_train, y=y_train)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train2, X_test2 = X_train[train_index], X_train[test_index]
        y_train2, y_test2 = y_train[train_index], y_train[test_index]


        fpr, tpr, thresholds = metrics.roc_curve(y_train2, X_train2, pos_label=1)
        # plt.plot(fpr, tpr)
        # plt.xlabel("False positive rate")
        # plt.ylabel("True positive rate")
        # plt.savefig("roc_curve_pos_neg_patches.png")
        # plt.close()

        # Get the treshold that has higher true positive rate and lower false positive rate
        best_treshold = 0
        best_distance = 100
        for i in range(len(fpr)):
            distance = np.sqrt(fpr[i]**2 + (1 - tpr[i])**2)
            if distance < best_distance:
                best_distance = distance
                best_treshold = thresholds[i]
            
        best_thresholds.append(best_treshold)

        y_pred2 = np.where(X_test2 > best_treshold, 1, 0)

        dict_splits_results[f"split_{i}_val"] = classification_report(y_test2, y_pred2, target_names=["Negative", "Positive"], output_dict=True)

    # Make the mean of the results on the splits
    print("Results k fold val")
    precision_neg = []
    precision_pos = []
    recall_neg = []
    recall_pos = []
    f1_neg = []
    f1_pos = []
    for key in dict_splits_results.keys():
        precision_neg.append(dict_splits_results[key]["Negative"]["precision"])
        precision_pos.append(dict_splits_results[key]["Positive"]["precision"])
        recall_neg.append(dict_splits_results[key]["Negative"]["recall"])
        recall_pos.append(dict_splits_results[key]["Positive"]["recall"])
        f1_neg.append(dict_splits_results[key]["Negative"]["f1-score"])
        f1_pos.append(dict_splits_results[key]["Positive"]["f1-score"])

    plt.figure(figsize=(4, 5))
    plt.boxplot(best_thresholds, 
                showmeans=True, 
                meanline=True, 
                labels=["Treshold positive/negative"])
    plt.ylabel("Treshold value")
    plt.savefig("boxplot_tresholds.png")
    plt.close()


    print("Mean results k folds val:")
    print("Negative")
    print(f"Precision: {np.median(precision_neg):.3} +- {np.std(precision_neg):.3}")
    print(f"Recall: {np.median(recall_neg):.3} +- {np.std(recall_neg):.3}")
    print(f"F1: {np.median(f1_neg):.3} +- {np.std(f1_neg):.3}")
    print("")
    print("Positive")
    print(f"Precision: {np.median(precision_pos):.3} +- {np.std(precision_pos):.3}")
    print(f"Recall: {np.median(recall_pos):.3} +- {np.std(recall_pos):.3}")
    print(f"F1: {np.median(f1_pos):.3} +- {np.std(f1_pos):.3}")

    print("")
    print(f"Mean treshold: {np.mean(best_thresholds)} +- {np.std(best_thresholds)}")
    
    best_treshold = np.median(best_thresholds)
    print("")
    print(f"Best treshold: {best_treshold}")
    print("")

    plt.figure(figsize=(10, 5))
    plt.boxplot((precision_neg, recall_neg, f1_neg, precision_pos, recall_pos, f1_pos), 
                showmeans=True, 
                meanline=True, 
                meanprops={'color': 'blue'},
                medianprops={'color': 'orange'},
                labels=["Precision Negative", "Recall Negative", "F1 Negative", "Precision Positive", "Recall Positive", "F1 Positive"])
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.savefig("boxplot_metrics.png")
    plt.close()

    print("Results training data")
    y_pred = np.where(X_train > best_treshold, 1, 0)
    print(classification_report(y_train, y_pred, target_names=["Negative", "Positive"]))

    y_pred = np.where(X_test > best_treshold, 1, 0)

    print("Results test data")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=["Negative", "Positive"], columns=["Negative", "Positive"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_patches.png")
    plt.close()

    fpr, tpr, thresholds = metrics.roc_curve(y_train2, X_train2, pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig("roc_curve_patches_train.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(X_train[y_train == 0], bins=100, label="Negative", alpha=0.5)
    plt.hist(X_train[y_train == 1], bins=100, label="Positive", alpha=0.5)
    plt.xlim(0, 20)
    plt.xlabel("Percentage of positive patches")
    plt.ylabel("Number of patients")
    plt.legend()
    plt.savefig("histogram_pos_neg_patches.png")   
    plt.close()

    best_tres = 156
    train_loader_cropped = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/train_data.pkl", None, 1, run=run, annotated=False, shuffle=True)
    test_loader_cropped = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/test_data.pkl", None, 1, run=run, annotated=False, shuffle=True)
    
    clf = load(f'/fhome/gia07/project/runs/{run}/Classifier_patches/{name_model}.joblib')
    # dict_train = classify_all_patches_patients_and_save(clf, train_loader_cropped, treshold_pos_neg=best_treshold, tresh=best_tres)

    # if not os.path.exists(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification"):
    #     os.makedirs(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification",exist_ok=True)
        
    # with open(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification/dict_train_treshold.pkl", 'wb') as file:
    #     pickle.dump(dict_train, file)

    dict_test = classify_all_patches_patients_and_save(clf, test_loader_cropped, treshold_pos_neg=best_treshold, tresh=best_tres)

    # with open(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification/dict_test_treshold.pkl", 'wb') as file:
    #     pickle.dump(dict_test, file)

