from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

import os
from joblib import dump, load
import pickle
from sklearn.metrics import classification_report
from metrics import MSE_Red_channel as vectordifference
from metrics import MSE_Red_channel_cropped
from dataset import DatasetPred, create_dataloader_predicted


def train_model(train_loader_annotated,  tres=155):
    X, y = vectordifference(train_loader_annotated, treshold=tres)
    from sklearn.model_selection import GridSearchCV 
    
    # defining parameter range 
    # param_grid = {'max_depth': [1], 
    #             'class_weight': ['balanced', None]}  
    
    # grid = GridSearchCV(tree.DecisionTreeClassifier(random_state=42), param_grid, refit = True, verbose = 0, scoring='f1_macro') 
    
    # # fitting the model for grid search 
    # grid.fit(X, y) 

    # # print best parameter after tuning
    # print(grid.best_params_)

    # # Get the best model
    # clf = grid.best_estimator_
    clf = tree.DecisionTreeClassifier(max_depth=1, random_state=42, class_weight='balanced').fit(X, y)
    return clf, y, clf.predict(X)

def test_model(test_loader_annotated, clf, tres=155):
    X, y_true = vectordifference(test_loader_annotated, treshold=tres)
    y_pred = clf.predict(X)
    target_names = ['negative_uncertainty', 'positive']
    print(classification_report(y_true, y_pred, target_names=target_names))
    return y_pred, y_true

def classify_all_patches_patients_and_save(clf, dataloader, tresh=155):
    # list_names = []
    # data = (dataloader.dataset)
    # for i in tqdm(range(len(data))):
    #     list_names.append(data[i][2])
    
    X, list_names = MSE_Red_channel_cropped(dataloader, treshold=tresh)
    y_pred = clf.predict(X)
        
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
    # train_loader_annotated = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/train_data.pkl", None, 1, shuffle=True, run=run)
    # test_loader_annotated = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/test_data.pkl", None, 1, shuffle=True, run=run)
    # hist_f1 = []
    # hist_tresh = []
    # best_f1 = 0
    # for i in range(150, 160, 1):
    #     print("Treshold:", i)
    #     clf, y_true, y_pred = train_model(train_loader_annotated, tres=i)
    #     #Compute macro F1 score
    #     f1 = classification_report(y_true, y_pred, target_names=['negative_uncertainty', 'positive'], output_dict=True)["macro avg"]["f1-score"]
    #     hist_f1.append(f1)
    #     hist_tresh.append(i)
    
    #     print("F1:", f1)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_tres = i
    #         best_clf = clf
    #         print("New best F1")

    #     if not os.path.exists(f"/fhome/gia07/project/runs/{run}/Classifier_patches/"):
    #         os.makedirs(f"/fhome/gia07/project/runs/{run}/Classifier_patches/",exist_ok=True)

    # print("Best treshold:", best_tres)
    # print("Best F1:", best_f1)
    # print("Results test")
    # y_pred, y_true = test_model(test_loader_annotated, best_clf, tres=best_tres)
    # dump(best_clf, f'/fhome/gia07/project/runs/{run}/Classifier_patches/{name_model}.joblib')
    # print("Model saved")

    # tree.plot_tree(best_clf)
    # import matplotlib.pyplot as plt
    # plt.savefig(f"/fhome/gia07/project/{run}_{name_model}_{best_tres}.png", dpi=1000)
    # # Clean the plot
    # plt.close()


    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, cmap="Blues", fmt="d")
    # plt.title('Confusion matrix')
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # ax.xaxis.set_ticklabels(['negative_uncertainty', 'positive'])
    # ax.yaxis.set_ticklabels(['negative_uncertainty', 'positive'])
    # plt.savefig("confusion_matrix_patches.png")
    # plt.close()

    # plt.plot(hist_tresh, hist_f1)
    # plt.xlabel("Treshold")
    # plt.ylabel("F1")
    # plt.savefig("F1_patches.png")

    best_tres = 156
    train_loader_cropped = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/train_data.pkl", None, 1, run=run, annotated=False, shuffle=True)
    test_loader_cropped = create_dataloader_predicted("/fhome/gia07/project/Train_test_splits/test_data.pkl", None, 1, run=run, annotated=False, shuffle=True)
    
    clf = load(f'/fhome/gia07/project/runs/{run}/Classifier_patches/{name_model}.joblib')
    dict_train = classify_all_patches_patients_and_save(clf, train_loader_cropped, tresh=best_tres)

    if not os.path.exists(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification"):
        os.makedirs(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification",exist_ok=True)
        
    with open(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification/dict_train_cropped_positive_negative.pkl", 'wb') as file:
        pickle.dump(dict_train, file)

    dict_test = classify_all_patches_patients_and_save(clf, test_loader_cropped)

    with open(f"/fhome/gia07/project/runs/{run}/Ground_truth_patient_classification/dict_test_cropped_positive_negative.pkl", 'wb') as file:
        pickle.dump(dict_test, file)
