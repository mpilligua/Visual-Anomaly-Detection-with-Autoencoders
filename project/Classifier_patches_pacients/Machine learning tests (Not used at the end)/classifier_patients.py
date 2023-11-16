import pickle
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def process_data(data, patients_id, labels_ids, lable2num):
    """
    data: dictionary of each patient with the labels of each image patch

    return: X, y
    """
    new_data = {}
    for patient, labels in data.items():
        new_data[patient] = Counter(labels)
        total = sum(new_data[patient].values())
        for i in range(2):
            if i not in new_data[patient]:
                new_data[patient][i] = 0
            new_data[patient][i] /= total
        
        new_data[patient]["value"] = list(new_data[patient].values())[1]
    X = []
    y = []
    for patient, label in zip(patients_id, labels_ids):
        if patient in new_data.keys():
            X.append(new_data[patient]["value"])
            y.append(lable2num[label])

    return np.array(X).reshape(-1, 1), np.array(y)


if __name__ == '__main__':
    path_train_data  = "/fhome/gia07/project/runs_clf/efficientnet/Ground_truth_patient_classification/dict_train_cropped_positive_negative.pkl"
    path_test_data  = "/fhome/gia07/project/runs_clf/efficientnet/Ground_truth_patient_classification/dict_test_cropped_positive_negative.pkl"
    path_train_labels = "/fhome/gia07/project/Train_test_splits/train_data.pkl"
    path_test_labels = "/fhome/gia07/project/Train_test_splits/test_data.pkl"

    lable2num = {"NEGATIVA":0, "BAIXA":1, "ALTA":1}

    with open(path_train_data, 'rb') as file:
        dict_train = pickle.load(file)
    with open(path_test_data, 'rb') as file:
        dict_test = pickle.load(file)
    # Use the model predictions
    train_data = dict_train["prediction"]
    test_data = dict_test["prediction"]


    with open(path_train_labels, 'rb') as file:
        patients_id_train, train_labels = pickle.load(file)
    with open(path_test_labels, 'rb') as file:
        patients_id_test, test_labels = pickle.load(file)
        
    X_train, y_train = process_data(train_data, patients_id_train, train_labels, 
                                    lable2num=lable2num)
    X_test, y_test = process_data(test_data, patients_id_test, test_labels, 
                                    lable2num=lable2num)


    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_test[0], y_test[0])

    # print("KNN")
    # clf = KNeighborsClassifier(n_neighbors=4)
    # clf = clf.fit(X_train, y_train)

    # print("Results train")
    # y_pred = clf.predict(X_train)
    # print(classification_report(y_train, y_pred, target_names=lable2num.keys()))

    # print("Results test")
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=lable2num.keys()))

    # print("GradientBoostingClassifier")
    # clf = GradientBoostingClassifier(random_state=42, n_estimators=15, max_depth=3)
    # clf = clf.fit(X_train, y_train)

    # print("Results train")
    # y_pred = clf.predict(X_train)
    # print(classification_report(y_train, y_pred, target_names=lable2num.keys()))

    # print("Results test")
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=lable2num.keys()))

    # print("SVM")
    # from sklearn.model_selection import GridSearchCV 
    
    # # defining parameter range 
    # param_grid = {'C': [0, 0.1, 0.5, 1, 2, 5, 10, 100, 1000], 
    #             'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #             'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    #             'class_weight': ['balanced', None]}  
    
    # grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0, scoring='f1_macro') 
    
    # # fitting the model for grid search 
    # grid.fit(X_train, y_train) 

    # # print best parameter after tuning
    # print(grid.best_params_)

    # # Get the best model
    # clf = grid.best_estimator_

    # print("Results train")
    # y_pred = clf.predict(X_train)
    # print(classification_report(y_train, y_pred, target_names=lable2num.keys())) #["NEGATIVA", "POSITIVA"]))

    # print("Results test")
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=lable2num.keys())) #["NEGATIVA", "POSITIVA"]))

    print("DecisionTreeClassifier")
    clf = tree.DecisionTreeClassifier(max_depth=1, random_state=42)
    clf = clf.fit(X_train, y_train)

    print("Results train")
    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred, target_names=list(lable2num.keys())[:2]))

    print("Results test")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(lable2num.keys())[:2]))


    # Plot of the confusion matrix on the test set
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap="Blues", fmt="d")
    plt.title('Confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    ax.xaxis.set_ticklabels(list(lable2num.keys())[:2])
    ax.yaxis.set_ticklabels(list(lable2num.keys())[:2])
    plt.savefig("confusion_matrix.png")
