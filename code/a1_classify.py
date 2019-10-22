from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import csv
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    numerator = 0
    for i in range(len(C)):
        numerator += C[i][i]
    denominator = 0.0
    for i in C:
        for j in i:
            denominator += j
    return numerator/denominator

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    results = []
    for k in range(len(C)):
        numerator = C[k][k]
        denominator = 0.0
        for j in range(len(C)):
            denominator += C[k][j]
        results.append(numerator / denominator)
    return results

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    results = []
    for k in range(len(C)):
        numerator = C[k][k]
        denominator = 0.0
        for i in range(len(C)):
            denominator += C[i][k]
        results.append(numerator / denominator)
    return results

def train_classifiers(filename):
    '''Make a 80/20 split, train 5 classifiers, then compute
    Accuracy, Recall, Precision for each of them

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    result = np.load(filename)
    data = result[result.files[0]] if result.files else []
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Create and fit
    linear_svc = SVC(kernel='linear', max_iter=1000)
    linear_svc.fit(X_train, y_train)
    svc = SVC(kernel='rbf', gamma=2.0, max_iter=1000)
    svc.fit(X_train, y_train)
    forest = RandomForestClassifier(n_estimators=10, max_depth=5)
    forest.fit(X, y)
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(X, y)
    ada = AdaBoostClassifier()
    ada.fit(X, y)

    # get predictors
    linear_label = linear_svc.predict(X_test)
    svc_label = svc.predict(X_test)
    forest_label = forest.predict(X_test)
    mlp_label = mlp.predict(X_test)
    ada_label = ada.predict(X_test)

    #generate confusion matrices
    linear_confusion = confusion_matrix(y_test, linear_label)
    svc_confusion = confusion_matrix(y_test, svc_label)
    forest_confusion = confusion_matrix(y_test, forest_label)
    mlp_confusion = confusion_matrix(y_test, mlp_label)
    ada_confusion = confusion_matrix(y_test, ada_label)

    confusion_array = [linear_confusion, svc_confusion, forest_confusion, mlp_confusion, ada_confusion]

    with open('a1_3.1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(confusion_array)):
            acc = accuracy(confusion_array[i])
            recall_array = recall(confusion_array[i])
            precision_array = precision(confusion_array[i])
            matrix_array = []
            for j in confusion_array[i]:
                for k in j:
                    matrix_array.append(k)
            row = [i+1] + [acc] + recall_array + precision_array + matrix_array
            writer.writerow(row)
    csvfile.close()

    accuracy_array = list(map(lambda a: accuracy(a), confusion_array))
    max_index = accuracy_array.index(max(accuracy_array))

    return (X_train, X_test, y_train, y_test,max_index+1)


def vary_data_ammount(X_train, X_test, y_train, y_test,iBest):
    '''Train the best model with increasing amount of data

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    X_1k, y_1k = [], []
    subsamples = [1000, 5000, 10000, 15000, 20000]
    acc_array = []
    if iBest == 1:
        classifier = SVC(kernel='linear', max_iter=1000)
    elif iBest == 2:
        classifier = SVC(kernel='rbf', gamma=2.0, max_iter=1000)
    elif iBest == 3:
        classifier = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif iBest == 4:
        classifier = MLPClassifier(alpha=0.05)
    else:
        classifier = AdaBoostClassifier()

    for subsample in subsamples:
        pickArray = np.random.choice(X_train.shape[0], subsample, replace=False)
        X_curr = X_train[pickArray]
        y_curr = y_train[pickArray]
        if subsample == 1000:
            X_1k = X_curr
            y_1k = y_curr
        classifier.fit(X_curr, y_curr)
        predictor = classifier.predict(X_test)
        confusion = confusion_matrix(y_test, predictor)
        acc = accuracy(confusion)
        acc_array.append(acc)

    with open('a1_3.2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(acc_array)
    csvfile.close()

    return (X_1k, y_1k)

def feature_analysis(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' Select the K best features from the corresponding model

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    k_list = [5, 10, 20, 30, 40, 50]
    best_k_1 = []
    best_k_32 = []
    X_new_1k = []
    X_new_32k = []

    acc_list = []

    if i == 1:
        classifier = SVC(kernel='linear', max_iter=1000)
    elif i == 2:
        classifier = SVC(kernel='rbf', gamma=2.0, max_iter=1000)
    elif i == 3:
        classifier = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif i == 4:
        classifier = MLPClassifier(alpha=0.05)
    else:
        classifier = AdaBoostClassifier()

    for k in k_list:
        curr = [k]
        selector = SelectKBest(f_classif, k=k)
        Xk_1k = selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_.argsort()[:k]
        curr.extend(pp)
        best_k_1.append(curr)
        if k == 5:
            classifier.fit(Xk_1k, y_1k)
            predictor = classifier.predict(selector.transform(X_test))
            confusion = confusion_matrix(y_test, predictor)
            acc_list.append(accuracy(confusion))

    for k in k_list:
        curr = [k]
        selector = SelectKBest(f_classif, k=k)
        Xk_32k = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_.argsort()[:k]
        curr.extend(pp)
        best_k_32.append(curr)
        if k == 5:
            classifier.fit(Xk_32k, y_train)
            predictor = classifier.predict(selector.transform(X_test))
            confusion = confusion_matrix(y_test, predictor)
            acc_list.append(accuracy(confusion))

    with open('a1_3.3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in best_k_32:
            writer.writerow(i)
        writer.writerow(acc_list)
    csvfile.close()

def cross_val( filename, i ):
    ''' Perform 5-Fold cross validation

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
    '''
    result = np.load(filename)
    data = result[result.files[0]] if result.files else []
    X = data[:, :-1]
    y = data[:, -1]

    print("linear")
    linear_svc = SVC(kernel='linear', max_iter=1000)
    print("svc")
    svc = SVC(kernel='rbf', gamma=2.0, max_iter=1000)
    print("forest")
    forest = RandomForestClassifier(n_estimators=10, max_depth=5)
    print("mlp")
    mlp = MLPClassifier(alpha=0.05)
    print("ada")
    ada = AdaBoostClassifier()

    classifier_list = [linear_svc, svc, forest, mlp, ada]
    accuracy_matrix = []

    folds = KFold(5, shuffle=True)
    for classifier in classifier_list:
        accuracy_list = []
        for train_index, test_index in folds.split(X):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            classifier.fit(X_train_fold, y_train_fold)
            predictor = classifier.predict(X_test_fold)
            confusion = (y_test_fold, predictor)
            accuracy_list.append(accuracy(confusion))
        accuracy_matrix.append(accuracy_list)

    p_values = []
    p_values.extend(accuracy_matrix)
    del p_values[i-1]
    s_list = []

    for a in p_values:
        s_list.append(stats.ttest_rel(accuracy_matrix[i-1], a))

    with open('a1_3.4.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in accuracy_matrix:
            writer.writerow(i)
        writer.writerow(s_list)

    csvfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    print("31")
    X_train, X_test, y_train, y_test, iBest = train_classifiers(args.input)
    print("32")
    X_1k, y_1k = vary_data_ammount(X_train, X_test, y_train, y_test, iBest)
    print("33")
    feature_analysis(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    print("34")
    cross_val(args.input, iBest)

