import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt


def rf_adaboost(data, labels):
    print("--- Running RF Adaboost")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def naive_bayes(data, labels):
    print("--- Running Naive Bayes")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    model = GaussianNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def knn(data, labels, savepath: str = None):
    print("--- Running kNN")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    #n_neighbours = [k for k in range(1, 250, 4)]  # steps of 4 from 1 to 253
    #n_neighbours = [k for k in range(253, 1000, 16)]  # steps of 16 from 253 to 989
    #n_neighbours = [k for k in range(1039, 2000, 100)]  # steps of 100 from 1039 to 1939
    n_neighbours = [k for k in range(2339, 4000, 400)]  # steps of 400 from 2339 to

    train_means = []
    train_stds = []
    test_means = []
    test_stds = []

    for k in n_neighbours:
        print(f"Using {k} nearest neighbors...")

        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
        print(f"Train accuracy: {train_accuracy}")

        y_pred = model.predict(x_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy}")

        print("-- Mean (std)")
        train_mean = np.mean(train_accuracy)
        train_means.append(train_mean)
        train_std = np.std(train_accuracy)
        train_stds.append(train_std)
        print(f"Train accuracy {train_mean} ({train_std})")

        test_mean = np.mean(test_accuracy)
        test_means.append(test_mean)
        test_std = np.std(test_accuracy)
        test_stds.append(test_std)
        print(f"Test accuracy {test_mean} ({test_std})")

    if savepath is not None:
        df = pd.DataFrame()
        df['k'] = n_neighbours
        df['Train_mean'] = train_means
        df['Train_std'] = train_stds
        df['Test_mean'] = test_means
        df['Test_std'] = test_stds
        df.to_csv(Path(savepath), index=False)


def lda(data, labels):
    print("--- Running LDA")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    clf = LinearDiscriminantAnalysis()

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def support_vm(data, labels):
    print("--- Running SVM")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    clf = svm.SVC(kernel='rbf')

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def run_algorithms_raw(filepath: str):
    df = pd.read_csv(filepath)
    labels = df['Label']
    data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)
    #lda(data, labels)
    #rf_adaboost(data, labels)
    support_vm(data, labels)
    #naive_bayes(data, labels)
    #knn(data, labels)


def run_algorithms_preprocessed(filepath: str):
    """"""
    df = pd.read_csv(filepath)
    labels = df['Label']
    data = df.drop(labels=['Epoch', 'Label'], axis=1)
    #lda(data, labels)
    #rf_adaboost(data, labels)
    #support_vm(data, labels)
    #naive_bayes(data, labels)
    knn(data, labels, savepath='knn-P01_imagined_preprocessed_part6.csv')


def run_algorithms_feis(filepath: str):
    """"""
    df = pd.read_csv(filepath)
    labels = df['Label']
    data = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
    lda(data, labels)
    rf_adaboost(data, labels)
    support_vm(data, labels)
    naive_bayes(data, labels)
    knn(data, labels)


if __name__ == '__main__':
    #filepath = 'binary_preprocessed.csv'
    #run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 01 imagined preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_01/imagined/preprocessed.csv'
    run_algorithms_preprocessed(filepath)
    """
    print("----------------------------------------------------------------\n"
          "Participant 01 inner preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_01/inner/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 02 imagined preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_02/imagined/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 02 inner preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_02/inner/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 03 imagined preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_03/imagined/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 03 inner preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_03/inner/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 04 imagined preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_04/imagined/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 04 inner preprocessed\n"
          "----------------------------------------------------------------")
    filepath = '../data_preprocessed/participant_04/inner/preprocessed.csv'
    run_algorithms_preprocessed(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 01 imagined raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_01/imagined/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 01 inner raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_01/inner/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 02 imagined raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_02/imagined/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 02 inner raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_02/inner/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 03 imagined raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_03/imagined/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 03 inner raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_03/inner/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 04 imagined raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_04/imagined/thinking_labelled.csv'
    run_algorithms_raw(filepath)

    print("----------------------------------------------------------------\n"
          "Participant 04 inner raw\n"
          "----------------------------------------------------------------")
    filepath = '../raw_eeg_recordings_labelled/participant_04/inner/thinking_labelled.csv'
    run_algorithms_raw(filepath)"""
