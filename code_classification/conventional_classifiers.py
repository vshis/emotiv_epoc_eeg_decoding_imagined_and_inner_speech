from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from pathlib import Path
import os


def _get_savedir(model):
    savedir = f'classification_results/{model}'

    if str(model) == 'SVC()':
        savedir = f'classification_results/{model}_{model.kernel}'
    elif str(model) == 'AdaBoostClassifier()':
        savedir = f'classification_results/{model}_{model.n_estimators}_{model.learning_rate}'
    elif str(model) == 'KNeighborsClassifier()':
        savedir = f'classification_results/{model}_{model.n_neighbors}'

    return savedir


def run_model(data, labels, model):
    """Returns accuracies: mean_train, std_train, mean_test, std_test"""
    print(f"--- Running {model}")
    folds = 5
    kfold = KFold(folds, shuffle=True)

    train_accuracies = []
    test_accuracies = []

    counter = 0
    for train_id, test_id in kfold.split(data, labels):
        print(f"Fold {counter+1}/{folds}")
        x_train = data.iloc[train_id]
        y_train = labels.iloc[train_id]
        x_test = data.iloc[test_id]
        y_test = labels.iloc[test_id]

        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
        train_accuracies.append(train_accuracy)

        y_pred = model.predict(x_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_accuracies.append(test_accuracy)
        counter += 1

    mean_train = float(np.mean(train_accuracies))
    std_train = float(np.std(train_accuracies))
    mean_test = float(np.mean(test_accuracies))
    std_test = float(np.std(test_accuracies))
    print(f"Mean (std) train accuracy:{mean_train} ({std_train})")
    print(f"Mean (std) test accuracy:{mean_test} ({std_test})")

    return mean_train, std_train, mean_test, std_test


def prep_and_run(model, filepath: str, data_type: str):
    data_types = ['raw', 'preprocessed', 'feis_raw']
    if data_type not in data_types:
        raise ValueError(f"Unsupported data type.")

    df = pd.read_csv(filepath)
    labels = df['Label']
    if data_type == 'raw':
        data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)
    elif data_type == 'preprocessed':
        data = df.drop(labels=['Epoch', 'Label'], axis=1)
    elif data_type == 'feis_raw':
        data = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
    print(data.shape)
    exit()
    return run_model(data, labels, model)


def run_model_for_participant(participant_n: int, model):
    speech_modes = ['imagined', 'inner']
    print(f"___________________\nModel: {model}")
    print(f"Participant number {participant_n}")

    accuracies = {}
    for speech_mode in speech_modes:
        print(f"------\nSpeech mode: {speech_mode}\n------")
        # Raw
        print("------------- Raw")
        filepath = f'../raw_eeg_recordings_labelled/participant_0{participant_n}/{speech_mode}/thinking_labelled.csv'
        mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'raw')
        accuracies[f'raw_{speech_mode}_train_mean'] = mean_train
        accuracies[f'raw_{speech_mode}_train_std'] = std_train
        accuracies[f'raw_{speech_mode}_test_mean'] = mean_test
        accuracies[f'raw_{speech_mode}_test_std'] = std_test

        # Preprocessed
        print("------------- Preprocessed")
        filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
        mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'preprocessed')
        accuracies[f'preprocessed_{speech_mode}_train_mean'] = mean_train
        accuracies[f'preprocessed_{speech_mode}_train_std'] = std_train
        accuracies[f'preprocessed_{speech_mode}_test_mean'] = mean_test
        accuracies[f'preprocessed_{speech_mode}_test_std'] = std_test

        # Features
        print("------------- Frequency Features")
        features = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/features.npy')
        labels = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/labels.npy')
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)[0]
        mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
        accuracies[f'features_{speech_mode}_train_mean'] = mean_train
        accuracies[f'features_{speech_mode}_train_std'] = std_train
        accuracies[f'features_{speech_mode}_test_mean'] = mean_test
        accuracies[f'features_{speech_mode}_test_std'] = std_test

        # MFCC Features
        print("------------- MFCC Features")
        features = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/mfcc_features.npy')
        labels = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/mfcc_labels.npy')
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)[0]
        mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
        accuracies[f'mfcc_{speech_mode}_train_mean'] = mean_train
        accuracies[f'mfcc_{speech_mode}_train_std'] = std_train
        accuracies[f'mfcc_{speech_mode}_test_mean'] = mean_test
        accuracies[f'mfcc_{speech_mode}_test_std'] = std_test

        # Linear Features
        print("------------- Linear Features")
        features = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/linear_features.npy')
        labels = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/linear_labels.npy')
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)[0]
        mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
        accuracies[f'linear_{speech_mode}_train_mean'] = mean_train
        accuracies[f'linear_{speech_mode}_train_std'] = std_train
        accuracies[f'linear_{speech_mode}_test_mean'] = mean_test
        accuracies[f'linear_{speech_mode}_test_std'] = std_test

    savedir = _get_savedir(model)

    # if saving to new file
    #df = pd.DataFrame()
    #df['Method'] = [model]
    # if adding to existing file
    df = pd.read_csv(f'{savedir}/participant_0{participant_n}.csv')

    for header, values in list(accuracies.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/participant_0{participant_n}.csv'), index=False)


def run_model_for_binary(model):
    print(f"Model: {model}")
    print("Binary classification")

    accuracies = {}
    # Raw
    """
    print("------------- Raw")
    filepath = f'binary_data/p01_imagined_raw_binary.csv'
    mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'raw')
    accuracies[f'raw_train_mean'] = mean_train
    accuracies[f'raw_train_std'] = std_train
    accuracies[f'raw_test_mean'] = mean_test
    accuracies[f'raw_test_std'] = std_test

    # Preprocessed
    print("------------- Preprocessed")
    filepath = f'binary_data/p01_imagined_preprocessed_binary.csv'
    mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'preprocessed')
    accuracies[f'preprocessed_train_mean'] = mean_train
    accuracies[f'preprocessed_train_std'] = std_train
    accuracies[f'preprocessed_test_mean'] = mean_test
    accuracies[f'preprocessed_test_std'] = std_test

    # Features
    print("------------- Frequency Features")
    features = np.load(f'features/even_windows/binary/features.npy')
    labels = np.load(f'features/even_windows/binary/labels.npy')
    features = pd.DataFrame(features)
    labels = pd.DataFrame(labels)[0]
    mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
    accuracies[f'features_train_mean'] = mean_train
    accuracies[f'features_train_std'] = std_train
    accuracies[f'features_test_mean'] = mean_test
    accuracies[f'features_test_std'] = std_test

    # MFCC features
    print("------------- MFCC Features")
    features = np.load(f'features/even_windows/binary/mfcc_features.npy')
    labels = np.load(f'features/even_windows/binary/mfcc_labels.npy')
    features = pd.DataFrame(features)
    labels = pd.DataFrame(labels)[0]
    mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
    accuracies[f'mfcc_train_mean'] = mean_train
    accuracies[f'mfcc_train_std'] = std_train
    accuracies[f'mfcc_test_mean'] = mean_test
    accuracies[f'mfcc_test_std'] = std_test"""

    # Linear features
    print("------------- Linear Features")
    features = np.load(f'features/even_windows/binary/linear_features.npy')
    labels = np.load(f'features/even_windows/binary/linear_labels.npy')
    features = pd.DataFrame(features)
    labels = pd.DataFrame(labels)[0]
    mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
    accuracies[f'linear_train_mean'] = mean_train
    accuracies[f'linear_train_std'] = std_train
    accuracies[f'linear_test_mean'] = mean_test
    accuracies[f'linear_test_std'] = std_test

    savedir = _get_savedir(model)

    # if saving to new file
    #df = pd.DataFrame()
    #df['Method'] = [model]
    # if adding to existing file
    df = pd.read_csv(f'{savedir}/binary.csv')

    for header, values in list(accuracies.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/binary.csv'), index=False)


def run_model_for_p00(model):
    speech_modes = ['imagined', 'inner']
    print(f"Model: {model}")
    print(f"Participant number 00")

    accuracies = {}
    for speech_mode in speech_modes:
        print(f"------\nSpeech mode: {speech_mode}\n------")
        # Raw
        print("------------- Raw")
        filepath = f'../raw_eeg_recordings_labelled/participant_00/{speech_mode}/thinking_labelled.csv'
        mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'raw')
        accuracies[f'raw_{speech_mode}_train_mean'] = mean_train
        accuracies[f'raw_{speech_mode}_train_std'] = std_train
        accuracies[f'raw_{speech_mode}_test_mean'] = mean_test
        accuracies[f'raw_{speech_mode}_test_std'] = std_test

    savedir = _get_savedir(model)

    # if saving to new file
    df = pd.DataFrame()
    df['Method'] = [model]
    # if adding to existing file
    #df = pd.read_csv(f'{savedir}/participant_00.csv')

    for header, values in list(accuracies.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/participant_00.csv'), index=False)


def run_model_for_feis(model):
    print(f"Model: {model}")
    print(f"FEIS classification")

    accuracies = {}

    # Raw
    """print("------------- Raw")
    filepath = f'feis_data/feis-01-thinking.csv'
    mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'feis_raw')
    accuracies[f'raw_train_mean'] = mean_train
    accuracies[f'raw_train_std'] = std_train
    accuracies[f'raw_test_mean'] = mean_test
    accuracies[f'raw_test_std'] = std_test

    # Preprocessed
    print("------------- Preprocessed")
    filepath = f'feis_data/preprocessed.csv'
    mean_train, std_train, mean_test, std_test = prep_and_run(model, filepath, 'preprocessed')
    accuracies[f'preprocessed_train_mean'] = mean_train
    accuracies[f'preprocessed_train_std'] = std_train
    accuracies[f'preprocessed_test_mean'] = mean_test
    accuracies[f'preprocessed_test_std'] = std_test

    # Frequency Features
    print("------------- Frequency Features")
    features = np.load(f'features/even_windows/feis/features.npy')
    labels = np.load(f'features/even_windows/feis/labels.npy')
    features = pd.DataFrame(features)
    labels = pd.DataFrame(labels)[0]
    mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
    accuracies[f'features_train_mean'] = mean_train
    accuracies[f'features_train_std'] = std_train
    accuracies[f'features_test_mean'] = mean_test
    accuracies[f'features_test_std'] = std_test

    # MFCC Features
    print("------------- MFCC Features")
    features = np.load(f'features/even_windows/feis/mfcc_features.npy')
    labels = np.load(f'features/even_windows/feis/mfcc_labels.npy')
    features = pd.DataFrame(features)
    labels = pd.DataFrame(labels)[0]
    mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
    accuracies[f'mfcc_train_mean'] = mean_train
    accuracies[f'mfcc_train_std'] = std_train
    accuracies[f'mfcc_test_mean'] = mean_test
    accuracies[f'mfcc_test_std'] = std_test
"""
    # Linear Features
    print("------------- Linear Features")
    features = np.load(f'features/even_windows/feis/linear_features.npy')
    labels = np.load(f'features/even_windows/feis/linear_labels.npy')
    features = pd.DataFrame(features)
    labels = pd.DataFrame(labels)[0]
    mean_train, std_train, mean_test, std_test = run_model(features, labels, model)
    accuracies[f'linear_train_mean'] = mean_train
    accuracies[f'linear_train_std'] = std_train
    accuracies[f'linear_test_mean'] = mean_test
    accuracies[f'linear_test_std'] = std_test

    savedir = _get_savedir(model)

    # if saving to new file
    #df = pd.DataFrame()
    #df['Method'] = [model]
    # if adding to existing file
    df = pd.read_csv(f'{savedir}/feis.csv')

    for header, values in list(accuracies.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/feis.csv'), index=False)


if __name__ == '__main__':
    models = [
        LinearDiscriminantAnalysis(),  # LDA
        SVC(kernel='rbf', cache_size=2000),  # SVM
        SVC(kernel='poly', degree=2, cache_size=2000),  # SVM
        SVC(kernel='poly', degree=3, cache_size=2000),  # SVM
        SVC(kernel='poly', degree=4, cache_size=2000),  # SVM
        SVC(kernel='sigmoid', cache_size=2000),  # SVM
        GaussianNB(),  # naive bayes
        AdaBoostClassifier(n_estimators=50, learning_rate=1.0),  # RF
        AdaBoostClassifier(n_estimators=100, learning_rate=1.0),  # RF
        AdaBoostClassifier(n_estimators=200, learning_rate=1.0),  # RF
        AdaBoostClassifier(n_estimators=400, learning_rate=1.0),  # RF
        AdaBoostClassifier(n_estimators=50, learning_rate=1.0),  # RF
        AdaBoostClassifier(n_estimators=50, learning_rate=0.5),  # RF
        AdaBoostClassifier(n_estimators=50, learning_rate=0.1),  # RF
        AdaBoostClassifier(n_estimators=50, learning_rate=1.5),  # RF
        AdaBoostClassifier(n_estimators=50, learning_rate=2.0),  # RF
        KNeighborsClassifier(n_neighbors=1, n_jobs=-2),  # kNN
        KNeighborsClassifier(n_neighbors=5, n_jobs=-2),  # kNN
        KNeighborsClassifier(n_neighbors=13, n_jobs=-2),  # kNN
        KNeighborsClassifier(n_neighbors=25, n_jobs=-2),  # kNN
        KNeighborsClassifier(n_neighbors=53, n_jobs=-2),  # kNN
        KNeighborsClassifier(n_neighbors=89, n_jobs=-2),  # kNN
        KNeighborsClassifier(n_neighbors=131, n_jobs=-2),  # kNN
              ]
    participants = [i for i in range(1, 5)]

    for model in models:
        for participant_n in participants:
            run_model_for_participant(participant_n, model)
        #run_model_for_binary(model)
        #run_model_for_p00(model)
        #run_model_for_feis(model)
