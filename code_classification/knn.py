import pandas as pd
import feature_extraction
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


def knn(data, labels, using_features=False, binary=False, feis=False):
    print("--- Running kNN")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    n_neighbours = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]

    if using_features:
        n_neighbours = [1, 5, 13, 37, 101, 445, 845, 1001, 1115]

    if binary:
        n_neighbours = [1, 5, 13, 25, 37, 53, 75, 101, 139]

    if feis:
        n_neighbours = [1, 5, 13, 25, 75, 101, 223, 375, 555]

    train_accuracies = []
    test_accuracies = []

    for k in n_neighbours:
        print(f"Using {k} nearest neighbors...")

        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
        print(f"Train accuracy: {train_accuracy}")
        train_accuracies.append(train_accuracy)

        y_pred = model.predict(x_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy}")
        test_accuracies.append(test_accuracy)

    return n_neighbours, train_accuracies, test_accuracies


def run_algorithms(filepath: str, algorithm: str, data_type: str, binary=False, feis=False):
    data_types = ['raw', 'preprocessed', 'feis_raw']
    if data_type not in data_types:
        raise ValueError(f"Given data type {data_type} is not in available data types: {data_types}")

    df = pd.read_csv(filepath)
    labels = df['Label']
    if data_type == 'raw':
        data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)
    elif data_type == 'preprocessed':
        data = df.drop(labels=['Epoch', 'Label'], axis=1)
    elif data_type == 'feis_raw':
        data = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
    if algorithm == 'knn':
        return knn(data, labels, binary=binary, feis=feis)


def knn_for_participant(participant_n: int):
    speech_modes = ['imagined', 'inner']
    print(f"Participant number {participant_n}")

    accuracies = {}
    for speech_mode in speech_modes:
        print(f"------------\nSpeech mode {speech_mode}\n------------")
        # Raw
        print("------ Raw...")
        filepath = f'../raw_eeg_recordings_labelled/participant_0{participant_n}/{speech_mode}/thinking_labelled.csv'
        _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='raw')
        accuracies[f'raw_{speech_mode}_train'] = train_accuracies
        accuracies[f'raw_{speech_mode}_test'] = test_accuracies

        # Preprocessed
        print("------ Preprocessed...")
        filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
        _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='preprocessed')
        accuracies[f'preprocessed_{speech_mode}_train'] = train_accuracies
        accuracies[f'preprocessed_{speech_mode}_test'] = test_accuracies

        # Features
        print("------ Features...")
        filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
        features, labels = feature_extraction.get_features(filepath, verbose=False)
        _, train_accuracies, test_accuracies = knn(features, labels, using_features=True)
        accuracies[f'features_{speech_mode}_train'] = train_accuracies
        accuracies[f'features_{speech_mode}_test'] = test_accuracies

    ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]

    df = pd.DataFrame()
    df['k'] = ks

    for header, values in list(accuracies.items()):
        df[header] = values

    df.to_csv(Path(f'knn_results/participant_0{participant_n}.csv'), index=False)


def knn_for_binary():
    accuracies = {}
    # Raw
    print("------ Raw...")
    filepath = f'binary_data/p01_imagined_raw_binary.csv'
    _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='raw', binary=True)
    accuracies[f'raw_train'] = train_accuracies
    accuracies[f'raw_test'] = test_accuracies

    # Preprocessed
    print("------ Preprocessed...")
    filepath = f'binary_data/p01_imagined_preprocessed_binary.csv'
    _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='preprocessed', binary=True)
    accuracies[f'preprocessed_train'] = train_accuracies
    accuracies[f'preprocessed_test'] = test_accuracies

    # Features
    print("------ Features...")
    filepath = f'binary_data/p01_imagined_preprocessed_binary.csv'
    features, labels = feature_extraction.get_features(filepath, verbose=False)
    _, train_accuracies, test_accuracies = knn(features, labels, using_features=True, binary=True)
    accuracies[f'features_train'] = train_accuracies
    accuracies[f'features_test'] = test_accuracies

    ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]

    df = pd.DataFrame()
    df['k'] = ks

    for header, values in list(accuracies.items()):
        df[header] = values

    df.to_csv(Path(f'knn_results/binary.csv'), index=False)


def knn_for_raw_only():
    speech_modes = ['imagined', 'inner']
    accuracies = {}
    for speech_mode in speech_modes:
        print(f"------------\nSpeech mode {speech_mode}\n------------")
        filepath = f'../raw_eeg_recordings_labelled/participant_00/{speech_mode}/thinking_labelled.csv'
        _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='raw')
        accuracies[f'raw_{speech_mode}_train'] = train_accuracies
        accuracies[f'raw_{speech_mode}_test'] = test_accuracies

    ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]

    df = pd.DataFrame()
    df['k'] = ks

    for header, values in list(accuracies.items()):
        df[header] = values

    df.to_csv(Path(f'knn_results/participant_00.csv'), index=False)


def knn_for_feis():
    accuracies = {}
    # Raw
    print("------ Raw...")
    filepath = f'../../feis-01-thinking.csv'
    _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='feis_raw', feis=True)
    accuracies[f'raw_train'] = train_accuracies
    accuracies[f'raw_test'] = test_accuracies

    # Preprocessed
    print("------ Preprocessed...")
    filepath = f'../../feis_preprocessed/preprocessed.csv'
    _, train_accuracies, test_accuracies = run_algorithms(filepath, 'knn', data_type='preprocessed', feis=True)
    accuracies[f'preprocessed_train'] = train_accuracies
    accuracies[f'preprocessed_test'] = test_accuracies

    # Features
    print("------ Features...")
    filepath = f'../../feis_preprocessed/preprocessed.csv'
    features, labels = feature_extraction.get_features(filepath, verbose=False)
    _, train_accuracies, test_accuracies = knn(features, labels, using_features=True, feis=True)
    accuracies[f'features_train'] = train_accuracies
    accuracies[f'features_test'] = test_accuracies

    ks = [1, 5, 13, 25, 75, 101, 223, 375, 555]

    df = pd.DataFrame()
    df['k'] = ks

    for header, values in list(accuracies.items()):
        df[header] = values

    df.to_csv(Path(f'knn_results/feis.csv'), index=False)


if __name__ == '__main__':
    #participant_ns = [i for i in range(1, 5)]
    #for participant_n in participant_ns:
    #    knn_for_participant(participant_n)
    knn_for_feis()
