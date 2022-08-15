from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import pandas as pd
import numpy as np
import feature_extraction
from pathlib import Path


def lda(data, labels):
    """Returns accuracies: mean_train, std_train, mean_test, std_test"""
    print("--- Running LDA")

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
    print(f"Train accuracy: {train_accuracy}")

    y_pred = model.predict(x_test)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy}")

    return train_accuracy, test_accuracy


def run_algorithm(filepath: str, data_type: str, binary=False, feis=False):
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
    return lda(data, labels)


def run_lda_for_participant(participant_n: int):
    speech_modes = ['imagined', 'inner']
    print(f"Participant number {participant_n}")

    accuracies = {}
    for speech_mode in speech_modes:
        print(f"------\nSpeech mode {speech_mode}\n------")
        # Raw
        print("------------- Raw")
        filepath = f'../raw_eeg_recordings_labelled/participant_0{participant_n}/{speech_mode}/thinking_labelled.csv'
        train_accuracy, test_accuracy = run_algorithm(filepath, 'raw')
        accuracies[f'raw_{speech_mode}_train'] = train_accuracy
        accuracies[f'raw_{speech_mode}_test'] = test_accuracy

        # Preprocessed
        print("------------- Preprocessed")
        filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
        train_accuracy, test_accuracy = run_algorithm(filepath, 'preprocessed')
        accuracies[f'preprocessed_{speech_mode}_train'] = train_accuracy
        accuracies[f'preprocessed_{speech_mode}_test'] = test_accuracy

        # Features
        print("------------- Features")
        filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
        features, labels = feature_extraction.get_features(filepath, verbose=False)
        train_accuracy, test_accuracy = lda(features, labels)
        accuracies[f'preprocessed_{speech_mode}_train'] = train_accuracy
        accuracies[f'preprocessed_{speech_mode}_test'] = test_accuracy

    df = pd.DataFrame()
    df['Method'] = 'LDA'

    for header, values in list(accuracies.items()):
        df[header] = values

    df.to_csv(Path(f'lda_results/participant_0{participant_n}.csv'), index=False)


if __name__ == '__main__':
    participants = [i for i in range(1, 5)]
    for participant_n in participants:
        run_lda_for_participant(participant_n)
