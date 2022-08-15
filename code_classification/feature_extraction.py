from pathlib import Path
import antropy
import scipy.signal
import torch
from scipy.signal import periodogram
from itertools import chain
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from hurst import compute_Hc
import mne
from mne_features.feature_extraction import extract_features
import cmath
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SAMPLING_FREQUENCY = 256  # Hz
EPOCH_DURATION = 3  # seconds
NUMBER_OF_WINDOWS = 5


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 3))
        self.pool = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(16, 16 * 16, (14, 1))
        self.fc1 = nn.Linear(7 * 14, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        # x shape
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def cnn(data, labels):
    print("Using CNN")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    data = np.split(data, 319)
    labels = np.split(labels, 319)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    model = CNNModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(0, 5):
        print(f"Epoch number {epoch}")
        for i, inputs in enumerate(x_train, 0):
            # inputs = inputs.to(device)
            minibatch_labels = y_train[i]  # .to(device)
            running_loss = 0.0
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, minibatch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item
            if i % 50 == 49:
                print(f"[{epoch}, {i + 1:5d}] loss: {running_loss / 50:.3f}")
                running_loss = 0.0

    print("finished training")

    correct = 0
    total = 0

    with torch.no_grad():
        for i, inputs in enumerate(x_test, 0):
            inputs = inputs.to(device)
            minibatch_labels = y_test[i].to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += minibatch_labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct // total}")


def using_mne_features():
    """Reads an -epo.fif file"""
    mne_epochs = mne.read_epochs(Path('../data_preprocessed/participant_04/imagined/preprocessed-epo.fif'))
    """
    epoch1_np = mne_epochs.get_data()[0]
    ch_names = mne_epochs.ch_names
    # ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

    info = mne.create_info(ch_names, sfreq=256)
    raw = mne.io.RawArray(epoch1_np, info)

    event_duration = 0.6
    events = mne.make_fixed_length_events(raw, id=1, start=0, duration=event_duration)
    labels = mne_epochs.metadata['Labels']
    event_dict = {labels[1]: 1}
    epochs_mne = mne.Epochs(raw, events, event_id=event_dict, baseline=None, reject=None)

    print(epochs_mne.get_data().shape)

"""
    labels = mne_epochs.metadata['Labels']
    data = mne_epochs.get_data()

    selected_funcs = {  # 'ptp_amp',
        # 'mean',
        # 'std',
        # 'skewness',
        # 'kurtosis',
        'rms',
        'higuchi_fd',
        'line_length',
        'spect_entropy'
    }
    features = extract_features(data, mne_epochs.info['sfreq'], selected_funcs)
    print(labels.shape)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    clf = svm.SVC(kernel='rbf')

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def _feature_windows(epoch_df: pd.DataFrame):
    channels_df = epoch_df.drop(columns=['Epoch'], axis=1)
    # channels_df.reset_index(drop=True, inplace=True)
    # channels_df.drop(channels_df.tail(1).index, inplace=True)

    windows_no_overlap = np.array_split(channels_df, NUMBER_OF_WINDOWS)
    # half_windows = [np.array_split(window, 2) for window in windows_no_overlap]  # split each window in 2
    # windows = []
    # for index in range(len(windows_no_overlap) - 1):
    #    windows.append(windows_no_overlap[index])
    #    windows.append(pd.concat([half_windows[index][1], half_windows[index + 1][0]]))
    # windows.append(windows_no_overlap[-1])
    five_windows_features = []
    for window in windows_no_overlap:
        window_features = []
        for column in window:
            channel = window[column]
            sample_fs, power_spectral_density = periodogram(channel, fs=SAMPLING_FREQUENCY)
            hilbert = scipy.signal.hilbert(channel)

            # features
            higuchi = antropy.higuchi_fd(channel, kmax=10)
            katz = antropy.katz_fd(channel)
            hurst_exp, c, data = compute_Hc(channel, kind='change')
            spectral_entropy = antropy.spectral_entropy(channel, SAMPLING_FREQUENCY, nperseg=150, method='welch')
            spectral_power = (1 * SAMPLING_FREQUENCY / len(channel)) * np.mean(power_spectral_density)
            phase = cmath.phase(np.mean(hilbert))
            magnitute = np.linalg.norm(np.mean(hilbert))

            window_features.append([higuchi,
                                    katz,
                                    hurst_exp,
                                    spectral_entropy,
                                    spectral_power,
                                    phase,
                                    magnitute
                                    ])
        window_features = list(chain.from_iterable(window_features))
        five_windows_features.append(window_features)
    # returns array of shape (n, x), where n is number of windows and x is number of features * number of channels
    return np.array(five_windows_features)


def get_features(filepath: str, save_features: bool = False, savedir: str = None, verbose=True):
    df = pd.read_csv(filepath)
    labels = [df.loc[df['Epoch'] == epoch, 'Label'].iloc[0] for epoch in range(df['Epoch'].max() + 1)]
    labels_np = np.empty((0, 1), dtype=str)
    for label in labels:
        labels_np = np.append(labels_np, [label] * NUMBER_OF_WINDOWS)
    labels = labels_np
    df.drop(columns=['Label'], axis=1, inplace=True)

    number_of_epochs = df['Epoch'].max()
    features = []
    for epoch_number in range(number_of_epochs + 1):
        if verbose:
            print(f"Extracting features from epoch {epoch_number}/{number_of_epochs}")
        epoch = df[df['Epoch'] == epoch_number]
        windows_features = _feature_windows(epoch)
        features.append(windows_features)

    features = np.concatenate(features)

    if save_features is True:
        if savedir is None:
            raise ValueError("Provide a savepath for the features")
        np.save(file=Path(f"{savedir}features.npy"), arr=features)
        np.save(file=Path(f"{savedir}labels.npy"), arr=labels)

    return features, labels


def using_windows(filepath: str = None, preloaded_features=None, preloaded_labels=None, save_features_labels=False,
                  savedir=None):
    if filepath is None and preloaded_features is None:
        raise ValueError("Please provide either filepath or preloaded features")
    elif filepath is not None:
        features, labels = get_features(filepath)

    elif preloaded_features is not None and preloaded_labels is not None:
        features = preloaded_features
        labels = preloaded_labels
    else:
        raise ValueError("No conditions are met.")

    rf_adaboost(features, labels)  # 8 - 8.9% accuracy
    naive_bayes(features, labels)  # 8.5% accuracy
    knn(features, labels)  # 8% accuracy with around 100 neighbors


def rf_adaboost(data, labels):
    print("Running RF Adaboost")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    model = AdaBoostClassifier(n_estimators=310, learning_rate=0.3)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def naive_bayes(data, labels):
    print("Running Naive Bayes")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    model = GaussianNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def knn(data, labels):
    print("Running kNN")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    n_neighbours = [i for i in range(1, 101, 5)]
    for k in n_neighbours:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        print(f"Using {k} nearest neighbors...")
        print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def support_vm(data, labels):
    print("Running SVM")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    clf = svm.SVC(kernel='rbf')

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    filepath = '../data_preprocessed/participant_01/imagined/preprocessed.csv'
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    using_windows(preloaded_features=features, preloaded_labels=labels)
