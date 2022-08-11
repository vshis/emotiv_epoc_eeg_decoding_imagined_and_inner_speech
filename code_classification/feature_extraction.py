import math
from pathlib import Path
import antropy
import scipy.signal
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

SAMPLING_FREQUENCY = 256  # Hz
EPOCH_DURATION = 3  # seconds
NUMBER_OF_WINDOWS = 5


def using_mne_features():
    """Reads a -epo.fif file"""
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


def feature_windows(epoch_df: pd.DataFrame):
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
            higuchi = antropy.higuchi_fd(window[column], kmax=10)
            hurst_exp, c, data = compute_Hc(window[column], kind='change')
            spectral_entropy = antropy.spectral_entropy(window[column], SAMPLING_FREQUENCY, nperseg=150, method='welch')
            spectral_power = (1 * SAMPLING_FREQUENCY / len(channel)) * np.mean(power_spectral_density)
            phase = cmath.phase(np.mean(hilbert))
            magnitute = np.linalg.norm(np.mean(hilbert))

            window_features.append([higuchi,
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


def using_windows():
    filepath = '../data_preprocessed/participant_02/imagined/preprocessed.csv'
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
        print(f"Extracting features from epoch {epoch_number}/{number_of_epochs}")
        epoch = df[df['Epoch'] == epoch_number]
        windows_features = feature_windows(epoch)
        features.append(windows_features)

    features = np.concatenate(features)

    rf_adaboost(features, labels)  # 8 - 8.9% accuracy
    # naive_bayes(features, labels)  # 8.5% accuracy
    # knn(features, labels)  # 8% accuracy with around 100 neighbors


def rf_adaboost(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    model = AdaBoostClassifier(n_estimators=310, learning_rate=0.3)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def naive_bayes(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    model = GaussianNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def knn(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    n_neighbours = [100, 105, 110, 115]
    for k in n_neighbours:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        print(f"Using {k} nearest neighbors...")
        print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


"""
    clf = svm.SVC(kernel='rbf')

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")"""

if __name__ == '__main__':
    using_windows()

# print(f"Precision: {metrics.precision_score(y_test, y_pred)}")
# print(f"Recall: {metrics.recall_score(y_test, y_pred)}")

# datapath = Path('../data_preprocessed/preprocessed.csv')
# df = pd.read_csv(datapath)
# labels_df = df['Label']
# epochs_df = df['Epoch']
# df_channels = df.drop(labels=['Epoch', 'Label'], axis=1)
# np_channels = df_channels.to_numpy()
# print(np_channels.shape)
# print(signal.welch(np_channels)[1].shape)

"""
cancer = datasets.load_breast_cancer()
reshaped = np.reshape(cancer.data, (569, 15, 2))
print(cancer.target.shape)
print(reshaped.shape)

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=69)

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
"""
