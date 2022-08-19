from pathlib import Path
import antropy
import scipy.signal
from scipy.signal import periodogram
from itertools import chain
import numpy as np
import pandas as pd
from hurst import compute_Hc
import librosa
import cmath

SAMPLING_FREQUENCY = 256  # Hz
NUMBER_OF_WINDOWS = 5
OVERLAP = False


def extract_linear(channel):
    mean = np.mean(channel)
    abs_mean = abs(mean)
    std = np.std(channel)
    sum_data = np.sum(channel)
    variance = np.var(channel)
    maximum = np.max(channel)
    abs_maximum = abs(maximum)
    minimum = np.min(channel)
    abs_minimum = abs(minimum)
    min_plus_max = maximum + minimum
    max_minus_min = maximum - minimum
    features_list = [
        mean,
        abs_mean,
        std,
        sum_data,
        variance,
        maximum,
        abs_maximum,
        minimum,
        abs_minimum,
        min_plus_max,
        max_minus_min
    ]
    return features_list


def extract_non_linear(channel):
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
    features_list = [higuchi,
                     katz,
                     hurst_exp,
                     spectral_entropy,
                     spectral_power,
                     phase,
                     magnitute
                     ]
    return features_list


def extract_mfcc(channel):
    channel = channel.values
    mfccs = librosa.feature.mfcc(y=channel, sr=SAMPLING_FREQUENCY, n_mfcc=13, fmin=13, n_fft=256)
    return list(chain.from_iterable(mfccs))


def _feature_windows(epoch_df: pd.DataFrame, features_type='frequency'):
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
            if features_type == 'mfcc':
                features_list = extract_mfcc(channel)
            elif features_type == 'linear':
                features_list = extract_linear(channel)
            else:
                features_list = extract_non_linear(channel)
            window_features.append(features_list)
        window_features = list(chain.from_iterable(window_features))
        five_windows_features.append(window_features)
    # returns array of shape (n, x), where n is number of windows and x is number of features * number of channels
    return np.array(five_windows_features)


def get_features(filepath: str, savedir: str = None, verbose=True, features_type='frequency'):
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
        windows_features = _feature_windows(epoch, features_type)
        features.append(windows_features)

    features = np.concatenate(features)

    if savedir is not None:
        np.save(file=Path(f"{savedir}{features_type}_features.npy"), arr=features)
        np.save(file=Path(f"{savedir}{features_type}_labels.npy"), arr=labels)

    return features, labels


def extract_participants_features(features_type='frequency'):
    for participant_n in range(1, 5):
        print(f"Participant {participant_n}...")
        for speech_type in ['imagined', 'inner']:
            print(f"Speech type: {speech_type}. Features type: {features_type}.")
            filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_type}/preprocessed.csv'
            save_dir = f'features/even_windows/participant_0{participant_n}/{speech_type}/'
            get_features(filepath=filepath, savedir=save_dir, verbose=False, features_type=features_type)


def extract_features_binary(features_type='frequency'):
    print(f"Binary. Features type: {features_type}.")
    filepath = 'binary_data/p01_imagined_preprocessed_binary.csv'
    save_dir = 'features/even_windows/binary/'
    get_features(filepath=filepath, savedir=save_dir, verbose=False, features_type=features_type)


def extract_features_feis(features_type='frequency'):
    print(f"FEIS. Features type: {features_type}.")
    filepath = 'feis_data/preprocessed.csv'
    save_dir = 'features/even_windows/feis/'
    get_features(filepath=filepath, savedir=save_dir, verbose=False, features_type=features_type)


if __name__ == '__main__':
    #extract_participants_features(features_type='linear')
    extract_features_binary(features_type='linear')
    #extract_features_feis()
    extract_features_feis(features_type='linear')
    exit(0)
