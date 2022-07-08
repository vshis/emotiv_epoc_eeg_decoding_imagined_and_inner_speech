import antropy
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import integrate, stats
import sys


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def mean(x):
    return np.mean(x)


def absmean(x):
    return np.mean(np.abs(x))


def maximum(x):
    return np.max(x)


def absmaximum(x):
    return np.max(np.abs(x))


def minimum(x):
    return np.min(x)


def absminimum(x):
    return np.min(np.abs(x))


def minplusmax(x):
    return np.max(x) + np.min(x)


def maxminusmin(x):
    return np.max(x) - np.min(x)


def curvelength(x):
    cl = 0
    for i in range(x.shape[0] - 1):
        cl += abs(x[i] - x[i + 1])
    return cl


def energy(x):
    return np.sum(np.multiply(x, x))


def non_linear_energy(x):
    # NLE(x[n]) = x[n]^2 - x[n+1]*x[n-1]
    # NLE = sum(NLE(x[n])
    return np.sum((x[1:-1] ** 2) - (x[2:] * x[:-2]))


def spectral_entropy(x):
    return antropy.spectral_entropy(x, sampling_frequency, method="welch", normalize=True)


def standard_deviation(x):
    return np.std(x)


def variance(x):
    return np.var(x)


def skewness(x):
    return stats.skew(x)


def kurtosis(x):
    return stats.kurtosis(x)


# entropy features

def sample_entropy(x):
    return antropy.sample_entropy(x, order=2, metric='chebyshev')


def perm_entropy(x):
    return antropy.perm_entropy(x, order=3, normalize=True)


def svd_entropy(x):
    return antropy.svd_entropy(x, order=3, delay=1, normalize=True)


def app_entropy(x):
    return antropy.app_entropy(x, order=2, metric='chebyshev')


def petrosian(x):
    return antropy.petrosian_fd(x)


def katz(x):
    return antropy.katz_fd(x)


def higuchi(x):
    return antropy.higuchi_fd(x, kmax=10)


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def dfa(x):
    return antropy.detrended_fluctuation(x)


def window_data(data: np.ndarray):
    window_len = sampling_frequency // 2  # windows of length 500ms
    stride = window_len // 2

    no_offset_windows = np.split(data, 10)  # 10 since 5 second epochs with 500ms windows
    offset_windows = np.split(data[stride:-stride], 9)  #
    windows = [0] * 19
    windows[::2] = no_offset_windows  # every second element from 0th element of list
    windows[1::2] = offset_windows  # every second element from 1st element of list

    return np.array(windows, dtype=np.float32)


def extract_features_from_window(window: np.ndarray):
    """
    For a single window, extracts features into an array of shape (n.features, electrodes)
    Returns a flattened vector of (n.features, electrodes)
    """
    output = np.zeros((len(features_functions), window.shape[1]))
    for i in range(len(features_functions)):
        for j in range(window.shape[1]):
            output[i, j] = features_functions[i](window[:, j])

    output = output.reshape(-1)

    return output


def make_simple_features(windowed_data: np.ndarray):
    """Makes simple features (without differentials)"""
    simple_features = []

    for window_index in range(len(windowed_data)):
        simple_features.append(extract_features_from_window(windowed_data[window_index]))

    return np.array(simple_features)


def make_d_and_dd_features(features_array: np.ndarray):
    """
    :param features_array: array of undifferentiated features
    :return: array of undifferentiated, differentiated and double differentiated features respectively
    """
    d = np.diff(features_array, axis=0)
    dd = np.diff(d, axis=0)
    return np.hstack((features_array[2:], d[1:], dd))


def combined_features(data):
    windows = window_data(data)
    simple_features = make_simple_features(windows)
    features = make_d_and_dd_features(simple_features)
    return features


def get_labels(experiment, features_dir):
    print(f"Saving labels for experiment {experiment}")

    if not Path(features_dir + "/" + experiment.stem).is_dir():
        Path(features_dir + "/" + experiment.stem).mkdir()

    experiment_df = pd.read_csv(experiment)
    labels = experiment_df['Label']
    labels.to_csv(Path(features_dir + "/" + experiment.stem + "/labels.csv"), index=False)


def get_features_list():
    # list of all features plus first differential, plus second differential of each feature
    features_list = []
    features_list += [function.__name__ for function in features_functions]
    features_list += ["d_" + function.__name__ for function in features_functions]
    features_list += ["dd_" + function.__name__ for function in features_functions]
    return features_list


def make_and_save_features(experiment, features_dir):
    print(f"Making features for {experiment}.")
    if not Path(features_dir + "/" + experiment.stem).is_dir():
        Path(features_dir + "/" + experiment.stem).mkdir()

    experiment_df = pd.read_csv(experiment)
    eeg_data_df = experiment_df[
        ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']]
    n_epochs = len(eeg_data_df.index) / 256 / 5  # sampling frequency 256Hz with 5 seconds of data in each token
    if not n_epochs % 1 == 0:
        raise TypeError(f"{experiment} does not seem to contain the right amount of data."
                        f"Length of data received is {len(eeg_data_df.index)}")

    eeg_data_np = np.split(eeg_data_df.to_numpy(), n_epochs)
    epochs = []

    for index, epoch in enumerate(eeg_data_np):
        print(f"Making features for epoch {index} of {len(eeg_data_np)}, experiment {experiment}.")
        epoch = combined_features(epoch)
        epochs.append(epoch)

    epochs = np.array(epochs, dtype=np.float32)
    np.save(Path(features_dir + "/" + experiment.stem + "/" + "features"), epochs)


if __name__ == '__main__':
    if not Path("features").is_dir():
        Path("features").mkdir()

    sampling_frequency = 256
    experiments_dir = "testing_experiment"
    features_dir = "features"

    experiments_list = list(Path(experiments_dir).glob('*.csv'))
    features_functions = [mean, absmean, maximum, absmaximum, minimum, absminimum, minplusmax, maxminusmin, curvelength,
                          energy, non_linear_energy, spectral_entropy, standard_deviation, variance, skewness, kurtosis,
                          sample_entropy, perm_entropy, svd_entropy, app_entropy, petrosian, katz, higuchi, rms, dfa]
    get_features_list()
    get_labels(experiments_list[0], features_dir)
    make_and_save_features(experiments_list[0], features_dir)
