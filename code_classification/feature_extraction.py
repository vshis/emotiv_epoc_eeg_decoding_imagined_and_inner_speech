from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import mne
from mne_features.feature_extraction import extract_features

SAMPLING_RATE = 256  # Hz
EPOCH_DURATION = 3  # seconds


def using_mne_features():
    """Reads a -epo.fif file"""
    mne_epochs = mne.read_epochs(Path('../data_preprocessed/epochs-epo.fif'))

    labels = mne_epochs.metadata['Labels']
    data = mne_epochs.get_data()

    selected_funcs = {#'ptp_amp',
                      #'mean',
                      #'std',
                      #'skewness',
                      #'kurtosis',
                      'rms',
                      'higuchi_fd',
                      'line_length',
                      'spect_entropy'
                      }
    features = extract_features(data, mne_epochs.info['sfreq'], selected_funcs)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    clf = svm.SVC(kernel='rbf')

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def window_data(epoch_df: pd.DataFrame):
    channels_df = epoch_df.drop(columns=['Epoch'], axis=1)
    channels_df.reset_index(drop=True, inplace=True)
    channels_df.drop(channels_df.tail(1).index, inplace=True)

    windows_no_overlap = np.array_split(channels_df, EPOCH_DURATION * 2)  # 0.5 s windows
    half_windows = [np.array_split(window, 2) for window in windows_no_overlap]  # split each window in 2
    windows = []
    for index in range(len(windows_no_overlap) - 1):
        windows.append(windows_no_overlap[index])
        windows.append(pd.concat([half_windows[index][1], half_windows[index + 1][0]]))
    windows.append(windows_no_overlap[-1])

    print(windows)
    print(len(windows))
    #window = channels_df.loc[0:SAMPLING_RATE/2 - 1]
    return


if __name__ == '__main__':
    filepath = '../data_preprocessed/preprocessed.csv'
    df = pd.read_csv(filepath)
    labels_df = df['Label']
    df.drop(columns=['Label'], axis=1, inplace=True)

    number_of_epochs = df['Epoch'].max()
    for epoch_number in range(number_of_epochs + 1):
        epoch = df[df['Epoch'] == epoch_number]
        windows = window_data(epoch)
        raise StopIteration()

#print(f"Precision: {metrics.precision_score(y_test, y_pred)}")
#print(f"Recall: {metrics.recall_score(y_test, y_pred)}")

#datapath = Path('../data_preprocessed/preprocessed.csv')
#df = pd.read_csv(datapath)
#labels_df = df['Label']
#epochs_df = df['Epoch']
#df_channels = df.drop(labels=['Epoch', 'Label'], axis=1)
#np_channels = df_channels.to_numpy()
#print(np_channels.shape)
#print(signal.welch(np_channels)[1].shape)

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