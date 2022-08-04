from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import mne
from mne_features.feature_extraction import extract_features

mne_epochs = mne.read_epochs(Path('../data_preprocessed/epochs-epo.fif'))

labels = mne_epochs.metadata['Labels']

data = mne_epochs.get_data()
selected_funcs = {'ptp_amp',
                  'mean',
                  'std',
                  'skewness',
                  'kurtosis',
                  'rms',
                  'higuchi_fd',
                  'line_length',
                  'spect_entropy',
                  'energy_freq_bands'}
features = extract_features(data, mne_epochs.info['sfreq'], selected_funcs)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
clf = svm.SVC(kernel='rbf')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

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