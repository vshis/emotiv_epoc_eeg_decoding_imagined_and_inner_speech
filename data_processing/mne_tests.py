from itertools import chain

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from autoreject import AutoReject
from autoreject import Ransac


SAMPLING_FREQUENCY = 256  # hz
EPOCH_DURATION = 5.0  # seconds
COLUMNS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']


def load_raw_data(filepath: str, verbose=False, data_type='feis'):
    """
    This function is configured for raw FEIS data. Configure the function for data collected from other sources.
    :param data_type: feis (256Hz) or my (128Hz) data
    :param verbose: show imported data info
    :param filepath: path to the file of interest
    :return: Raw MNE (mne.io.RawArray) object of the data
    """
    data_types = ['feis', 'my']
    if data_type not in data_types:
        raise ValueError("Invalid data type. Expected on of: %s" % data_types)

    df = pd.read_csv(Path(filepath))
    df_labels = df['Label']

    if data_type == 'feis':  # FEIS data
        channels_df = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
    elif data_type == 'my':  # my data LABELED
        channels_df = df.drop(labels=['Time:128Hz', 'Epoch', 'Label', 'Stage'], axis=1)
    else:
        return
    sfreq = SAMPLING_FREQUENCY  # Hz

    channels_list = [channels_df[channel].to_numpy() for channel in channels_df.columns]

    channels_np = np.array(channels_list)

    ch_names = list(channels_df.columns)
    ch_types = 'eeg'
    montage_1020 = mne.channels.make_standard_montage('standard_1020')

    info = mne.create_info(ch_names, sfreq, ch_types)  # Raw MNE objects must have associated info with them

    raw = mne.io.RawArray(channels_np, info)  # create Raw MNE object from NumPy array
    raw.set_montage(montage_1020)
    if verbose:
        print(raw.info)
    return raw


def create_epochs_object(mne_raw_object: mne.io.RawArray, filepath: str, epoch_duration: float):
    events = mne.make_fixed_length_events(mne_raw_object, id=1, start=-2, duration=epoch_duration)
    event_dict = {'thinking': 1}
    epochs_mne = mne.Epochs(mne_raw_object,
                            events,
                            event_id=event_dict,
                            preload=True,
                            baseline=None,
                            tmin=-2, tmax=epoch_duration)
    df = pd.read_csv(Path(filepath))
    labels = [df.loc[df['Epoch'] == epoch, 'Label'].iloc[0] for epoch in range(df['Epoch'].max())]
    epochs_mne.metadata = pd.DataFrame(data=labels, columns=['Labels'], index=range(len(labels)))
    return epochs_mne


def filter_raw(raw_data: mne.io.RawArray):
    """
    High passes data over 1 Hz and notch filters at 50 Hz with 4 Hz bandwidth
    :param raw_data: instance of mne.io.RawArray
    :return: self, modifies RawArray directly
    """
    raw_data.filter(l_freq=1., h_freq=None)
    raw_data.notch_filter(50., trans_bandwidth=4.)


def save_mne_epochs_to_csv(epochs_mne: mne.Epochs):
    epochs_data_np_3d = epochs_mne.get_data()

    # reconstructing epochs data from 3D array of (epochs, channels, data) to 2D (channels, epochs * data)
    # also deleting first 2 seconds (2 * sampling freq) of data from the beginning of each epoch
    # as 2 seconds were added to each epoch earlier, it is fictitious data that we want to remove, so that each epoch
    # is 5 seconds long
    epochs_reshaped_list = [np.delete(epochs_data_np_3d[epoch_index], range(2 * SAMPLING_FREQUENCY + 1), axis=1)
                            for epoch_index
                            in range(epochs_data_np_3d.shape[0])]

    epochs_data_np_2d = np.concatenate(epochs_reshaped_list, axis=1)

    output_df = pd.DataFrame()
    labels_df = epochs_mne.metadata['Labels']

    labels_list = [[label] * int(SAMPLING_FREQUENCY * EPOCH_DURATION) for label in labels_df]
    labels_list = list(chain.from_iterable(labels_list))

    epochs_numbers_list = [[epoch_number] * int(SAMPLING_FREQUENCY * EPOCH_DURATION)
                           for epoch_number in range(len(labels_df))]
    epochs_numbers_list = list(chain.from_iterable(epochs_numbers_list))
    output_df['Epoch'] = epochs_numbers_list

    for index, column in enumerate(COLUMNS):
        output_df[column] = epochs_data_np_2d[index]

    output_df['Label'] = labels_list
    output_df.to_csv('preprocessed.csv', index=False)


def preprocess(epochs_mne: mne.Epochs):
    """ AutoReject -> ICA -> Baseline correct """
    #epochs_mne.set_eeg_reference(ref_channels=['AF3', 'AF4'])

    ica = mne.preprocessing.ICA(n_components=14, random_state=69, max_iter=800)
    #br = mne.set_bipolar_reference(epochs_mne, anode='AF3', cathode='AF4', ch_name='BIPOLAR_REFERENCE')
    #br.plot(block=True, scalings='auto')

    ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=69, n_jobs=1, verbose=True)
    ar.fit(epochs_mne)
    epochs_ar, reject_log = ar.transform(epochs_mne, return_log=True)

    ica.fit(epochs_mne[~reject_log.bad_epochs])

    #epochs_mne.plot(block=True, scalings=dict(eeg=150))
    eog_indices_af3, eog_scores_af3 = ica.find_bads_eog(epochs_mne, ch_name='AF3')
    eog_indices_af4, eog_scores_af4 = ica.find_bads_eog(epochs_mne, ch_name='AF4')
    ica.exclude = list(set(eog_indices_af3 + eog_indices_af4))
    print(f"ica.exclude contents: {ica.exclude}")
    ica.apply(epochs_mne, exclude=ica.exclude)
    epochs_mne_baseline = epochs_mne.apply_baseline(baseline=(None, None))
    #epochs_mne_baseline.plot(block=True, scalings=dict(eeg=150))
    return epochs_mne_baseline


if __name__ == '__main__':
    filepath = 'testing_data/thinking.csv'  # location of the file of interest
    data_mne = load_raw_data(filepath, verbose=False, data_type='feis')  # loads data from csv to mne.io.RawArray
    filter_raw(data_mne)  # notch filter and high pass
    epochs_mne = create_epochs_object(data_mne, filepath, epoch_duration=EPOCH_DURATION)
    epochs_mne = preprocess(epochs_mne)
    save_mne_epochs_to_csv(epochs_mne)

    #ica.fit(epochs_mne)
    #epochs_mne.plot(block=True, scalings='auto')
    #ica.apply(epochs_mne)
    #epochs_mne.plot(block=True, scalings='auto')
    #ica = mne.preprocessing.ICA(n_components=13, random_state=69, max_iter=800)
    #ica.fit(epochs_mne[~reject_log.bad_epochs])
    # do ica fit back onto raw data after AR


