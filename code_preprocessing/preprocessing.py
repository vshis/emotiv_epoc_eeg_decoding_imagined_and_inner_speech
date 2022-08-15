import numpy as np
import pandas as pd
import mne
from itertools import chain
from pathlib import Path

SAMPLING_FREQUENCY = 256  # hz
EPOCH_DURATION = 5.0  # seconds
COLUMNS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']


def load_raw_data(filepath: str = None, pre_loaded_df: pd.DataFrame = None, verbose=False, data_type='feis'):
    """
    Loads in data as pandas DataFrame (or directly accepts a preloaded DataFrame). Returns mne.io.RawArray object with
    10-20 montage.
    :param filepath: path to the file of interest. Default=None.
    :param pre_loaded_df: parse a pre-loaded dataframe to the function instead of a filepath. Default=None. By default,
    filepath will be checked before pre_loaded_df is.
    :param data_type: feis (256Hz) or my (128Hz) data. Default='feis'
    :param verbose: show imported data info. Default=False
    :return: Raw MNE (mne.io.RawArray) object of the data
    """
    data_types = ['feis', 'my', 'p00']
    if data_type not in data_types:
        raise ValueError("Invalid data type. Expected one of: %s" % data_types)

    if filepath is not None:
        df = pd.read_csv(Path(filepath))
    elif pre_loaded_df is not None:
        df = pre_loaded_df
    else:
        raise ValueError("Either a filepath to a csv file or a preloaded pandas dataframe is required.")

    if data_type == 'feis':  # FEIS data
        channels_df = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
    elif data_type == 'my' or data_type == 'p00':  # my data LABELLED
        channels_df = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)
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
    if data_type != 'p00':
        raw.set_montage(montage_1020)
    if verbose:
        print(raw.info)
    return raw


def create_epochs_object(mne_raw_object: mne.io.RawArray, epoch_duration: float,
                         filepath: str = None, pre_loaded_df: pd.DataFrame = None):
    """
    Filepath required again as some epochs may be dropped when creating Epochs object.
    Filepath gets checked for before preloaded df.
    """
    events = mne.make_fixed_length_events(mne_raw_object, id=1, start=-1, duration=epoch_duration)
    event_dict = {'thinking': 1}
    epochs_mne = mne.Epochs(mne_raw_object,
                            events,
                            event_id=event_dict,
                            preload=True,
                            tmin=-1, tmax=epoch_duration)

    if filepath is not None:
        df = pd.read_csv(Path(filepath))
    elif pre_loaded_df is not None:
        df = pre_loaded_df
    else:
        raise ValueError("Either a filepath to a csv file or a preloaded pandas dataframe is required.")

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
    # raw_data.notch_filter(50., trans_bandwidth=4.)


def save_mne_epochs_to_csv(epochs_mne: mne.Epochs, save_path: str):
    """Use epochs.to_data_frame instead"""
    df = epochs_mne.to_data_frame()
    df.drop(labels=['time', 'condition'], axis=1, inplace=True)
    df['epoch'] = df['epoch'] - 1
    df.rename({'epoch': 'Epoch'}, axis=1, inplace=True)
    labels_df = epochs_mne.metadata['Labels']
    labels_list = [[label] * int(SAMPLING_FREQUENCY * EPOCH_DURATION + 1) for label in labels_df]
    labels_list = list(chain.from_iterable(labels_list))
    df['Label'] = labels_list
    df.to_csv(f'{save_path}/preprocessed.csv', index=False)


def preprocess_continuous(raw_mne: mne.io.RawArray):
    ica = mne.preprocessing.ICA(n_components=14, max_iter=800, random_state=69)
    ica.fit(raw_mne)
    eog_indices_af3, eog_scores_af3 = ica.find_bads_eog(raw_mne, ch_name='AF3')
    eog_indices_af4, eog_scores_af4 = ica.find_bads_eog(raw_mne, ch_name='AF4')
    ica.exclude = list(set(eog_indices_af3 + eog_indices_af4))
    # ica.exclude = [2]  # participant 01 imagined
    # ica.exclude = [1]  # participant 01 inner
    # ica.exclude = [3]  # participant 02 imagined
    # ica.exclude = [4]  # participant 02 inner
    # ica.exclude = [4]  # participant 03 imagined
    # ica.exclude = [5]  # participant 03 inner
    # ica.exclude = [2]  # participant 04 imagined
    # ica.exclude = [1]  # participant 04 inner
    # ica.exclude = [1]  # feis
    print(f"ica.exclude contents: {ica.exclude}")
    ica.plot_components()
    ica.apply(raw_mne, exclude=ica.exclude)


def interpolate_channel(raw_mne: mne.io.RawArray, channel: str):
    """Interpolates bad channels. Used for Participant 03 data"""
    raw_mne.info['bads'].append(channel)
    raw_mne.interpolate_bads(method=dict(eeg='MNE'))


def drop_stages(raw_mne: mne.io.RawArray, filepath: str):
    """Drops all stages except thinking and returns a new mne.io.Raw object with only thinking stages present"""
    raw_df = raw_mne.to_data_frame()
    df = pd.read_csv(filepath)
    raw_df['Epoch'] = df['Epoch']
    raw_df['Label'] = df['Label']
    raw_df['Stage'] = df['Stage']
    raw_df.drop(raw_df[raw_df['Stage'] != 'thinking'].index, inplace=True)
    raw_df.drop(labels=['time'], axis=1, inplace=True)
    thinking_raw_mne = load_raw_data(pre_loaded_df=raw_df, verbose=False, data_type='my')
    return thinking_raw_mne, raw_df


if __name__ == '__main__':
    # participant 00
    # imagined
    # filepath = '../raw_eeg_recordings_labelled/participant_00/imagined/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_00/imagined/'
    # inner
    # filepath = '../raw_eeg_recordings_labelled/participant_00/inner/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_00/inner/'

    # participant 01
    # imagined
    # filepath = '../raw_eeg_recordings_labelled/participant_01/imagined/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_01/imagined/'
    # inner
    # filepath = '../raw_eeg_recordings_labelled/participant_01/inner/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_01/inner/'

    # participant 02
    # imagined
    # filepath = '../raw_eeg_recordings_labelled/participant_02/imagined/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_02/imagined/'
    # inner
    # filepath = '../raw_eeg_recordings_labelled/participant_02/inner/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_02/inner/'

    # participant 03
    # imagined
    # filepath = '../raw_eeg_recordings_labelled/participant_03/imagined/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_03/imagined/'
    # inner
    # filepath = '../raw_eeg_recordings_labelled/participant_03/inner/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_03/inner/'

    # participant 04
    # imagined
    # filepath = '../raw_eeg_recordings_labelled/participant_04/imagined/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_04/imagined/'
    # inner
    # filepath = '../raw_eeg_recordings_labelled/participant_04/inner/full_labelled.csv.zip'
    # save_path = '../data_preprocessed/participant_04/inner/'

    # feis
    filepath = '../../feis-01-full_eeg.csv'
    save_path = '../../feis_preprocessed/'

    raw_mne = load_raw_data(filepath=filepath, verbose=False, data_type='feis')  # loads data from csv to mne.io.RawArray
    raw_plot = raw_mne.plot(block=True, scalings=dict(eeg=150), show_scrollbars=False, show_scalebars=False,
                            show_options=False)
    # raw_plot.savefig('plot')

    # interpolate_channel(raw_mne=raw_mne, channel='T8')  # participant 03 imagined
    # interpolate_channel(raw_mne=raw_mne, channel='F8')  # participant 01 imagined

    filter_raw(raw_mne)  # apply high-pass filter
    preprocess_continuous(raw_mne)  # ICA

    raw_mne.plot(block=True, scalings=dict(eeg=150))  # plot raw data after ICA

    thinking_raw_mne, thinking_df = drop_stages(raw_mne=raw_mne, filepath=filepath)  # drop all stages other than think

    thinking_raw_mne.plot(block=True, scalings=dict(eeg=5000))
    epochs_mne = create_epochs_object(mne_raw_object=thinking_raw_mne,
                                      epoch_duration=EPOCH_DURATION,
                                      pre_loaded_df=thinking_df)

    epochs_mne.plot(block=True, scalings=dict(eeg=500000000))
    epochs_mne.crop(tmin=0.0, tmax=EPOCH_DURATION)  # remove the extra time before saving

    save_mne_epochs_to_csv(epochs_mne=epochs_mne, save_path=save_path)  # save as csv
    epochs_mne.save(f'{save_path}/preprocessed-epo.fif')  # save as .fif
