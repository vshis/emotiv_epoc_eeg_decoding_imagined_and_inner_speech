import numpy as np
import pandas as pd
import mne
from pathlib import Path


def load_data(filepath: str, verbose=False, data_type='feis'):
    """
    This function is configured for raw FEIS data. Configure the function for data collected from other sources.
    :param data_type: feis (256Hz) or my (128Hz) data
    :param verbose: show imported data info
    :param filepath: path to the file of interest
    :return: Raw MNE object of the data
    """
    data_types = ['feis', 'my']
    if data_type not in data_types:
        raise ValueError("Invalid data type. Expected on of: %s" % data_types)

    df = pd.read_csv(Path(filepath))

    if data_type == 'feis':  # FEIS data
        channels_df = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
        sfreq = 256  # Hz
    elif data_type == 'my':  # my data LABELED
        channels_df = df.drop(labels=['Time:128Hz', 'Epoch', 'Label', 'Stage'], axis=1)
        sfreq = 128  # Hz
    else:
        return
    channels_list = []
    for channel in channels_df.columns:
        channel_np = channels_df[channel].to_numpy()
        channels_list.append(channel_np)

    channels_np = np.array(channels_list)

    ch_names = list(channels_df.columns)
    ch_types = 'eeg'
    montage_1020 = mne.channels.make_standard_montage('standard_1020')

    info = mne.create_info(ch_names, sfreq, ch_types)  # Raw MNE objects must have associated info with them

    raw = mne.io.RawArray(channels_np, info)  # create Raw MNE object of the parsed data
    raw.set_montage(montage_1020)
    if verbose:
        print(raw.info)
    return raw


def create_epochs_object(mne_raw_object: mne.io.RawArray, filepath: str):
    events = mne.make_fixed_length_events(mne_raw_object, start=0, duration=5.0)
    epochs_mne = mne.Epochs(mne_raw_object, events)
    df = pd.read_csv(Path(filepath))
    labels = [df.loc[df['Epoch'] == epoch, 'Label'].iloc[0] for epoch in range(df['Epoch'].max()+1)]
    epochs_mne.metadata = pd.DataFrame(data=labels, columns=['Labels'], index=range(len(labels)))
    return epochs_mne


if __name__ == '__main__':
    filepath = 'testing_data/thinking.csv'
    data_mne = load_data(filepath, verbose=False, data_type='feis')
    #raw_data = data_mne.copy()
    create_epochs_object(data_mne, filepath)


    data_mne.plot(block=True, scalings='auto')
    data_mne.plot_psd()
    epochs_mne.plot(scalings='auto')
    epochs_mne.plot_psd()

    #print(mne.find_events(data_mne, shortest_event=1, initial_event=True, stim_channel='F3'))
    #ica = mne.preprocessing.ICA(n_components=14, random_state=69, max_iter=800)
    #data_mne.filter(l_freq=1., h_freq=None)
    #data_mne.notch_filter(50)
    #data_mne.plot_psd()
    #ica.fit(data_mne, picks='eeg')
    #ica.exclude = ['AF3', 'AF4']
    #ica.apply(data_mne)
    #raw_data.plot(block=True, scalings='auto')
    #raw_data.plot_psd()


