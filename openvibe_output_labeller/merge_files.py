import pandas as pd
import os
from pathlib import Path
from shutil import rmtree


def _merge_dfs(files_list: list):
    """
    :param files_list: list of pathlib.Path objects pointing to csvs to merge
    :return: dataframe of merged csv files
    """
    # read csv files, dropping all columns except Epoch and Channel #s
    dfs = [pd.read_csv(filepath).iloc[:, 1:] for filepath in files_list]
    # epoch numbers in concatenated dfs are adjusted to fit the order
    dfs[1]['Epoch'] += dfs[0]['Epoch'].iloc[-1] + 1
    dfs[2]['Epoch'] += dfs[1]['Epoch'].iloc[-1] + 1
    return pd.concat(dfs)


def search_and_merge(recordings_dir: str, save_dir: str, delete_source_dirs=False):
    """
    Merges partial recordings' csv files into a single csv file, removing Time:xxxHz column along the way
    :param delete_source_dirs: if set to True, directories containing partial data will be deleted
    :param recordings_dir: directory with all participants' recordings
    :param save_dir: directory to save the files in, preserving folder structure
    :return: None
    """
    files_paths = list(Path(recordings_dir).rglob('*.csv'))
    speech_types = ['imagined', 'inner']
    recording_types = set([file.stem for file in files_paths])  # thinking, articulators, stimuli, resting
    number_of_participants = len(next(os.walk(recordings_dir))[1])
    for participant_number in range(number_of_participants):
        for speech_type in speech_types:
            for recording_type in recording_types:
                to_merge = [file for file in files_paths if
                            f'participant_{participant_number:02d}' in str(file)
                            and speech_type in str(file)
                            and recording_type in str(file)]
                print(f"Data for participant: {participant_number}, merging: {speech_type} {recording_type}")
                merged_df = _merge_dfs(to_merge)
                save_path = f'{save_dir}/participant_{participant_number:02d}/{speech_type}'
                if not Path(save_path).is_dir():
                    Path(save_path).mkdir(parents=True)
                merged_df.to_csv(f'{save_path}/{recording_type}.csv', index=False)

            # after we are done with the recording type for a participant, we want to delete the individual files and
            # folders associated with the recording type (since we will no longer need them)
            if delete_source_dirs:
                for index in range(3):
                    rmtree(f'{recordings_dir}/participant_{participant_number:02d}/{speech_type}_0{index}')
                print(f"Deleted partial data folders for participant: {participant_number}, speech type: {speech_type}")


if __name__ == '__main__':
    recordings_directory = 'test_data'  # point to the directory with participants' folders in it
    save_directory = 'test_data'  # point to dir where to save files
    search_and_merge(recordings_dir=recordings_directory, save_dir=save_directory, delete_source_dirs=True)
