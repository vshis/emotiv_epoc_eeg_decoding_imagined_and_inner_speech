import os
import pandas as pd
from pathlib import Path


def _combine(stim_path, art_path, think_path, rest_path):
    """Combine separate eeg recording stages (thinking, resting, stimuli, articulators) into a continuous recoding"""
    stim = pd.read_csv(stim_path)
    art = pd.read_csv(art_path)
    think = pd.read_csv(think_path)
    rest = pd.read_csv(rest_path)

    full = pd.DataFrame(columns=stim.columns)

    stage_order = [stim, art, think, rest]
    number_of_epochs = stim['Epoch'].max()

    for epoch_number in range(number_of_epochs + 1):
        print(f"Combining stages of epoch {epoch_number}/{number_of_epochs}")
        for stage in stage_order:
            full = pd.concat([full, stage.loc[stage['Epoch'] == epoch_number]])

    return full


def search_and_combine(recordings_dir: str, save_dir: str):
    files_paths = list(Path(recordings_dir).rglob('*.csv'))
    speech_types = ['imagined', 'inner']
    number_of_participants = len(next(os.walk(recordings_dir))[1])
    for participant_number in range(number_of_participants):
        for speech_type in speech_types:
            to_combine = [file for file in files_paths if
                          f'participant_{participant_number:02d}' in str(file)
                          and speech_type in str(file)]
            stim_path = [path for path in to_combine if 'stimuli' in path.stem][0]
            art_path = [path for path in to_combine if 'articulators' in path.stem][0]
            think_path = [path for path in to_combine if 'thinking' in path.stem][0]
            rest_path = [path for path in to_combine if 'resting' in path.stem][0]

            full = _combine(stim_path, art_path, think_path, rest_path)

            save_path = f'{save_dir}/participant_{participant_number:02d}/{speech_type}'
            if not Path(save_path).is_dir():
                Path(save_path).mkdir(parents=True)
            full.to_csv(f'{save_path}/full_labelled.csv.zip', index=False, compression='zip')


if __name__ == '__main__':
    save_dir = 'test_data/full_labelled.csv.zip'

    stim = 'test_data/stimuli_labelled.csv'
    art = 'test_data/articulators_labelled.csv'
    think = 'test_data/thinking_labelled.csv'
    rest = 'test_data/resting_labelled.csv'

    full = _combine(stim, art, think, rest)
    full.to_csv(save_dir, index=False, compression='zip')
