import pandas as pd


def combine():
    """Combine separate eeg recording stages (thinking, resting, stimuli, articulators) into a continuous recoding"""
    stim = pd.read_csv('test_data/stimuli_labelled.csv')
    art = pd.read_csv('test_data/articulators_labelled.csv')
    think = pd.read_csv('test_data/thinking_labelled.csv')
    rest = pd.read_csv('test_data/resting_labelled.csv')

    full = pd.DataFrame(columns=stim.columns)

    stage_order = [stim, art, think, rest]
    number_of_epochs = stim['Epoch'].max()

    for epoch_number in range(number_of_epochs + 1):
        print(f"Combining stages of epoch {epoch_number}/{number_of_epochs}")
        for stage in stage_order:
            full = pd.concat([full, stage.loc[stage['Epoch'] == epoch_number]])

    full.to_csv('test_data/full_labelled.csv', index=False)


if __name__ == '__main__':
    combine()
