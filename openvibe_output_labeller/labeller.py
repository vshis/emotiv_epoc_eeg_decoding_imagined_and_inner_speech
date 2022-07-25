from pathlib import Path
import pandas as pd
import warnings


class Labeller:
    def __init__(self, sequences=None, openvibe_output_path=None):
        # use dictionary to match OpenVibe's Stimulation ID labels to phone labels
        self.labels = {"OVTK_StimulationId_Label_01": "p",
                       "OVTK_StimulationId_Label_02": "t",
                       "OVTK_StimulationId_Label_03": "k",
                       "OVTK_StimulationId_Label_04": "f",
                       "OVTK_StimulationId_Label_05": "s",
                       "OVTK_StimulationId_Label_06": "sh",
                       "OVTK_StimulationId_Label_07": "v",
                       "OVTK_StimulationId_Label_08": "z",
                       "OVTK_StimulationId_Label_09": "zh",
                       "OVTK_StimulationId_Label_0A": "m",
                       "OVTK_StimulationId_Label_0B": "n",
                       "OVTK_StimulationId_Label_0C": "ng",
                       "OVTK_StimulationId_Label_0D": "fleece",
                       "OVTK_StimulationId_Label_0E": "goose",
                       "OVTK_StimulationId_Label_0F": "trap",
                       "OVTK_StimulationId_Label_10": "thought"
                       }

        if sequences is None:
            raise ValueError("Please provide the sequences files.")
        else:
            self.sequences_dirs = sequences
        print(f"Using sequences {self.sequences_dirs}")

        # set the path to directory for the csv files output from OpenVibe
        if openvibe_output_path is None:
            self.directory = Path("../output/")
        else:
            self.directory = Path(openvibe_output_path)
        print(f"Labelling files in directory {self.directory} and its child directories")

    def label_csvs(self):
        # keep in mind that this code will read all csv files in the chosen directory and its children
        # so make sure no other csv files exist in the chosen directory
        file_number = 0
        for file in self.directory.rglob('*.csv'):
            file_df = pd.read_csv(file)  # create pandas dataframe
            # remove garbage columns, change the headers if necessary
            try:
                file_df = file_df.drop(labels=['Event Id', 'Event Date', 'Event Duration'], axis=1)
            except KeyError as e:
                warnings.warn(f"{e}. Garbage columns were not removed from dataframe.")

            # create columns for label and stage
            file_df['Label'] = ""
            file_df['Stage'] = file.stem

            sequences_dict = self.open_sequence_files()

            # choose sequence based on the name of the parent directory of the current csv file
            if 'imagined' in str(file.parent):
                index = int(str(file.parent)[-1])
                sequence = sequences_dict[f'sequence_imagined_{index}']
            elif 'inner' in str(file.parent):
                index = int(str(file.parent)[-1])
                sequence = sequences_dict[f'sequence_inner_{index}']
            else:
                raise ValueError(f"Unidentified file path {file.parent}. "
                                 f"Does not contain reference to imagined or inner.")

            print(f"Labelling file {file_number}: {file}")
            print(f"Using sequence: {sequence}")

            # use epochs to set the labels, since the label is consistent throughout an epoch
            for epoch in range(max(file_df['Epoch']) + 1):
                file_df.loc[file_df['Epoch'] == epoch, 'Label'] = self.labels[sequence[epoch]]

            # save labelled files in the chosen save path
            save_path = Path("../raw_eeg_recordings_labelled/" + str(file.parent).strip('..\\raw_eeg_recordings'))
            if not save_path.is_dir():
                save_path.mkdir(parents=True)
            file_df.to_csv(str(save_path) + '/' + file.stem + "_labelled.csv", index=False)  # save dataframe as csv
            print(f"File number {file_number} saved in: {save_path}")
            file_number += 1

    def open_sequence_files(self):
        """
        Reads the sequence files specified in the self.sequences_dirs
        :return: dictionary of sequence file names as keys and associated contents as values
        """
        sequences_dict = {}
        for sequence_dir in self.sequences_dirs:
            with open(sequence_dir, "r") as file:
                sequence = []
                for line in file:
                    line_stripped = line.strip(',\n')
                    sequence.append(line_stripped)
            sequences_dict[sequence_dir.stem] = sequence
        return sequences_dict


if __name__ == '__main__':
    # ensure correct sequences paths are given
    sequence_imagined_0 = Path('../sequences/sequences/sequence_imagined_0.txt')
    sequence_imagined_1 = Path('../sequences/sequences/sequence_imagined_1.txt')
    sequence_imagined_2 = Path('../sequences/sequences/sequence_imagined_2.txt')
    sequence_inner_0 = Path('../sequences/sequences/sequence_inner_0.txt')
    sequence_inner_1 = Path('../sequences/sequences/sequence_inner_1.txt')
    sequence_inner_2 = Path('../sequences/sequences/sequence_inner_2.txt')
    sequence_paths = [sequence_imagined_0, sequence_imagined_1, sequence_imagined_2,
                      sequence_inner_0, sequence_inner_1, sequence_inner_2]

    labeler = Labeller(sequences=sequence_paths, openvibe_output_path="../raw_eeg_recordings/")
    labeler.label_csvs()
