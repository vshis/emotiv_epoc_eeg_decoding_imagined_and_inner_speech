from pathlib import Path
import pandas as pd
import warnings


class Labeller:
    def __init__(self, sequences_list=None, input_path=None, save_dir=None):
        """
        :param sequences_list: List of sequence pathlib Paths
        :param input_path: string path to the folder which needs to be recursively searched for csv files to label
        :param save_dir: string path to the folder where to save the labelled data
        """
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

        if sequences_list is None:
            raise ValueError("Please provide the sequences files.")
        else:
            self.sequences_dirs = sequences_list
        print(f"Using sequences {self.sequences_dirs}")

        # set the path to directory for the csv files output from OpenVibe
        if input_path is None:
            raise ValueError("Please provide directory which you want to search for csv files to be labelled.")
        else:
            self.directory = Path(input_path)
        print(f"Labelling files in directory {self.directory} and its child directories")

        if save_dir is None:
            raise ValueError("Please provide directory for saving labelled csv files.")
        else:
            self.save_dir = save_dir
        print(f"Saving files to directory {self.save_dir} \n")

    def label_csvs(self):
        # keep in mind that this code will read all csv files in the chosen directory and its children
        # so make sure no other csv files exist in the chosen directory
        only_one_sequence_file = False
        sequences_dict = self.open_sequence_files()
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

            # choose sequence based on the name of the parent directory of the current csv file
            if 'imagined' in str(file.parent):
                index = int(str(file.parent)[-1])
                sequence = sequences_dict[f'sequence_imagined_{index}']
            elif 'inner' in str(file.parent):
                index = int(str(file.parent)[-1])
                sequence = sequences_dict[f'sequence_inner_{index}']
            else:
                print("Filepath does not contain 'imagined' or 'inner'. "
                      "Assuming that only a single sequence file is being used to label. "
                      "Using first sequence in dictionary.\n")
                only_one_sequence_file = True
                sequence = list(sequences_dict.values())[0]

            print(f"Labelling file {file_number}: {file}")
            # print(f"Using sequence: {sequence}")

            # use epochs to set the labels, since the label is consistent throughout an epoch
            for epoch in range(max(file_df['Epoch']) + 1):
                file_df.loc[file_df['Epoch'] == epoch, 'Label'] = self.labels[sequence[epoch]]

            # save labelled files in the chosen save path
            save_path = Path(self.save_dir + str(file.parent).strip('..\\raw_eeg_recordings'))
            if only_one_sequence_file:
                save_path = Path(self.save_dir + str(file.parent))

            if not save_path.is_dir():
                save_path.mkdir(parents=True)
            file_df.to_csv(str(save_path) + '/' + file.stem + "_labelled.csv", index=False)  # save dataframe as csv
            print(f"File number {file_number} saved in: {save_path}\n")
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


def label_my_file():
    """
    See the files in the sample_data_for_label_my_files_script folder for examples of data used
    To label your own files:
    1. Set the sequence_path variable to your .txt sequence used in timeline-generator.lua
    2. Set the input_files_path variable to the folder, to recursively search through for csv files to label
    3. Set the save_path variable to the folder, to save the labelled data in
    """
    # select path to the txt of sequence used in experiment-timeline.lua script that you want to use to label your data
    sequence_path = 'test_data'  # EDIT THIS
    # select directory to recursively search in for csv files to label
    input_files_path = 'test_data'  # EDIT THIS
    # select directory where labelled files will be saved
    save_path = 'test_data'  # EDIT THIS

    # DO NOT edit anything below
    my_labeller = Labeller(sequences_list=[Path(sequence_path)],
                           input_path=input_files_path,
                           save_dir=save_path)
    my_labeller.label_csvs()


if __name__ == '__main__':
    label_my_file()
