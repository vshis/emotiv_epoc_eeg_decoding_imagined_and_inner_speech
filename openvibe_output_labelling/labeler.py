from pathlib import Path
import pandas as pd
import warnings


class Labeler:
    def __init__(self, sequence=None, openvibe_output_path=None):
        # use dictionary to match Stim ID labels to phone labels
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
                       "OVTK_StimulationId_Label_10": "thought",
                       "OVTK_StimulationId_Label_11": "gnaw",
                       "OVTK_StimulationId_Label_12": "knew",
                       "OVTK_StimulationId_Label_13": "pot",
                       "OVTK_StimulationId_Label_14": "pat",
                       "OVTK_StimulationId_Label_15": "diy",
                       "OVTK_StimulationId_Label_16": "tiy",
                       "OVTK_StimulationId_Label_17": "piy",
                       "OVTK_StimulationId_Label_18": "uw",
                       "OVTK_StimulationId_Label_19": "iy"
                       }

        if sequence is None:
            raise "Please define a sequence"
        else:
            self.sequence = sequence

        # set the path to directory for the csv files output from OpenVibe
        if openvibe_output_path is None:
            self.directory = Path("../output/")
        else:
            self.directory = Path(openvibe_output_path)
        self.dfs = []

    def label_csvs(self):
        # keep in mind that this code will read all csv files in the chosen directory
        # so make sure no other csv files exist in the chosen directory
        for file in self.directory.glob('*.csv'):
            file_df = pd.read_csv(file)  # create pandas dataframe
            # remove garbage columns, change the headers if necessary
            try:
                file_df = file_df.drop(labels=['Event Id', 'Event Date', 'Event Duration'], axis=1)
            except KeyError as e:
                warnings.warn(f"{e}. Garbage columns were not removed from dataframe.")

            # create columns for label and stage
            file_df['Label'] = ""
            file_df['Stage'] = file.stem

            # use epochs to set the labels, since the label is consistent throughout an epoch
            for epoch in range(max(file_df['Epoch']) + 1):
                file_df.loc[file_df['Epoch'] == epoch, 'Label'] = self.labels[sequence[epoch]]

            # dfs.append(file_df)  # add the completed dataframe to a list of dataframes for concatenation
            if not Path("output_labelled").is_dir():
                Path("output_labelled").mkdir()
            file_df.to_csv("output_labelled/" + file.stem + "_labelled.csv", index=False)  # save dataframe as csv

        # concatenate all csv files and output
        # full_dataframe = pd.concat(dfs)
        # full_dataframe = full_dataframe.sort_values('Epoch')
        # full_dataframe.to_csv("full_data_labelled.csv", index=False)


if __name__ == '__main__':
    # set variables with matching strings corresponding to all the stimulation id labels in OpenVibe paradigm
    OVTK_StimulationId_Label_01 = "OVTK_StimulationId_Label_01"
    OVTK_StimulationId_Label_02 = "OVTK_StimulationId_Label_02"
    OVTK_StimulationId_Label_03 = "OVTK_StimulationId_Label_03"
    OVTK_StimulationId_Label_04 = "OVTK_StimulationId_Label_04"
    OVTK_StimulationId_Label_05 = "OVTK_StimulationId_Label_05"
    OVTK_StimulationId_Label_06 = "OVTK_StimulationId_Label_06"
    OVTK_StimulationId_Label_07 = "OVTK_StimulationId_Label_07"
    OVTK_StimulationId_Label_08 = "OVTK_StimulationId_Label_08"
    OVTK_StimulationId_Label_09 = "OVTK_StimulationId_Label_09"
    OVTK_StimulationId_Label_0A = "OVTK_StimulationId_Label_0A"
    OVTK_StimulationId_Label_0B = "OVTK_StimulationId_Label_0B"
    OVTK_StimulationId_Label_0C = "OVTK_StimulationId_Label_0C"
    OVTK_StimulationId_Label_0D = "OVTK_StimulationId_Label_0D"
    OVTK_StimulationId_Label_0E = "OVTK_StimulationId_Label_0E"
    OVTK_StimulationId_Label_0F = "OVTK_StimulationId_Label_0F"
    OVTK_StimulationId_Label_10 = "OVTK_StimulationId_Label_10"
    OVTK_StimulationId_Label_11 = "OVTK_StimulationId_Label_11"
    OVTK_StimulationId_Label_12 = "OVTK_StimulationId_Label_12"
    OVTK_StimulationId_Label_13 = "OVTK_StimulationId_Label_13"
    OVTK_StimulationId_Label_14 = "OVTK_StimulationId_Label_14"
    OVTK_StimulationId_Label_15 = "OVTK_StimulationId_Label_15"
    OVTK_StimulationId_Label_16 = "OVTK_StimulationId_Label_16"
    OVTK_StimulationId_Label_17 = "OVTK_StimulationId_Label_17"
    OVTK_StimulationId_Label_18 = "OVTK_StimulationId_Label_18"
    OVTK_StimulationId_Label_19 = "OVTK_StimulationId_Label_19"

    # ensure correct sequence in the cur_sequence.txt file
    with open("cur_sequences.txt", "r") as file:
        sequence = []
        for line in file:
            line_stripped = line.strip(',\n')
            sequence.append(line_stripped)

    labeler = Labeler(sequence=sequence, openvibe_output_path="../output/")
    labeler.label_csvs()
