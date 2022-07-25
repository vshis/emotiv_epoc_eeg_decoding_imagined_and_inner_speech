import labeller
from pathlib import Path


def label_my_file():
    """
    See the files in the sample_data_for_label_my_files_script folder for examples of data used
    To label your own files:
    1. Set the sequence_path variable to your .txt sequence used in timeline-generator.lua
    2. Set the input_files_path variable to the folder, to recursively search through for csv files to label
    3. Set the save_path variable to the folder, to save the labelled data in
    """
    # select path to the txt of sequence used in experiment-timeline.lua script that you want to use to label your data
    sequence_path = 'sample_data_for_label_my_files_script/sequence.txt'  # EDIT THIS
    # select directory to recursively search in for csv files to label
    input_files_path = 'sample_data_for_label_my_files_script/to_label'  # EDIT THIS
    # select directory where labelled files will be saved
    save_path = 'sample_data_for_label_my_files_script/labelled/'  # EDIT THIS

    # DO NOT edit anything below
    my_labeller = labeller.Labeller(sequences_list=[Path(sequence_path)],
                                    input_path=input_files_path,
                                    save_dir=save_path)
    my_labeller.label_csvs()


if __name__ == '__main__':
    label_my_file()
