"""
Run the script to:
 1. label all raw recording files in raw_eeg_recordings directory
 2. Merge all imagined recordings for a participant into a single imagined csv per stage
 3. Merge all inner recordings for a participant into a single inner csv per stage
 4. Combine the stages for a full continuous file per participant per speech modality
 """

from pathlib import Path
from labeller import Labeller
import merge
import combine


def label_merge_combine():
    # ensure correct sequences paths are given
    sequence_imagined_0 = Path('../sequences/sequences/sequence_imagined_0.txt')
    sequence_imagined_1 = Path('../sequences/sequences/sequence_imagined_1.txt')
    sequence_imagined_2 = Path('../sequences/sequences/sequence_imagined_2.txt')
    sequence_inner_0 = Path('../sequences/sequences/sequence_inner_0.txt')
    sequence_inner_1 = Path('../sequences/sequences/sequence_inner_1.txt')
    sequence_inner_2 = Path('../sequences/sequences/sequence_inner_2.txt')
    sequence_paths = [sequence_imagined_0, sequence_imagined_1, sequence_imagined_2,
                      sequence_inner_0, sequence_inner_1, sequence_inner_2]

    labeller = Labeller(sequences_list=sequence_paths,
                        input_path="../raw_eeg_recordings/",
                        save_dir="../raw_eeg_recordings_labelled/")
    labeller.label_csvs()
    recordings_dir = "../raw_eeg_recordings_labelled/"
    save_dir = "../raw_eeg_recordings_labelled/"
    merge.search_and_merge(recordings_dir=recordings_dir, save_dir=save_dir, delete_source_dirs=True)
    combine.search_and_combine(recordings_dir=recordings_dir, save_dir=save_dir)


if __name__ == '__main__':
    label_merge_combine()
