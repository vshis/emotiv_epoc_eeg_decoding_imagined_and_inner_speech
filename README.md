# An investigation into decoding imagined and inner speech processes from commercial EEG devices
Repository in support of the University of Bath Computer Science MSc dissertation

Introductory video to the project can be found on this [YouTube link](https://youtu.be/1hoWxHhVOWI)

The full dissertation report can be found [here](/dissertation_report_final.pdf)

To execute data collection pipeline see the following file: 

[Click here for information on how to execute the data collection pipeline](/installers/configuration.txt)

To label all raw data in the raw_eeg_recordings directory, run the label_merge_combine/label_merge_combine.py script, ensuring the destination directory (raw_eeg_recordings_labelled by default) for the newly labelled files is empty.

Package versions used:
* Python 3.8.8
* PyTorch 1.12.1+cu116
* Pandas 1.4.2
* NumPy 1.22.2
* SciPy 1.8.0
* Scikit-learn 1.1.1
* MNE 1.0.3
* Matplotlib 3.5.1
* Antropy 0.1.4
* Hurst 0.0.5
* Librosa 0.9.2
* WandB 0.13.2
