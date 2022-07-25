import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_figs(directory):
    for file in directory.glob('thinking.csv'):
        file_df = pd.read_csv(file)
        # channels_df = file_df.drop(labels=['Time:128Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
        channels_df = file_df.drop(labels=['Epoch', 'Event Id', 'Event Date', 'Event Duration'], axis=1)
        # print(channels_df.iloc[:, 0])
        fig, axs = plt.subplots(4, 4)
        for index, column in enumerate(channels_df.columns):
            if index == 0:
                continue
            #axs.flat[index - 1].plot(channels_df.iloc[:, 0], channels_df.iloc[:, index])
            axs.flat[index - 1].plot(channels_df.iloc[:, index])
            axs.flat[index - 1].set_title(column)
        plt.show()


directory = Path("../openvibe_output")
plot_figs(directory)
"""
for file in directory.glob('thinking.csv'):
    file_df = pd.read_csv(file)
    ch1 = file_df.loc[:, 'Channel 1']
    print(len(ch1))
    print(len(set(ch1)))

"""
