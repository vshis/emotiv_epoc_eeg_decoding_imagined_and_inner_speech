import warnings
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

directory = Path("")

for file in directory.glob('*.csv'):
    file_df = pd.read_csv(file)
    try:
        channels_df = file_df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)
    except KeyError as e:
        raise f"Columns were not removed from {file}"
    plt.plot(file_df.iloc[:, 0], channels_df.iloc[:, 0])
    plt.show()
