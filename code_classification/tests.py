import pandas as pd
import os
from pathlib import Path


def get_results_paths(results_dir, algorithms):
    paths = [path for path in Path(results_dir).iterdir() if path.is_dir()]
    indices_bool = [any(map(str(i).__contains__, algorithms)) for i in paths]

    results_paths = [path for index, path in enumerate(paths) if indices_bool[index]]

    return results_paths


if __name__ == '__main__':
    results_dir = 'classification_results/'
    algorithms = ['LinearDiscriminantAnalysis', 'GaussianNB', 'SVC', 'AdaBoostClassifier', 'KNeighborsClassifier']
    results_paths = get_results_paths(results_dir, algorithms)
    participant_ns = [i for i in range(1, 5)]

    for participant_n in participant_ns:
        dfs = []
        for results_path in results_paths:
            filepath = f'{str(results_path)}/participant_0{participant_n}.csv'
            df = pd.read_csv(filepath)
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(f'classification_results/ALL_CONVENTIONAL/participant_0{participant_n}.csv', index=False)

    for data_type in ['binary', 'feis']:
        dfs = []
        for results_path in results_paths:
            filepath = f'{str(results_path)}/{data_type}.csv'
            df = pd.read_csv(filepath)
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(f'classification_results/ALL_CONVENTIONAL/{data_type}.csv', index=False)

    # p00
    algorithms.remove('SVC')
    results_paths = get_results_paths(results_dir, algorithms)
    dfs = []
    for results_path in results_paths:
        filepath = f'{str(results_path)}/participant_00.csv'
        df = pd.read_csv(filepath)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(f'classification_results/ALL_CONVENTIONAL/participant_00.csv', index=False)
