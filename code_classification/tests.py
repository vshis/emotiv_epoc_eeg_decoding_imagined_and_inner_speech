import pandas as pd
import os
from pathlib import Path
import sigfig
import math


def get_results_paths(results_dir, algorithms):
    paths = [path for path in Path(results_dir).iterdir() if path.is_dir()]
    indices_bool = [any(map(str(i).__contains__, algorithms)) for i in paths]

    results_paths = [path for index, path in enumerate(paths) if indices_bool[index]]

    return results_paths


def combine_results():
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


def print_rounded(df, df100_rounded, data_type='test'):
    print(f"Accurices (std) for data type: {data_type}\n")
    floors = {'train': 0, 'test': 2}
    for row_n in range(df100_rounded.shape[0]):
        print(f"{df.iloc[[row_n]].values[0][0]} ", end="")
        for index in range(floors[data_type], 41, 4):
            mean = df100_rounded.iloc[[row_n]].values[0][index]
            std = df100_rounded.iloc[[row_n]].values[0][index + 1]
            if not math.isnan(mean):
                mean = sigfig.round(mean, sigfigs=3)
                std = sigfig.round(std, sigfigs=3)
            print(f"& {mean} ({std:.2f}) ", end="")
        print("\\\\")


if __name__ == '__main__':
    df = pd.read_csv('classification_results/ALL_CONVENTIONAL/participant_01.csv')
    df100 = df.select_dtypes(exclude=['object']) * 100
    #df100_rounded = df100.round(2)
    df100_rounded = df100
    headers = ['raw_imagined_train_mean', 'raw_imagined_train_std',
               'raw_imagined_test_mean', 'raw_imagined_test_std',
               'preprocessed_imagined_train_mean', 'preprocessed_imagined_train_std',
               'preprocessed_imagined_test_mean', 'preprocessed_imagined_test_std',
               'linear_imagined_train_mean', 'linear_imagined_train_std',
               'linear_imagined_test_mean', 'linear_imagined_test_std',
               'features_imagined_train_mean', 'features_imagined_train_std',
               'features_imagined_test_mean', 'features_imagined_test_std',
               'mfcc_imagined_train_mean', 'mfcc_imagined_train_std',
               'mfcc_imagined_test_mean', 'mfcc_imagined_test_std',
               'raw_inner_train_mean', 'raw_inner_train_std',
               'raw_inner_test_mean', 'raw_inner_test_std',
               'preprocessed_inner_train_mean', 'preprocessed_inner_train_std',
               'preprocessed_inner_test_mean', 'preprocessed_inner_test_std',
               'linear_inner_train_mean', 'linear_inner_train_std',
               'linear_inner_test_mean', 'linear_inner_test_std',
               'features_inner_train_mean', 'features_inner_train_std',
               'features_inner_test_mean', 'features_inner_test_std',
               'mfcc_inner_train_mean', 'mfcc_inner_train_std',
               'mfcc_inner_test_mean', 'mfcc_inner_test_std']
    new_df = pd.DataFrame()
    for header in headers:
        new_df[header] = df100_rounded[header]
    # print_rounded(df, df100_rounded, data_type='train')
    print_rounded(df, new_df, data_type='test')
