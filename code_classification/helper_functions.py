import pandas as pd
from pathlib import Path
import sigfig
import math

import warnings
warnings.filterwarnings("ignore")

NAMES = {
    "AdaBoostClassifier()": "AB (e=50, lr=1.0)",
    "AdaBoostClassifier(learning_rate=0.1)": "AB (e=50, lr=0.1)",
    "AdaBoostClassifier(learning_rate=0.5)": "AB (e=50, lr=0.5)",
    "AdaBoostClassifier(learning_rate=1.5)": "AB (e=50, lr=1.5)",
    "AdaBoostClassifier(learning_rate=2.0)": "AB (e=50, lr=2.0)",
    "AdaBoostClassifier(n_estimators=100)": "AB (e=100, lr=1.0)",
    "AdaBoostClassifier(n_estimators=200)": "AB (e=200, lr=1.0)",
    "AdaBoostClassifier(n_estimators=400)": "AB (e=400, lr=1.0)",
    "GaussianNB()": "Gaussian NB",
    "KNeighborsClassifier(n_jobs=-2)": "KNN (k=5)",
    "KNeighborsClassifier(n_jobs=-2, n_neighbors=1)": "KNN (k=1)",
    "KNeighborsClassifier(n_jobs=-2, n_neighbors=13)": "KNN (k=13)",
    "KNeighborsClassifier(n_jobs=-2, n_neighbors=139)": "KNN (k=139)",
    "KNeighborsClassifier(n_jobs=-2, n_neighbors=25)": "KNN (k=25)",
    "KNeighborsClassifier(n_jobs=-2, n_neighbors=53)": "KNN (k=53)",
    "KNeighborsClassifier(n_jobs=-2, n_neighbors=89)": "KNN (k=89)",
    "LinearDiscriminantAnalysis()": "LDA",
    "SVC(cache_size=2000)": "SVM (RBF)",
    "SVC(cache_size=2000, degree=2, kernel='poly')": "SVM (polynomial, degree=2)",
    "SVC(cache_size=2000, degree=4, kernel='poly')": "SVM (polynomial, degree=4)",
    "SVC(cache_size=2000, kernel='poly')": "SVM (polynomial)",
    "SVC(cache_size=2000, kernel='sigmoid')": "SVM (sigmoid)",
}


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
        # print(f"{df.iloc[[row_n]].values[0][0]} ", end="")
        print(f"{list(NAMES.values())[row_n]} ", end="")
        for index in range(floors[data_type], 40, 4):  # p01-04
        #for index in range(floors[data_type], 20, 4):  # binary
        #for index in range(floors[data_type], 8, 4):  # p00
            mean = df100_rounded.iloc[[row_n]].values[0][index]
            std = df100_rounded.iloc[[row_n]].values[0][index + 1]
            # if not math.isnan(mean):
            if math.isnan(mean):
                print(f"& n/a ", end="")
            else:
                mean = sigfig.round(mean, sigfigs=3)
                std = sigfig.round(std, sigfigs=3)
                if mean == 100.0:
                    print(f"& {mean:.0f} ({std:.2f}) ", end="")
                else:
                    print(f"& {mean} ({std:.2f}) ", end="")
        print("\\\\")


def print_conventional_results():
    headers_p00 = ['raw_imagined_train_mean', 'raw_imagined_train_std',
                   'raw_imagined_test_mean', 'raw_imagined_test_std',
                   'raw_inner_train_mean', 'raw_inner_train_std',
                   'raw_inner_test_mean', 'raw_inner_test_std']

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

    headers_binary = [
        'raw_train_mean', 'raw_train_std',
        'raw_test_mean', 'raw_test_std',
        'preprocessed_train_mean', 'preprocessed_train_std',
        'preprocessed_test_mean', 'preprocessed_test_std',
        'linear_train_mean', 'linear_train_std',
        'linear_test_mean', 'linear_test_std',
        'features_train_mean', 'features_train_std',
        'features_test_mean', 'features_test_std',
        'mfcc_train_mean', 'mfcc_train_std',
        'mfcc_test_mean', 'mfcc_test_std'
    ]

    df = pd.read_csv('classification_results/ALL_CONVENTIONAL/participant_04.csv')
    df100 = df.select_dtypes(exclude=['object']) * 100
    # df100_rounded = df100.round(2)
    df100_rounded = df100
    new_df = pd.DataFrame()

    # p00
    # for header in headers_p00:
    #    new_df[header] = df100_rounded[header]
    # print_rounded(df, df100_rounded, data_type='test')

    # binary & feis
    #for header in headers_binary:
    #    new_df[header] = df100_rounded[header]
    #print_rounded(df, new_df, data_type='train')
    #print()
    #print_rounded(df, new_df, data_type='test')

    # p01-04
    for header in headers:
        new_df[header] = df100_rounded[header]
    print_rounded(df, new_df, data_type='train')
    print()
    print_rounded(df, new_df, data_type='test')


def print_eegnet_results():
    files = [path for path in Path('classification_results/eegnet').iterdir()]
    for file in files:
        print(file.name)
        values = pd.read_csv(file).values[0]
        for index in range(0, len(values), 2):
            print(f"{values[index]:.2f} ({values[index+1]:.2f})")
        print()


if __name__ == '__main__':
    print_eegnet_results()


