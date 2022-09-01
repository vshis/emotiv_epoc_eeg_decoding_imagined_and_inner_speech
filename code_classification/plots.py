import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_curve(points, plt, label: str, colour='b'):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    alpha = np.linspace(0, 1, 75)

    interpolator = interp1d(distance, points, kind='quadratic', axis=0)
    interpolated_points = interpolator(alpha)

    plt.plot(*interpolated_points.T, f'-{colour}', label=label)


def plot_for_participant():
    for participant_n in range(2, 3):
        df_p1 = pd.read_csv(f'classification_results/knn_results/participant_0{participant_n}.csv')

        # k_locs = [i**2 for i in range(0, 12)]
        # k_locs = [0, 1, 3, 9, 25, 75, 100, 115]
        # ks = [df['k'].iloc[k] for k in k_locs]
        # ks.append(df['k'].max())
        ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]

        # accuracies_test = [df.loc[df['k'] == k, 'Test_mean'].item() for k in ks]
        # accuracies_train = [df.loc[df['k'] == k, 'Train_mean'].item() for k in ks]

        # raw
        accuracies_train = [df_p1.loc[df_p1['k'] == k, 'raw_imagined_train'].item() for k in ks]
        accuracies_test = [df_p1.loc[df_p1['k'] == k, 'raw_imagined_test'].item() for k in ks]

        points_test = np.vstack((ks, accuracies_test)).T
        points_train = np.vstack((ks, accuracies_train)).T

        # plot_curve(points_train, plt, 'raw_imagined_train')
        plot_curve(points_test,
                   plt,
                   f'Raw',
                   colour='m'
                   )

        # prep
        accuracies_train = [df_p1.loc[df_p1['k'] == k, 'preprocessed_imagined_train'].item() for k in ks]
        accuracies_test = [df_p1.loc[df_p1['k'] == k, 'preprocessed_imagined_test'].item() for k in ks]

        points_test = np.vstack((ks, accuracies_test)).T
        points_train = np.vstack((ks, accuracies_train)).T

        # plot_curve(points_train, plt, 'preprocessed_imagined_train')
        plot_curve(points_test,
                   plt,
                   f'Preprocessed',
                   colour='b'
                   )

        # features
        accuracies_train = [df_p1.loc[df_p1['k'] == k, 'features_imagined_train'].item() for k in ks]
        # accuracies_test = [df_p1.loc[df_p1['k'] == k, 'features_imagined_test'].item() for k in ks]
        accuracies_test = [0.077244259, 0.073068894, 0.070981211, 0.068893528, 0.064718163, 0.056367432, 0.045929019,
                           0.044732789,
                           0.043841336]
        ks = [1, 5, 13, 37, 101, 445, 845, 1001, 1115]
        points_test = np.vstack((ks, accuracies_test)).T
        points_train = np.vstack((ks, accuracies_train)).T

        # plot_curve(points_train, plt, 'preprocessed_imagined_train')
        plot_curve(points_test,
                   plt,
                   f'Features',
                   colour='r'
                   )

        # plt.plot(*points.T, 'ok', label='original points')
        # plt.plot(df['k'], df['Test_mean'], label='original_test')
        # plt.plot(df['k'], df['Train_mean'], label='original_train')


def plot_for_binary():
    # binary
    df_p1 = pd.read_csv(f'classification_results/knn_results/binary.csv')

    # k_locs = [i**2 for i in range(0, 12)]
    # k_locs = [0, 1, 3, 9, 25, 75, 100, 115]
    # ks = [df['k'].iloc[k] for k in k_locs]
    # ks.append(df['k'].max())
    ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]

    # accuracies_test = [df.loc[df['k'] == k, 'Test_mean'].item() for k in ks]
    # accuracies_train = [df.loc[df['k'] == k, 'Train_mean'].item() for k in ks]

    # raw
    accuracies_train = [df_p1.loc[df_p1['k'] == k, 'raw_train'].item() for k in ks]
    accuracies_test = [df_p1.loc[df_p1['k'] == k, 'raw_test'].item() for k in ks]
    ks = [1, 5, 13, 37, 101, 445, 845, 1115, 1599]

    points_test = np.vstack((ks, accuracies_test)).T
    points_train = np.vstack((ks, accuracies_train)).T

    # plot_curve(points_train, plt, 'raw_imagined_train')
    plot_curve(points_test,
               plt,
               f'Raw',
               colour='m'
               )

    # prep
    ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]
    accuracies_train = [df_p1.loc[df_p1['k'] == k, 'preprocessed_train'].item() for k in ks]
    accuracies_test = [df_p1.loc[df_p1['k'] == k, 'preprocessed_test'].item() for k in ks]
    ks = [1, 5, 13, 37, 101, 445, 845, 1115, 1599]
    points_test = np.vstack((ks, accuracies_test)).T
    points_train = np.vstack((ks, accuracies_train)).T

    # plot_curve(points_train, plt, 'preprocessed_imagined_train')
    plot_curve(points_test,
               plt,
               f'Preprocessed',
               colour='b'
               )

    # features
    ks = [1, 5, 13, 37, 101, 445, 845, 1539, 3939]
    accuracies_train = [df_p1.loc[df_p1['k'] == k, 'features_train'].item() for k in ks]
    # accuracies_test = [df_p1.loc[df_p1['k'] == k, 'features_test'].item() for k in ks]
    accuracies_test = [0.512, 0.511, 0.503, 0.499, 0.491, 0.490, 0.487, 0.476, 0.471]
    ks = [1, 5, 13, 25, 37, 53, 75, 101, 139]
    points_test = np.vstack((ks, accuracies_test)).T
    points_train = np.vstack((ks, accuracies_train)).T

    # plot_curve(points_train, plt, 'preprocessed_imagined_train')
    plot_curve(points_test, plt, f'Features', colour='r')
    # plt.plot(*points_test.T)
    # plt.plot(*points.T, 'ok', label='original points')
    # plt.plot(df['k'], df['Test_mean'], label='original_test')
    # plt.plot(df['k'], df['Train_mean'], label='original_train')


def trend_plots():
    plt.figure()

    plot_for_participant()
    # plot_for_binary()

    plt.rcParams.update({'font.size': 12})
    plt.xlabel("k", size=12)
    plt.ylabel("Accuracy", size=12)
    plt.yticks([i / 100 for i in range(0, 101, 10)], size=12)
    plt.xticks(size=12)
    plt.grid(axis='y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_types = ['Raw', 'Preprocessed', 'LF', 'NLF', 'MFCC']
    x_pos = np.arange(len(data_types))
    # binary
    # means = [90.0, 90.0, 100, 99.0, 100]
    # stds = [12.3, 12.3, 0.00, 2.00, 0.00]
    bar_w = 0.1
    # p01
    # im
    im_mean_p01 = [7.50, 5.00, 6.25, 6.62, 6.38]
    im_stdd_p01 = [3.75, 1.53, 1.48, 1.75, 1.08]
    # in
    in_mean_p01 = [6.25, 6.25, 9.00, 7.25, 8.00]
    in_stdd_p01 = [6.56, 3.42, 2.46, 2.08, 1.65]
    # p02
    # im
    im_mean_p02 = [10.6, 6.88, 7.12, 5.75, 6.38]
    im_stdd_p02 = [4.24, 4.15, 2.58, 2.07, 1.83]
    # in
    in_mean_p02 = [8.75, 6.25, 5.75, 6.50, 7.00]
    in_stdd_p02 = [8.48, 5.23, 1.21, 2.29, 1.65]
    # p03
    # im
    im_mean_p03 = [8.12, 6.25, 7.75, 7.25, 8.00]
    im_stdd_p03 = [3.19, 1.98, 0.50, 1.46, 0.73]
    # in
    in_mean_p03 = [8.12, 8.75, 8.00, 7.25, 6.25]
    in_stdd_p03 = [3.19, 6.06, 1.33, 1.46, 1.19]
    # p04
    # im
    im_mean_p04 = [7.50, 8.12, 6.88, 5.50, 7.62]
    im_stdd_p04 = [3.19, 5.08, 1.19, 2.28, 2.22]
    # in
    in_mean_p04 = [6.88, 8.75, 7.62, 6.12, 5.12]
    in_stdd_p04 = [3.64, 3.64, 1.91, 1.87, 0.73]
    # FEIS
    mean_feis = [8.75, 7.50, 9.50, 5.00, 7.00]
    stdd_feis = [5.00, 4.68, 2.69, 0.79, 2.81]

    means = [im_mean_p01, in_mean_p01, im_mean_p02, in_mean_p02, im_mean_p03, in_mean_p03, im_mean_p04, in_mean_p04, mean_feis]
    stds = [im_stdd_p01, in_stdd_p01, im_stdd_p02, in_stdd_p02, im_stdd_p03, in_stdd_p03, im_stdd_p04, in_stdd_p04, stdd_feis]
    #means = [mean_feis, im_mean_p01, im_mean_p02, im_mean_p03, im_mean_p04, in_mean_p01, in_mean_p02, in_mean_p03,in_mean_p04]
    #stds = [stdd_feis, im_stdd_p01, im_stdd_p02, im_stdd_p03, im_stdd_p04, in_stdd_p01, in_stdd_p02, in_stdd_p03, in_stdd_p04]

    br1 = np.arange(len(im_mean_p01))
    br2 = [x + bar_w for x in br1]
    br3 = [x + bar_w for x in br2]
    br4 = [x + bar_w for x in br3]
    br5 = [x + bar_w for x in br4]
    br6 = [x + bar_w for x in br5]
    br7 = [x + bar_w for x in br6]
    br8 = [x + bar_w for x in br7]
    br9 = [x + bar_w for x in br8]

    brs = [br1, br2, br3, br4, br5, br6, br7, br8, br9]
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
               'tab:cyan']
    labels = ['P01 imagined',
              'P01 inner',
              'P02 imagined',
              'P02 inner',
              'P03 imagined',
              'P03 inner',
              'P04 imagined',
              'P04 inner',
              'FEIS'
              ]
    """labels = ['FEIS',
              'P01 imagined',
              'P02 imagined',
              'P03 imagined',
              'P04 imagined',
              'P01 inner',
              'P02 inner',
              'P03 inner',              
              'P04 inner',
              ]"""

    fig, ax = plt.subplots()

    for index, item in enumerate(means):
        ax.bar(brs[index], means[index], yerr=stds[index], color=colours[index], width=bar_w, edgecolor='grey',
               label=labels[index])
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks([r + bar_w*4 for r in range(len(in_mean_p01))], data_types)
    # ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    plt.axhline(4, ls='--', c='grey', label='Chance level', lw=1)
    plt.legend()
    plt.show()

    """ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    plt.show()"""

    exit()
