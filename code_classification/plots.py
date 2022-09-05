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


def bar_plots_all_bars_and_mean_of_data_types():
    data_types = ['Raw', 'Preprocessed', 'LF', 'NLF', 'MFCC']
    x_pos = np.arange(len(data_types))
    # binary
    # means = [90.0, 90.0, 100, 99.0, 100]
    # stds = [12.3, 12.3, 0.00, 2.00, 0.00]
    bar_w = 0.1

    # EEGNET
    print(":::::::::::::::::::::::: EEGNet ::::::::::::::::::::::::\n")
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

    """
    # **************** ADAPTIVE BOOSTING e=200, lr=1.0 **********************
    # p01
    # im
    im_mean_p01 = [17.5, 13.9, 7.4, 7.52, 6.39]
    im_stdd_p01 = [0.163, 0.326, 0.628, 1.25, 0.767]
    # in
    in_mean_p01 = [16.7, 12.4, 9.28, 7.59, 7.27]
    in_stdd_p01 = [0.155, 0.0867, 0.713, 1.5, 0.54]
    # p02
    # im
    im_mean_p02 = [16.0, 12.1, 8.65, 9.09, 6.4]
    im_stdd_p02 = [0.313, 0.197, 1.08, 1.17, 1.08]
    # in
    in_mean_p02 = [16.5, 12.0, 6.65, 7.4, 6.14]
    in_stdd_p02 = [0.126, 0.22, 0.635, 0.615, 0.349]
    # p03
    # im
    im_mean_p03 = [16.7, 15.5, 9.28, 8.53, 6.65]
    im_stdd_p03 = [0.24, 0.189, 1.13, 0.853, 1.1]
    # in
    in_mean_p03 = [16.4, 14.0, 7.9, 8.46, 6.52]
    in_stdd_p03 = [0.333, 0.0842, 0.817, 0.399, 0.579]
    # p04
    # im
    im_mean_p04 = [15.9, 13.2, 8.46, 6.9, 7.02]
    im_stdd_p04 = [0.206, 0.17, 0.817, 0.779, 0.386]
    # in
    in_mean_p04 = [15.1, 12.3, 7.46, 6.9, 6.08]
    in_stdd_p04 = [0.224, 0.184, 0.933, 1.19, 1.46]
    # FEIS
    mean_feis = [17.6, 14.7, 8.43, 7.17, 5.41]
    stdd_feis = [0.271, 0.121, 2.0, 2.02, 1.46]
    """

    """
    # **************** GAUSSIAN NAIVE BAYES **********************
    # p01
    # im
    im_mean_p01 = [9.62, 9.16, 6.96, 7.02, 7.27]
    im_stdd_p01 = [0.0817, 0.0921, 1.49, 0.983, 0.779]
    # in
    in_mean_p01 = [10.9, 8.35, 6.96, 6.77, 6.52]
    in_stdd_p01 = [0.118, 0.063, 2.12, 1.53, 0.316]
    # p02
    # im
    im_mean_p02 = [9.8, 8.79, 8.09, 7.65, 9.34]
    im_stdd_p02 = [0.194, 0.0705, 0.666, 0.614, 1.69]
    # in
    in_mean_p02 = [9.5, 8.08, 7.59, 7.02, 6.39]
    in_stdd_p02 = [0.11, 0.0974, 0.687, 1.29, 0.921]
    # p03
    # im
    im_mean_p03 = [10.3, 9.29, 7.65, 6.9, 7.52]
    im_stdd_p03 = [0.133, 0.0494, 1.17, 1.14, 1.67]
    # in
    in_mean_p03 = [10.3, 9.37, 7.21, 7.71, 7.34]
    in_stdd_p03 = [0.155, 0.171, 1.24, 0.645, 0.938]
    # p04
    # im
    im_mean_p04 = [9.74, 8.76, 7.34, 7.46, 7.46]
    im_stdd_p04 = [0.0933, 0.107, 0.413, 1.34, 1.17]
    # in
    in_mean_p04 = [9.33, 8.87, 7.84, 7.34, 5.83]
    in_stdd_p04 = [0.108, 0.041, 1.15, 1.62, 0.767]
    # FEIS
    mean_feis = [12.8, 11.5, 7.92, 8.43, 6.04]
    stdd_feis = [0.172, 0.162, 1.41, 1.3, 1.11]
    """
    """
    # **************** LINEAR DISCRIMINANT ANALYSIS **********************
    # p01
    # im
    im_mean_p01 = [12.2, 11.2, 11.0, 10.3, 7.15]
    im_stdd_p01 = [0.134, 0.0747, 0.548, 1.97, 0.455]
    # in
    in_mean_p01 = [13.6, 10.5, 9.78, 9.15, 8.15]
    in_stdd_p01 = [0.132, 0.112, 1.71, 1.48, 0.385]
    # p02
    # im
    im_mean_p02 = [11.5, 9.66, 10.6, 8.15, 8.02]
    im_stdd_p02 = [0.146, 0.184, 0.948, 1.42, 0.767]
    # in
    in_mean_p02 = [11.5, 9.4, 8.4, 8.09, 6.77]
    in_stdd_p02 = [0.146, 0.11, 1.03, 1.39, 1.17]
    # p03
    # im
    im_mean_p03 = [13.0, 11.3, 12.2, 10.5, 7.59]
    im_stdd_p03 = [0.0857, 0.159, 0.154, 1.23, 0.233]
    # in
    in_mean_p03 = [13.2, 10.7, 11.0, 10.0, 7.71]
    in_stdd_p03 = [0.0958, 0.12, 0.728, 2.02, 0.4]
    # p04
    # im
    im_mean_p04 = [12.4, 10.7, 10.6, 8.78, 6.21]
    im_stdd_p04 = [0.104, 0.0991, 1.16, 1.44, 1.2]
    # in
    in_mean_p04 = [11.1, 10.6, 8.96, 8.28, 6.77]
    in_stdd_p04 = [0.184, 0.0969, 0.75, 1.23, 0.154]
    # FEIS
    mean_feis = [13.0, 12.4, 16.2, 11.8, 6.92]
    stdd_feis = [0.216, 0.228, 1.41, 1.92, 1.25]
    """
    means = [im_mean_p01, in_mean_p01, im_mean_p02, in_mean_p02, im_mean_p03, in_mean_p03, im_mean_p04, in_mean_p04, mean_feis]
    stds = [im_stdd_p01, in_stdd_p01, im_stdd_p02, in_stdd_p02, im_stdd_p03, in_stdd_p03, im_stdd_p04, in_stdd_p04, stdd_feis]
    #means = [im_mean_p01, in_mean_p01, im_mean_p02, in_mean_p02, im_mean_p03, in_mean_p03, im_mean_p04, in_mean_p04]
    #stds = [im_stdd_p01, in_stdd_p01, im_stdd_p02, in_stdd_p02, im_stdd_p03, in_stdd_p03, im_stdd_p04, in_stdd_p04]
    # means = [mean_feis, im_mean_p01, im_mean_p02, im_mean_p03, im_mean_p04, in_mean_p01, in_mean_p02, in_mean_p03,in_mean_p04]
    # stds = [stdd_feis, im_stdd_p01, im_stdd_p02, im_stdd_p03, im_stdd_p04, in_stdd_p01, in_stdd_p02, in_stdd_p03, in_stdd_p04]

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
               label=labels[index], capsize=4)
    ax.set_ylabel('Accuracy (%)', size=12)
    ax.set_xticks([r + bar_w*4 for r in range(len(in_mean_p01))], data_types, size=12)
    # ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    plt.axhline(6.25, ls='--', c='grey', label='Chance level', lw=1)
    plt.legend(loc=1, prop={'size': 11}, ncol=2)
    plt.show()

    # grey bars
    """
    mean_per_dtype = []
    std_per_dtype = []
    for index, data_type in enumerate(data_types):
        dtype_means = [mean[index] for mean in means]
        dtype_mean = np.mean(dtype_means)
        dtype_stdd = np.std(dtype_means)
        mean_per_dtype.append(dtype_mean)
        std_per_dtype.append(dtype_stdd)

    ax.bar(x_pos, mean_per_dtype, yerr=std_per_dtype, align='center', alpha=0.5, ecolor='black', capsize=4, width=0.5,
           color='grey')
    ax.set_ylabel('Accuracy (%)', size=12)
    ax.set_xticks(x_pos, size=12)
    ax.set_xticklabels(data_types)
    plt.axhline(6.25, ls='--', c='grey', label='Chance level', lw=1)
    plt.legend()
    ax.yaxis.grid(True)
    fig.set_size_inches(5.2, 3)
    plt.show()"""


def ab_plot_bar_imagined_vs_inner():
    data_types = ['Raw', 'Preprocessed', 'LF', 'NLF', 'MFCC']
    x_pos = np.arange(len(data_types))

    # **************** ADAPTIVE BOOSTING e=200, lr=1.0 **********************
    print(":::::::::::::::::::::::: AB ::::::::::::::::::::::::\n")
    # p01
    # im
    im_mean_p01 = [7.4, 7.52, 6.39]
    im_stdd_p01 = [0.628, 1.25, 0.767]
    # in
    in_mean_p01 = [9.28, 7.59, 7.27]
    in_stdd_p01 = [0.713, 1.5, 0.54]
    # p02
    # im
    im_mean_p02 = [8.65, 9.09, 6.4]
    im_stdd_p02 = [1.08, 1.17, 1.08]
    # in
    in_mean_p02 = [6.65, 7.4, 6.14]
    in_stdd_p02 = [0.635, 0.615, 0.349]
    # p03
    # im
    im_mean_p03 = [9.28, 8.53, 6.65]
    im_stdd_p03 = [1.13, 0.853, 1.1]
    # in
    in_mean_p03 = [7.9, 8.46, 6.52]
    in_stdd_p03 = [0.817, 0.399, 0.579]
    # p04
    # im
    im_mean_p04 = [8.46, 6.9, 7.02]
    im_stdd_p04 = [0.817, 0.779, 0.386]
    # in
    in_mean_p04 = [7.46, 6.9, 6.08]
    in_stdd_p04 = [0.933, 1.19, 1.46]

    imagined_means = [np.mean(im_mean_p01),
                      np.mean(im_mean_p02),
                      np.mean(im_mean_p03),
                      np.mean(im_mean_p04)]
    imagined_stds = [np.std(im_mean_p01),
                     np.std(im_mean_p02),
                     np.std(im_mean_p03),
                     np.std(im_mean_p04)]
    inner_means = [np.mean(in_mean_p01),
                   np.mean(in_mean_p02),
                   np.mean(in_mean_p03),
                   np.mean(in_mean_p04)]
    inner_stds = [np.std(in_mean_p01),
                  np.std(in_mean_p02),
                  np.std(in_mean_p03),
                  np.std(in_mean_p04)]

    means = [imagined_means, inner_means]
    stds = [imagined_stds, inner_stds]

    bar_w = 0.4

    br1 = np.arange(len(imagined_means))
    br2 = [x + bar_w for x in br1]
    brs = [br1, br2]

    # colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    colours = ['tab:cyan', 'tab:red']
    labels = ['Imagined',
              'Inner'
              ]

    fig, ax = plt.subplots()

    for index, item in enumerate(means):
        ax.bar(brs[index], means[index], yerr=stds[index], color=colours[index], width=bar_w, edgecolor='grey',
               label=labels[index], capsize=4)
    ax.set_ylabel('Accuracy (%)', size=12)
    ax.set_xticks([r + bar_w / 2 for r in range(len(imagined_means))], ['P01', 'P02', 'P03', 'P04'], size=12)
    # ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    #plt.axhline(6.25, ls='--', c='grey', label='Chance level', lw=1)
    plt.legend(loc=1, prop={'size': 11}, ncol=3)

    fig.set_size_inches(5.2, 4)
    plt.show()


def gnb_plot_bar_imagined_vs_inner():
    data_types = ['Raw', 'Preprocessed', 'LF', 'NLF', 'MFCC']
    x_pos = np.arange(len(data_types))

    # **************** GAUSSIAN NAIVE BAYES **********************
    print(":::::::::::::::::::::::: GNB ::::::::::::::::::::::::\n")
    # p01
    # im
    im_mean_p01 = [6.96, 7.02, 7.27]
    im_stdd_p01 = [0.0817, 0.0921, 1.49, 0.983, 0.779]
    # in
    in_mean_p01 = [6.96, 6.77, 6.52]
    in_stdd_p01 = [0.118, 0.063, 2.12, 1.53, 0.316]
    # p02
    # im
    im_mean_p02 = [8.09, 7.65, 9.34]
    im_stdd_p02 = [0.194, 0.0705, 0.666, 0.614, 1.69]
    # in
    in_mean_p02 = [7.59, 7.02, 6.39]
    in_stdd_p02 = [0.11, 0.0974, 0.687, 1.29, 0.921]
    # p03
    # im
    im_mean_p03 = [7.65, 6.9, 7.52]
    im_stdd_p03 = [0.133, 0.0494, 1.17, 1.14, 1.67]
    # in
    in_mean_p03 = [7.21, 7.71, 7.34]
    in_stdd_p03 = [0.155, 0.171, 1.24, 0.645, 0.938]
    # p04
    # im
    im_mean_p04 = [7.34, 7.46, 7.46]
    im_stdd_p04 = [0.0933, 0.107, 0.413, 1.34, 1.17]
    # in
    in_mean_p04 = [7.84, 7.34, 5.83]
    in_stdd_p04 = [0.108, 0.041, 1.15, 1.62, 0.767]

    imagined_means = [np.mean(im_mean_p01),
                      np.mean(im_mean_p02),
                      np.mean(im_mean_p03),
                      np.mean(im_mean_p04)]
    imagined_stds = [np.std(im_mean_p01),
                     np.std(im_mean_p02),
                     np.std(im_mean_p03),
                     np.std(im_mean_p04)]
    inner_means = [np.mean(in_mean_p01),
                   np.mean(in_mean_p02),
                   np.mean(in_mean_p03),
                   np.mean(in_mean_p04)]
    inner_stds = [np.std(in_mean_p01),
                  np.std(in_mean_p02),
                  np.std(in_mean_p03),
                  np.std(in_mean_p04)]

    means = [imagined_means, inner_means]
    stds = [imagined_stds, inner_stds]

    bar_w = 0.4

    br1 = np.arange(len(imagined_means))
    br2 = [x + bar_w for x in br1]
    brs = [br1, br2]

    # colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    colours = ['tab:cyan', 'tab:red']
    labels = ['Imagined',
              'Inner'
              ]

    fig, ax = plt.subplots()

    for index, item in enumerate(means):
        ax.bar(brs[index], means[index], yerr=stds[index], color=colours[index], width=bar_w, edgecolor='grey',
               label=labels[index], capsize=4)
    ax.set_ylabel('Accuracy (%)', size=12)
    ax.set_xticks([r + bar_w / 2 for r in range(len(imagined_means))], ['P01', 'P02', 'P03', 'P04'], size=12)
    # ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    plt.legend(loc=1, prop={'size': 11}, ncol=2)
    fig.set_size_inches(5.2, 4)
    plt.show()


def lda_plot_bar_imagined_vs_inner():
    data_types = ['Raw', 'Preprocessed', 'LF', 'NLF', 'MFCC']
    x_pos = np.arange(len(data_types))

    # **************** LINEAR DISCRIMINANT ANALYSIS **********************
    print(":::::::::::::::::::::::: LDA ::::::::::::::::::::::::\n")
    # p01
    # im
    im_mean_p01 = [11.0, 10.3, 7.15]
    im_stdd_p01 = [0.134, 0.0747, 0.548, 1.97, 0.455]
    # in
    in_mean_p01 = [9.78, 9.15, 8.15]
    in_stdd_p01 = [0.132, 0.112, 1.71, 1.48, 0.385]
    # p02
    # im
    im_mean_p02 = [10.6, 8.15, 8.02]
    im_stdd_p02 = [0.146, 0.184, 0.948, 1.42, 0.767]
    # in
    in_mean_p02 = [8.4, 8.09, 6.77]
    in_stdd_p02 = [0.146, 0.11, 1.03, 1.39, 1.17]
    # p03
    # im
    im_mean_p03 = [12.2, 10.5, 7.59]
    im_stdd_p03 = [0.0857, 0.159, 0.154, 1.23, 0.233]
    # in
    in_mean_p03 = [11.0, 10.0, 7.71]
    in_stdd_p03 = [0.0958, 0.12, 0.728, 2.02, 0.4]
    # p04
    # im
    im_mean_p04 = [10.6, 8.78, 6.21]
    im_stdd_p04 = [0.104, 0.0991, 1.16, 1.44, 1.2]
    # in
    in_mean_p04 = [8.96, 8.28, 6.77]
    in_stdd_p04 = [0.184, 0.0969, 0.75, 1.23, 0.154]

    imagined_means = [np.mean(im_mean_p01),
                      np.mean(im_mean_p02),
                      np.mean(im_mean_p03),
                      np.mean(im_mean_p04)]
    imagined_stds = [np.std(im_mean_p01),
                     np.std(im_mean_p02),
                     np.std(im_mean_p03),
                     np.std(im_mean_p04)]
    inner_means = [np.mean(in_mean_p01),
                   np.mean(in_mean_p02),
                   np.mean(in_mean_p03),
                   np.mean(in_mean_p04)]
    inner_stds = [np.std(in_mean_p01),
                  np.std(in_mean_p02),
                  np.std(in_mean_p03),
                  np.std(in_mean_p04)]

    means = [imagined_means, inner_means]
    stds = [imagined_stds, inner_stds]

    bar_w = 0.4

    br1 = np.arange(len(imagined_means))
    br2 = [x + bar_w for x in br1]
    brs = [br1, br2]

    # colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    colours = ['tab:cyan', 'tab:red']
    labels = ['Imagined',
              'Inner'
              ]

    fig, ax = plt.subplots()

    for index, item in enumerate(means):
        ax.bar(brs[index], means[index], yerr=stds[index], color=colours[index], width=bar_w, edgecolor='grey',
               label=labels[index], capsize=4)
    ax.set_ylabel('Accuracy (%)', size=12)
    ax.set_xticks([r + bar_w / 2 for r in range(len(imagined_means))], ['P01', 'P02', 'P03', 'P04'], size=12)
    # ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    plt.legend(loc=1, prop={'size': 11}, ncol=2)
    fig.set_size_inches(5.2, 4)
    plt.show()


def eegnet_plot_bar_imagined_vs_inner():
    data_types = ['Raw', 'Preprocessed', 'LF', 'NLF', 'MFCC']
    x_pos = np.arange(len(data_types))
    # binary
    # means = [90.0, 90.0, 100, 99.0, 100]
    # stds = [12.3, 12.3, 0.00, 2.00, 0.00]

    # EEGNET
    print(":::::::::::::::::::::::: EEGNet ::::::::::::::::::::::::\n")
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

    imagined_means = [np.mean(im_mean_p01),
                      np.mean(im_mean_p02),
                      np.mean(im_mean_p03),
                      np.mean(im_mean_p04)]
    imagined_stds = [np.std(im_mean_p01),
                     np.std(im_mean_p02),
                     np.std(im_mean_p03),
                     np.std(im_mean_p04)]
    inner_means = [np.mean(in_mean_p01),
                   np.mean(in_mean_p02),
                   np.mean(in_mean_p03),
                   np.mean(in_mean_p04)]
    inner_stds = [np.std(in_mean_p01),
                  np.std(in_mean_p02),
                  np.std(in_mean_p03),
                  np.std(in_mean_p04)]

    means = [imagined_means, inner_means]
    stds = [imagined_stds, inner_stds]

    bar_w = 0.4

    br1 = np.arange(len(imagined_means))
    br2 = [x + bar_w for x in br1]
    brs = [br1, br2]

    # colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    colours = ['tab:cyan', 'tab:red']
    labels = ['Imagined',
              'Inner'
              ]

    fig, ax = plt.subplots()

    for index, item in enumerate(means):
        ax.bar(brs[index], means[index], yerr=stds[index], color=colours[index], width=bar_w, edgecolor='grey',
               label=labels[index], capsize=4)
    ax.set_ylabel('Accuracy (%)', size=12)
    ax.set_xticks([r + bar_w / 2 for r in range(len(imagined_means))], ['P01', 'P02', 'P03', 'P04'], size=12)
    # ax.set_xticklabels(data_types)
    ax.yaxis.grid(True)
    plt.legend(loc=1, prop={'size': 11}, ncol=2)
    fig.set_size_inches(5.2, 4)
    plt.show()


if __name__ == '__main__':
    bar_plots_all_bars_and_mean_of_data_types()
    #ab_plot_bar_imagined_vs_inner()
    #gnb_plot_bar_imagined_vs_inner()
    #lda_plot_bar_imagined_vs_inner()
    #eegnet_plot_bar_imagined_vs_inner()
    exit()



