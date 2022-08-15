import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_curve(points, plt, label: str):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    alpha = np.linspace(0, 1, 75)

    interpolator = interp1d(distance, points, kind='quadratic', axis=0)
    interpolated_points = interpolator(alpha)

    plt.plot(*interpolated_points.T, '-', label=label)


df = pd.read_csv('full_knn-P01_imagined_preprocessed.csv')

#k_locs = [i**2 for i in range(0, 12)]
k_locs = [0, 1, 3, 9, 25, 75, 100, 115]
ks = [df['k'].iloc[k] for k in k_locs]
ks.append(df['k'].max())
print(ks)
print(len(ks))

"""
ks = list(np.rint(np.logspace(1, 12, 13, base=2)))
ks[0] = 1.0
ks = [int(k) for k in ks]
print(ks)
print(len(ks))
"""
accuracies_test = [df.loc[df['k'] == k, 'Test_mean'].item() for k in ks]
accuracies_train = [df.loc[df['k'] == k, 'Train_mean'].item() for k in ks]
points_test = np.vstack((ks, accuracies_test)).T
points_train = np.vstack((ks, accuracies_train)).T

plt.figure()

#plot_curve(points_test, plt, 'test')
plot_curve(points_train, plt, 'train')

#plt.plot(*points.T, 'ok', label='original points')
#plt.plot(df['k'], df['Test_mean'], label='original_test')
plt.plot(df['k'], df['Train_mean'], label='original_train')
plt.legend()
plt.show()
