import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d


df1 = pd.read_csv('knn-P01_imagined_preprocessed.csv')
df2 = pd.read_csv('knn-P01_imagined_preprocessed_part2.csv')
df3 = pd.read_csv('knn-P01_imagined_preprocessed_part3.csv')
df4 = pd.read_csv('knn-P01_imagined_preprocessed_part4.csv')
df5 = pd.read_csv('knn-P01_imagined_preprocessed_part5.csv')
df6 = pd.read_csv('knn-P01_imagined_preprocessed_part6.csv')

df = pd.concat([df1, df2, df3, df4, df5, df6])

k_locs = [i**2 for i in range(0, 12)]
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
accuracies = [df.loc[df['k'] == k, 'Test_mean'].item() for k in ks]
points = np.vstack((ks, accuracies)).T

distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
distance = np.insert(distance, 0, 0)/distance[-1]

inter_methods = [#'slinear',
                 'quadratic',
                 #'cubic'
                 ]
alpha = np.linspace(0, 1, 75)

inter_points = {}
for method in inter_methods:
    interpolator = interp1d(distance, points, kind=method, axis=0)
    inter_points[method] = interpolator(alpha)

plt.figure()
for method_name, curve in inter_points.items():
    plt.plot(*curve.T, '-', label=method_name)

#plt.plot(*points.T, 'ok', label='original points')
plt.plot(df['k'], df['Test_mean'], label='original')
plt.legend()
plt.show()
