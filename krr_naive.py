"""Quick script to demonstrate how to find kernel ridge regression.

Author: Juan Emmanuel Johnson
Date : 14th December, 2017
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as scio
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor, cho_solve, cholesky
from sklearn.metrics.pairwise import rbf_kernel
from time import time

# generate datasets
random_state = 123
num_points = 1000
x_data = np.arange(0, num_points)

datasets = {'x': x_data,
            'sin': np.sin(x_data),
            'cos': np.cos(x_data),
            'tanh': np.tanh(x_data)}

# reshape into a pandas data frame
datasets = pd.DataFrame(data=datasets)

# view the datasets
print(datasets.head())

# Split Data into Training and Testing
func_test = 'sin'
train_prnt = 0.6

x_train, x_test, y_train, y_test = train_test_split(datasets['x'].values,
                                                    datasets[func_test].values,
                                                    train_size=train_prnt,
                                                    random_state=random_state)


x_train, x_test = x_train[:, np.newaxis], x_test[:, np.newaxis]
y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]

# remove the mean from y training
y_train = y_train - np.mean(y_train)

# parameter heuristics
num_parameters = 20
mean_sigma = np.mean(pdist(x_train, metric='euclidean'))
min_sigma = np.log(mean_sigma * 0.1)
max_sigma = np.log(mean_sigma * 10)

sigma_values = np.logspace(min_sigma, max_sigma, num_parameters)
lam = 1.0
lam_values = np.logspace(-7, 2, num_parameters)

# construct kernel matrices
K_train = rbf_kernel(X=x_train, gamma=mean_sigma)
K_test = rbf_kernel(X=x_train, Y=x_test, gamma=mean_sigma)

# slow method: solve problem
t0 = time()
alpha = scio.linalg.solve(K_train + lam * np.eye(x_train.shape[0]), y_train)
t1 = time() - t0
print('Time taken for solve: {}'.format(t1))

# fast method: cholesky decomposition manually
t0 = time()
R = cholesky(K_train + lam * np.eye(x_train.shape[0]))
alpha = scio.linalg.solve(R, scio.linalg.solve(R.T, y_train))
t1 = time() - t0
print('Time taken for cholesky manually: {}'.format(t1))

# fast method: cholesky decomposition with functions
t0 = time()
R, lower = cho_factor(K_train + lam * np.eye(x_train.shape[0]))
alpha = cho_solve((R, lower), y_train)
t1 = time() - t0
print('\nTime taken for cholesky with functions: {:.4f} secs\n'.format(t1))

# project the data
y_pred = (K_test.T @ alpha).squeeze()
