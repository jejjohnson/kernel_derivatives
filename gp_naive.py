"""Quick script to demonstrate how to find gaussian process regression.

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
from matplotlib import pyplot as plt

# generate datasets
random_state = 123
num_points = 100
x_data = np.linspace(-2*np.pi, 2*np.pi, num_points)[:, np.newaxis]


sin = lambda x: np.sin(x).flatten()

cos = lambda x: np.cos(x).flatten()

tanh = lambda x: np.tanh(x).flatten()


f = sin
y_data = f(x_data) + 0.0001 * np.random.randn(num_points)


# plot the function
fig, ax = plt.subplots()

ax.plot(x_data, y_data)
plt.title('Original Function')
plt.show()

# ---------------------
# SPLIT INTO TRAINING
# ---------------------
train_prnt = 0.6

x_train, x_test = train_test_split(x_data, train_size=train_prnt, random_state=random_state)

# sort the training and test points (for plotting purposes)
x_train, x_test = np.sort(x_train), np.sort(x_test)


y_train = f(x_train) + 0.0001 * np.random.randn(x_train.shape[0])
y_test = f(x_test) + 0.0001 * np.random.randn(x_test.shape[0])

# remove the mean from y data
y_data = y_data - np.mean(y_data)
y_train = y_train - np.mean(y_train)


# ---------------------
# PARAMETER HEURISTICS
# ---------------------

num_parameters = 20
mean_sigma = np.mean(pdist(x_data, metric='euclidean'))
mean_gamma = 1 / (2 * mean_sigma**2)
min_sigma = np.log(mean_sigma * 0.1)
max_sigma = np.log(mean_sigma * 10)

sigma_values = np.logspace(min_sigma, max_sigma, num_parameters)
lam = 1.0e-10
lam_values = np.logspace(-7, 2, num_parameters)

# --------------------
# DATA
# --------------------

# construct kernel matrices
K = rbf_kernel(X=x_data, gamma=mean_gamma)

# perform the cholesky decomposition of the covariance matrix
L = cholesky(K + lam * np.eye(num_points))

# compute the mean of the plot points
Lk = np.linalg.solve(L, K)
mu = np.matmul(L.T, scio.linalg.solve(L, y_data))

# compute the variance
v = scio.linalg.solve(L, K)

# sample n sets
n_sets = 3
f_prior = np.matmul(L, np.random.normal(size=(x_data.shape[0], n_sets)))

# plot the 3 sampled functions
fig, ax = plt.subplots()

ax.plot(x_data, f_prior[:, 0])
ax.plot(x_data, f_prior[:, 1])
ax.plot(x_data, f_prior[:, 2])
plt.title('3 samples from GP Prior')
plt.show()


# --------------------
# TRAINING AND TESTING
# --------------------

# parameter heuristics
mean_sigma = np.mean(pdist(x_train, metric='euclidean'))
mean_gamma = 1 / (2 * mean_sigma**2)
lam = 1.0

# construct kernel matrix for our training points
K_train = rbf_kernel(X=x_train, Y=x_train, gamma=mean_gamma)

# perform the cholesky decomposition of the covariance matrix
L = cholesky(K_train + lam * np.eye(len(x_train)))

# compute the mean of the test points
K_test = rbf_kernel(X=x_train, Y=x_test, gamma=mean_gamma)
Lk = scio.linalg.solve(L, K_test)
mu_test = np.matmul(Lk.T, scio.linalg.solve(L, y_train))

# compute the standard deviation
v_test = np.diag(K_test) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(v_test)

# plot the 3 sampled functions
fig, ax = plt.subplots()

# ax.plot(x_data, y_data, 'r+', ms=8, label='data')
ax.plot(x_test, y_test, 'b+', label='test')
ax.fill_between(x_test.flatten(),
                mu_test.flatten()-3*stdv,
                mu_test.flatten()+3*stdv, color='gray')

plt.title('Train/Test')
plt.legend()
plt.show()
