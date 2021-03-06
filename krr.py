
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky
import scipy as scio
from matplotlib import pyplot as plt


class KRR(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regression with different regularizers.
    An implementation of KRR algorithm with different
    regularization parameters (weights, 1st derivative and the
    2nd derivative). Used the scikit-learn class system for demonstration
    purposes.

    Parameters
    ----------
    reg : str, {'w', 'df', 'df2'}, (default='w')
        the regularization parameter associated with the
        KRR solution
        
        alpha = inv(K + lam * reg) * y

    solver : str, {'reg', 'chol'}, (default='reg')
        the Ax=b solver used for the weights

    sigma : float, optional(default=None)
        the parameter for the kernel function.
        NOTE - gamma in scikit learn is defined as follows:
            gamma = 1 / (2 * sigma ^ 2)

    lam : float, options(default=None)
        the trade-off parameter between the mean squared error
        and the regularization term.
        
    rbf_solver : 'py', 'py_mem', cy', optional (default='py')
        the solver used to calculate the rbf derivative
    Attributes
    ----------
    weights_ : array, [N x D]
        the weights found fromminimizing the cost function

    K_ : array, [N x N]
        the kernel matrix with sigma parameter
    """

    def __init__(self, reg='w', solver='reg', sigma=None, lam=None, rbf_solver='py'):
        self.reg = reg
        self.solver = solver
        self.sigma = sigma
        self.lam = lam
        self.rbf_solver = rbf_solver

    def fit(self, x, y=None):

        # regularization
        if self.reg not in ['w', 'df', 'd2f']:
            raise ValueError('Unrecognized regularization.')

        # regularization trade off parameter
        if self.lam is None:

            # common heuristic for minimizing the lambda value
            self.lam = 1.0e-4

        # kernel length scale parameter
        if self.sigma is None:

            # common heuristic for finding the sigma value
            self.sigma = np.mean(pdist(x, metric='euclidean'))

        # check solver
        if self.solver not in ['reg', 'chol']:
            raise ValueError('Unrecognized solver. Please chose "chol" or "reg".')
            
        # check rbf solver
        if self.rbf_solver not in ['py', 'cy']:
            raise ValueError('Unrecognized rbf solver. Please choose "py" or "cy".')

        # gamma parameter for the kernel matrix
        self.gamma = 1 / (2 * self.sigma ** 2)

        # calculate kernel function
        self.X_fit_ = x
        self.K_ = rbf_kernel(self.X_fit_, Y=self.X_fit_, gamma=self.gamma)

        # Regularization with the weights
        if self.reg is 'w':

            # regularization simple
            regularization = np.eye(np.size(self.X_fit_))

            # K + lambda Identity
            mat_A = self.K_ + self.lam * regularization
            mat_b = y

        # Regularization with the 1st derivative
        elif self.reg is 'df':

            temp_weights = np.ones(self.X_fit_.shape[0])

            # calculate the derivative
            if self.rbf_solver == "cy":
                try:

                    from rbf_derivative_cy import rbf_derivative as rbf_derivative_cy
                    self.derivative_ = \
                        rbf_derivative_cy(x_train=np.float64(self.X_fit_),
                        x_function=np.float64(self.X_fit_),
                        kernel_mat=np.float64(self.K_),
                        weights=np.float64(temp_weights).squeeze(),
                        gamma=np.float64(self.gamma),
                        n_derivative=1)

                except ImportError:

                    warnings.warn("Chose 'cy' solver but not available.")

                    self.derivative_ = rbf_derivative(x_train=self.X_fit_,
                                                      x_function=self.X_fit_,
                                                      kernel_mat=self.K_,
                                                      weights=temp_weights,
                                                      gamma=self.gamma,
                                                      n_derivative=1)
            elif self.rbf_solver == "py":
                
                self.derivative_ = rbf_derivative(x_train=self.X_fit_,
                                                  x_function=self.X_fit_,
                                                  kernel_mat=self.K_,
                                                  weights=temp_weights,
                                                  gamma=self.gamma,
                                                  n_derivative=2)

            # K * K.T + lambda * Df * Df.T
            mat_A = np.dot(self.K_, self.K_) + self.lam * \
                    np.matmul(self.derivative_.T, self.derivative_)

            # K * y
            mat_b = np.dot(self.K_, y)

        # Regularization with the 2nd derivative
        elif self.reg is 'd2f':

            temp_weights = np.ones(self.X_fit_.shape[0])

            # calculate the derivative
            if self.rbf_solver == "cy":
                try:

                    from rbf_derivative_cy import rbf_derivative as rbf_derivative_cy
                    self.derivative2_ = rbf_derivative_cy(x_train=np.float64(x_train_transformed),
                                                         x_function=np.float64(x_test_transformed[ibatch_index]),
                                                         kernel_mat=np.float64(K_traintest),
                                                         weights=np.float64(KRR_model.dual_coef_).squeeze(),
                                                         gamma=np.float64(gamma),
                                                         n_derivative=2)
                except ImportError:

                    warnings.warn("Chose 'cy' solver but not available.")

                    self.derivative2_ = rbf_derivative(x_train=self.X_fit_,
                                                      x_function=self.X_fit_,
                                                      kernel_mat=self.K_,
                                                      weights=temp_weights,
                                                      gamma=self.gamma,
                                                      n_derivative=2)
            elif self.rbf_solver == "py":
                
                self.derivative2_ = rbf_derivative(x_train=self.X_fit_,
                                                      x_function=self.X_fit_,
                                                      kernel_mat=self.K_,
                                                      weights=temp_weights,
                                                      gamma=self.gamma,
                                                      n_derivative=2)

            # K * K.T + lambda * D2f * D2f.T
            mat_A = np.dot(self.K_, self.K_) + self.lam * \
                    np.matmul(self.derivative2_.T, self.derivative2_)
            # K * y
            mat_b = np.dot(self.K_, y)

        else:
            raise ValueError('Unrecognized regularization parameter.')

        # solve for the weights
        if self.solver is 'reg':

            # regular linalg solver
            self.weights_ = np.linalg.solve(mat_A, mat_b)

        elif self.solver is 'chol':

            try:
                # cholesky decomposition
                L = cholesky(mat_A)
                self.weights_ = scio.linalg.solve(L.T, scio.linalg.solve(L, mat_b))

            except np.linalg.LinAlgError:

                # if cholesky fails, use regular solver
                warnings.warn("Not Positive Definite. Trying regular solver.")
                self.weights_ = np.linalg.solve(mat_A, mat_b)

        else:
            # the case of an unrecognized solver
            raise ValueError('Unrecognized solver. Please chose "chol" or "reg".')

        return self

    def predict(self, x):

        # calculate the kernel function with new points
        K = rbf_kernel(X=x, Y=self.X_fit_, gamma=self.gamma)

        # return the project points
        return np.matmul(K, self.weights_)


def rbf_derivative(x_train, x_function, weights, kernel_mat=None,
                   n_derivative=1, gamma=1.0):
    """This function calculates the rbf derivative
    Parameters
    ----------
    x_train : array, [N x D]
        The training data used to find the kernel model.

    x_function  : array, [M x D]
        The test points (or vector) to use.

    weights   : array, [N x D]
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], default: None
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}
        chooses which nth derivative to calculate

    gamma : float, default: None
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D]
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """

    # initialize rbf kernel
    derivative = np.zeros(np.shape(x_function))

    # check for kernel mat
    if kernel_mat is None:
        kernel_mat = rbf_kernel(x_train, x_function, gamma=gamma)

    # consolidate the parameters
    theta = 2 * gamma

    # 1st derivative
    if n_derivative == 1:

        # loop through dimensions
        for dim in np.arange(0, np.shape(x_function)[1]):

            # loop through the number of test points
            for iTest in np.arange(0, np.shape(x_function)[0]):

                # loop through the number of test points
                for iTrain in np.arange(0, np.shape(x_train)[0]):

                    # calculate the derivative for the test points
                    derivative[iTest, dim] += theta * weights[iTrain] * \
                                              (x_train[iTrain, dim] -
                                               x_function[iTest, dim]) * \
                                              kernel_mat[iTrain, iTest]

    # 2nd derivative
    elif n_derivative == 2:

        # loop through dimensions
        for dim in np.arange(0, np.shape(x_function)[1]):

            # loop through the number of test points
            for iTest in np.arange(0, np.shape(x_function)[0]):

                # loop through the number of test points
                for iTrain in np.arange(0, np.shape(x_train)[0]):
                    derivative[iTest, dim] += weights[iTrain] * \
                                              (theta ** 2 *
                                               (x_train[iTrain, dim] - x_function[iTest, dim]) ** 2
                                               - theta) * \
                                              kernel_mat[iTrain, iTest]

    return derivative


def rbf_derivative_memory(x_train, x_function, kernel_mat,
                          weights, gamma, n_derivative=1):
    """This function calculates the rbf derivative using no
    loops but it requires a large memory load.
    
    Parameters
    ----------
    x_train : array, [N x D]
        The training data used to find the kernel model.

    x_function  : array, [M x D]
        The test points (or vector) to use.

    weights   : array, [N x D]
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], default: None
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}
        chooses which nth derivative to calculate

    gamma : float, default: None
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D]
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """
    n_train_samples = x_train.shape[0]
    n_test_samples = x_function.shape[0]
    n_dimensions = x_train.shape[1]
    
    # create empty derivative matrix
    derivative = np.empty(shape=(n_train_samples,
                                 n_test_samples,
                                 n_dimensions))
    
    # create empty block matrices and sum
    derivative = np.tile(weights[:, np.newaxis, np.newaxis], 
                           (1, n_test_samples, n_dimensions)) * \
                      (np.tile(x_function[np.newaxis, :, :],
                              (n_train_samples, 1, 1)) - \
                      np.tile(x_train[:, np.newaxis, :], 
                           (1, n_test_samples, 1))) * \
                      np.tile(kernel_mat[:, :, np.newaxis], 
                              (1, 1, n_dimensions))
    # multiply by the constant
    derivative *= 2 * gamma**2
    
    # sum all of the training samples to get M x N matrix
    derivative = derivative.sum(axis=0).squeeze()
    
    return derivative


def main():
    """Example script to test the KRR function.
    """
    # generate dataset
    random_state = 0
    num_points = 1000
    x_data = np.arange(0, num_points)

    datasets = {'x': x_data,
                'sin': np.sin(0.01 * x_data),
                'cos': np.cos(0.01 * x_data),
                'tanh': np.tanh(x_data)}

    # reshape into a pandas data frame
    datasets = pd.DataFrame(data=datasets)

    fig, ax = plt.subplots()

    # plot kernel model
    ax.plot(datasets['x'].values, datasets['cos'].values,
            color='k', label='data')

    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.title('Original Data')

    plt.show()

    # Split Data into Training and Testing
    func_test = 'cos'
    train_prnt = 0.6

    x_train, x_test, y_train, y_test = train_test_split(datasets['x'].values,
                                                        datasets[func_test].values,
                                                        train_size=train_prnt,
                                                        random_state=random_state)

    # make a new axis D [N x D]
    x_train, x_test = x_train[:, np.newaxis], x_test[:, np.newaxis]
    y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]

    # remove the mean from y training
    y_train = y_train - np.mean(y_train)

    # initialize the kernel ridge regression model
    krr_model = KRR(reg='df', solver='reg')

    # fit model to data
    krr_model.fit(x_train, y_train)

    # predict using the krr model
    y_pred = krr_model.predict(x_test)

    error = mean_absolute_error(y_test, y_pred)
    print('\nMean Absolute Error: {:.4f}\n'.format(error))

    # plot the results
    fig, ax = plt.subplots()

    # plot kernel model
    ax.scatter(x_test, y_pred, color='k', label='KRR Model')

    # plot data
    ax.scatter(x_test, y_test, color='r', label='Data')

    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.title('Fitted Model')

    plt.show()

    return None


if __name__ == "__main__":

    main()
