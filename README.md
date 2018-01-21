kernel_derivatives

## Notes

### Future Implementations

### Kernel Ridge Regression (Large Scale)

* [X] Naive Implementation
* Solver (Naive, Scikit (_solve_cholesky_kernel), cholesky solve, cholesky factor)
* Cross Validation (Naive, Parallel)
* Batch Predictions (Naive, Parallel)
* Data Transformations (Nystrom, RBFSample, RandomProjections)
* KRR Nystrom
* KRR RFF
* CUR Matrix Decomposition

### Future Benchmarks

* Euclidean Distances
* Nearest Neighbour Searches
* Eigenvalue Decompositions - [paper](https://www.researchgate.net/publication/281455336_NumPy_SciPy_Recipes_for_Data_Science_EigenvaluesEigenvectors_of_Covariance_Matrices)
* Kernel Matrix Approximations
* RBF Derivatives

#### Parts to Implement (Assessing Functions)

* Parameter Grid - [sklearn](http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/svm/plot_svm_parameters_selection.html) | [sklearn](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
* Learning Curve (Score, Training Examples) - [sklearn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) | [sklearn](http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html)
* Validation Curve (Score, Parameter) - [sklearn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py) | [sklearn](http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html)

### Optimizing Parameters

* Bayesian Optimization with Scikit-Optimize - [Video](https://www.youtube.com/watch?v=DGJTEBt0d-s)


### Saving and Loading Models

* Pickle (python objects), joblib (arrays) - [blog](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)


### Comparing Algorithms

* Feature Map RBF Kernels - [scikit](http://scikit-learn.org/stable/auto_examples/plot_kernel_approximation.html#sphx-glr-auto-examples-plot-kernel-approximation-py)
* Optimizing (Final Timing Comparison) - [blog](http://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/)
* Comparing Density Estimations - [blog](http://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/)
* Comparing Pairwise Distances - [blog](http://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/)
* Benchmarking Nearest Neighbor Searches - [blog](http://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/) | [paper](https://www.researchgate.net/publication/283568278_NumPy_SciPy_Recipes_for_Data_Science_Computing_Nearest_Neighbors)
* Euclidean Distance (memory views) - [blog I](http://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/) | [blog II](http://jakevdp.github.io/blog/2012/08/16/memoryview-benchmarks-2/)
* Squared Euclidean Distance Matrices - [paper](file:///Users/eman/Downloads/np-sp-recipes-1.pdf)
