# Some notes on Kernel Ridge Regression (KRR)

Date: 11th January, 2018

## Overview of some cost functions

Let $X = [x_1, x_2, \ldots, x_N]$ be a vector N sample points, $X \in \mathbf{R}^{N\times D}$, and let $Y = [y_1, y_2, \ldots, y_N]$ be a vector N labeled points, $Y \in \mathbf{R}^{N}$.

###### Ordinary Least Squares Regression (OLS)

$$
RSS(w) = ||{\bf y - w^{\top}x}||^2 + \lambda ||{\bf w}||^2 \\
RSS(w) = \frac{1}{2} \sum_{i}^{N} (y_i - w^{\top}x_i)^2
$$

where $w$ are the weights associated with $x$ points used to predict $\hat{y}$. This is the mean squared error. The unknown components of this equation, the weights $w$, can be solved using the following scheme:

$$
w =
$$

###### Regularized Regression

This function comes from the Tikinov regularization theorem. In this case we want to regularize with the weights. This puts a penalty on the derivative of the function which will enforce smoothness. Essentially, we do not want a function that has too many wiggles and peaks otherwise we would just be fitting all of the points.

$$
\begin{align}
RSS(w, \lambda) &= \sum_{i}^{N}(y_i - w^{\top}x_i)^2 + \lambda\sum_j^{N} w_j^2 \\
RSS(w, \lambda) &= (y-w^{\top}x)^{\top}(y-w^{\top}x) + \lambda w^{\top}w \\
RSS(w, \lambda) &= y^{\top}y - y^{\top}w^{\top}x - x^{\top}wy + x^{\top}ww^{\top}x + \lambda w^{\top}w \\
RSS(w, \lambda) &= y^{\top}y - 2x^{\top}wy + w^{\top} (x^{\top}x)w + \lambda w^{\top}w
\end{align}
$$

where $\lambda$ is the trade-off parameter between the model and the derivative of the function.

We can solve this by taking the derivative with respect to the weights as they are the unknown component of the equation:

$$
\begin{align}
RSS(w, \lambda) &= y^{\top}y - 2x^{\top}wy + w^{\top} (x^{\top}x)w + \lambda w^{\top}w \\
\frac{\partial RSS(w, \lambda)}{\partial w} &= 0 - 2x^{\top}y + 2(x^{\top}x)w + 2\lambda w \\
(x^{\top}x)w + \lambda w &= x^{\top}y\\
w &= \left(x^{\top}x + \lambda I\right)^{-1} x^{\top}y
\end{align}
$$

We now have a nice closed form solution for the regularized linear regression problem.

##### Generalized Regression Format with

$$
RSS(w, \lambda) = \frac{1}{2} \sum_{i}^{N}(y_i - f(x_i))^2 + \lambda\sum_j^{N} D^kf(x_j)^2
$$

where $k$ is the $k^{th}$ derivative of the function $f(x)$.


### Things to Recall

#### Calculus

The first derivative of a square matrix $A \in \mathbf{R}^{N \times N}$ and a vector $x \in \mathbf{R}^{N \times D}$
$$
\begin{align}
\frac{\partial \left(x^{\top}Ax  \right)}{\partial x} &= Ax + A^{\top}x \\
&= (A + A^{\top})x \\
&= 2Ax
\end{align}
$$

$$
\begin{align}
\frac{\partial \left(w^{\top}w  \right)}{\partial w} &= \frac{\partial \left(w^{\top} I w  \right)}{\partial w} \\
\frac{\partial \left(w^{\top} I w  \right)}{\partial w} &= (I + I^{\top})w \\
&= 2w
\end{align}
$$

$$
\frac{\partial \left(2w  \right)}{\partial w} = 2 > 0
$$

Any scalar is invariant under transposition such that:

$$
y^{\top}Xw = \left( y^{\top}X w \right)^{\top} = w^{\top}x^{\top}y
$$
