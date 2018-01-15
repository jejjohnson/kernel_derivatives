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
w=
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


---

### Previous Notes


The general regression problem:

$$\mathcal{C} = \frac{1}{n} \sum \phi \left( y_i, f(x_i) \right) + \text{ penalty}(f)$$

where:

* $\phi$ is the estimation error
* penalty($f$) is the regularization term.

Assume we have the following cost function for ridge regression where we have a regularization penalty (scaled by $\lambda$) added to the cost term like so ([1]):

$$\mathcal{C} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - f(x_i) \right)^2 + \lambda ||f||^{2}_{\mathcal{H}}$$

where:

* $\alpha$ are the weights
* $K$ is the kernel matrix
* $\lambda$ is the trade off parameter between the regularization and the mean squared error.
* $\Omega$ is the regularization that we choose (e.g. $||w||$, $||f||$, $||\partial f||$, $||\partial^2 f||$)

[1]: https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/10.pdf

This results in the following formula:

$$\mathbf{w} = (\mathbf{x}^T\mathbf{x}+ \lambda \mathbf{I})^{-1} \mathbf{x}^T \mathbf{y}$$

##### Residual Sum of  Squares (RSS)
Cost function:

$$
\begin{align}
\text{C}_{RS}\left(w, \lambda \right) &= ||y-Xw||^2 \\
\text{C}_{RSS}\left(w, \lambda \right) &= \left( y-Xw \right)^T \left( y-Xw \right) \\
\text{C}_{RSS}\left(w, \lambda \right) &= y^Ty - 2X^T w^T y + w^T \left( X^T X \right)w
\end{align}
$$

Derivative of the cost function w.r.t. $w$:

$$
\begin{align}
\frac{\partial \text{C}_{RSS}\left(w, \lambda \right)}{\partial w} &= - 2X^Ty + 2X^T Xw = 0 \\
\left( X^T Xw \right) &=  X^Ty \\
w &= \left( X^T X\right)^{-1} X^Ty
\end{align}
$$

##### Ridge Regression (RR) - Penalized Sum of Squares
Cost function:

$$
\begin{align}
\text{C}_{RR}\left(w, \lambda \right) &= ||y-Xw||^2 + \lambda ||w|| \\
\text{C}_{RR}\left(w, \lambda \right) &= \left( y-Xw \right)^T \left( y-Xw \right) + \lambda w^T w \\
\text{C}_{RR}\left(w, \lambda \right) &= y^Ty - 2X^T w^T y + w^T \left( X^T X \right)w + \lambda w^T w
\end{align}
$$

Derivative of the cost function w.r.t. $w$:

$$
\begin{align}
\frac{\partial \text{C}_{RSS}\left(w, \lambda \right)}{\partial w} &= 0 - 2X^Ty + 2X^T Xw + 2\lambda w = 0 \\
\left( X^T Xw + \lambda w \right) &=  X^Ty \\
w &= \left( X^T X + \lambda I \right)^{-1} X^Ty
\end{align}
$$

##### Kernel Ridge Regression (KRR)

Let:

* $X=\phi$
* $w=\phi^T \alpha$
* $K=\phi^T \phi$

Using the cost function for RR, $w = \left( X^T X + \lambda I \right)^{-1} X^Ty$, we can replace all values with the substitutions from above:

$$
\begin{align}
\phi^T \alpha &= \left( \phi^T \phi + \lambda I \right)^{-1} \phi^Ty \\
\alpha &= \left( \phi^T \phi + \lambda I \right)^{-1} y
\end{align}
$$


##### Kernel Ridge Regression w/ Derivative (KRRD)

Let $f=K\alpha$.

Our Cost function is as follows:


$$
\begin{align}
\text{C}_{KRRD}\left(\alpha, \lambda \right) &= ||y-f||^2 + \lambda ||Df|| \\
\text{C}_{KRRD}\left(\alpha, \lambda \right) &= \left( y-K\alpha \right)^T \left( y-K\alpha \right) + \lambda \alpha^T \triangledown K^T \triangledown K \alpha \\
\text{C}_{KRRD}\left(\alpha, \lambda \right) &= y^Ty - 2K^T \alpha^T y + \alpha^T K^T K \alpha + \lambda \alpha^T \triangledown K^T \triangledown K \alpha
\end{align}
$$

Derivative of the cost function w.r.t. $\alpha$:

$$
\begin{align}
\frac{\partial \text{C}_{KRRD}\left(\alpha, \lambda \right)}{\partial w} &= - 2K^Ty + 2K^T K\alpha + 2\lambda \triangledown K^T \triangledown K \alpha = 0 \\
 \left( K^T K \alpha+ \lambda \triangledown K^T \triangledown K \alpha \right) &=  K^Ty \\
\alpha &=  \left( K^T K + \lambda \triangledown K^T \triangledown K \right)^{-1} K^Ty
\end{align}
$$

##### Kernel Ridge Regression w/ 2nd Derivative (KRRD2)

$$
\begin{align}
\text{C}_{KRRD}\left(\alpha, \lambda \right) &= ||y-f||^2 + \lambda ||D^2f|| \\
\text{C}_{KRRD}\left(\alpha, \lambda \right) &= \left( y-K\alpha \right)^T \left( y-K\alpha \right) + \lambda \alpha^T \left( \triangledown^2 K \right)^T \triangledown^2 K \alpha \\
\text{C}_{KRRD}\left(\alpha, \lambda \right) &= y^Ty - 2K^T \alpha^T y + \alpha^T K^T K \alpha + \lambda \alpha^T \left( \triangledown^2 K \right)^T \triangledown^2 K \alpha
\end{align}
$$

Derivative of the cost function w.r.t. $\alpha$:

$$
\begin{align}
\frac{\partial \text{C}_{KRRD}\left(\alpha, \lambda \right)}{\partial w} &= - 2K^Ty + 2K^T K\alpha + 2\lambda \left(\triangledown^2 K\right)^T \triangledown^2 K \alpha = 0 \\
 \left( K^T K \alpha+ \lambda \left(\triangledown^2 K\right)^T \triangledown^2 K \alpha \right) &=  K^Ty \\
\alpha &=  \left( K^T K + \lambda \left(\triangledown^2 K\right)^T \triangledown^2 K \right)^{-1} K^Ty
\end{align}
$$
