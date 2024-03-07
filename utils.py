import numpy as np
import pandas as pd

from numba import njit


__all__ = (
    'get_exp_weights', 'RLS', 'weighted_mean', 'weighted_var',
    'weighted_zscore', 'weighted_cov', 'weighted_corr',
)

EPS = 1e-16

def get_exp_weights(tau, N):
    """get exponentially decay weights

    Args:
        tau (int): decay halflife
        N (int): sequence length

    Returns:
        numpy.ndarray: decay weights (sum to 1)
    """
    lambd = 0.5**(1. / tau)
    w = lambd**np.arange(N, 0, -1)
    w /= w.sum()
    return w


def RLS(y, X, R=None, r=None, w=None):
    """Constrained (Weighted) Least Square

    Args:
        y (numpy.ndarray): target values, [#obs]
        X (numpy.ndarray): variable matrix, [#obs, #var]
        R (numpy.ndarray, optional): constraints, [#var, #cons] or [#var]
        r (numpy.ndarray, optional): target value for constraints, [#cons]
        w (numpy.ndarray, optional): sample weights, [#obs]

    Returns:
        b (numpy.ndarray): regression coefficients
        resid (numpy.ndarray): regression residual
        tvalues (numpy.ndarray): T-stats for b
        r2 (float): R-Squared

    Reference:
        - [William, 1991] (https://www.jstor.org/stable/2109587)
    """

    # weights
    if w is not None:
        X = X * np.atleast_2d(np.sqrt(w)).T
        y = y * np.sqrt(w)

    # solve
    if R is not None:
        R = np.atleast_2d(R)
        if r is None:
            r = np.zeros(R.shape[0])
        z = np.zeros((len(R), len(R)))
        W = np.block([[X.T @ X, R.T], [R, z]])
        p = np.r_[X.T @ y, r]
        W_inv = np.linalg.pinv(W)
        m = X.shape[1]
        b = W_inv[:m] @ p
        X_inv = W_inv[:m, :m] @ X.T
    else:
        W_inv = np.linalg.pinv(X.T @ X)
        X_inv = W_inv @ X.T
        b = X_inv @ y

    # calc t-value
    resid = y - X @ b
    ss = (resid**2).sum() / (len(resid) - len(b))
    b_var = ss * X_inv @ X_inv.T
    tvalues = b / np.sqrt(np.diag(b_var))

    # calc r2
    r2 = 1 - (resid**2).sum() / (y**2).sum()

    # restore resid
    if w is not None:
        resid /= np.sqrt(w)

    return b, resid, tvalues, r2


def weighted_mean(X, w):
    """weighted mean

    Args:
        X (numpy.ndarray): variable matrix, [#obs, #var]
        w (numpy.ndarray): sample weights, [#obs]

    Returns:
        numpy.ndarray: weighted mean, [#var]
    """
    if X.shape != w.shape:
        w = w.reshape(len(X), -1)

    X = X * w
    mask = ~np.isnan(X)

    return np.nansum(X, axis=0) / \
        (np.nansum(mask * w, axis=0) + EPS)


def weighted_var(X, w, lags=0):
    """weighted variance

    Args:
        X (numpy.ndarray): variable matrix, [#obs, #var]
        w (numpy.ndarray): sample weights, [#obs]
        lags (int): delay lags for Newey-West correction

    Returns:
        numpy.ndarray: weighted variance, [#var]
    """

    mean = weighted_mean(X, w)
    X = X - mean # demean

    return weighted_mean(X[:len(X)-lags] * X[lags:], w[lags:])


def weighted_zscore(X, w):
    """weighted zscore

    This method use weighted mean and *normal std* for zscore.

    Args:
        X (numpy.ndarray): variable matrix, [#obs, #var]
        w (numpy.ndarray): sample weights, [#obs]

    Returns:
        numpy.ndarray: zscored variable matrix, [#obs, #var]
    """

    mean = weighted_mean(X, w)
    std = np.sqrt(weighted_var(X, w))

    return (X - mean) / (std + EPS)


@njit
def _weighted_nancov(X, w, lags=0):
    """weighted covariance with missing value and delay lags

    Args:
        X (numpy.ndarray): variable matrix as `numpy.cov`, [#var, #obs]
        w (numpy.ndarray): sample weights, [#obs]

    Returns:
        numpy.ndarray: weighted covariance
    """

    n, m = X.shape
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if lags == 0 and j < i:
                cov[i, j] = cov[j, i]
            else:
                vals = X[i][:m-lags] * X[j][lags:]
                mask = ~np.isnan(vals)
                w_mask = w[lags:][mask]
                cov[i, j] = np.sum(vals[mask] * w_mask) / (w_mask.sum() + EPS)

    return cov


def weighted_cov(X, w, lags=0):
    """weighted covariance

    cov(x, y, w) = sum_i (w_i * x_i * y_i)  / sum_i w_i
    where x_i, y_i are assumed to be centered.

    Args:
        X (numpy.ndarray): variable matrix, [#obs, #var]
        w (numpy.ndarray): sample weights, [#obs]
        lags (int): delay lags for Newey-West correction

    Returns:
        numpy.ndarray: weighted covariance, [#var]
    """

    mean = weighted_mean(X, w)
    X = X - mean # demean

    return  _weighted_nancov(X.T, w, lags)


def weighted_corr(X, w, lags=0):
    """weighted pearson correlation

    corr(x, y, w) = cov(x, y, w) / \sqrt(cov(x, x, w) * cov(y, y, w))

    Args:
        X (numpy.ndarray): variable matrix, [#obs, #var]
        w (numpy.ndarray): sample weights, [#obs]
        lags (int): delay lags for Newey-West correction

    Returns:
        numpy.ndarray: weighted correlation, [#var]
    """

    cov = weighted_cov(X, w, lags)
    sigma = np.sqrt(np.diag(cov))

    return cov / np.outer(sigma, sigma)
