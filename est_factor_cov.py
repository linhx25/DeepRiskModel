import os
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from utils import get_exp_weights, weighted_cov


def calc_factor_cov(factor_ret, tau=252, lags=0, return_type='cov'):
    """calculate covariance

    Args:
        factor_ret (pandas.DataFrame): factor return
        tau (int, optional): exponential decay halflife
        lags (int, optional): delay lags for Newey-West adjustment
            when `lags=0` Newey-West won't be used
        return_type (str): return cov/vola/corr

    Returns:
        pandas.DataFrame: factor cov/vola/corr

    Reference:
        - Newey, W. K., & West, K. D. (1986). A simple, positive semi-definite, \
          heteroskedasticity and autocorrelation consistent covariance matrix.
    """
    # get shape
    Tn, Fn = factor_ret.shape

    # get exponential decay weights
    w = get_exp_weights(tau, Tn)

    # calculate the cov matrix
    F = weighted_cov(factor_ret.values, w)

    # Newey-West adjustment for autocorrelation
    for d in range(1, lags + 1):
        S = weighted_cov(factor_ret.values, w, lags=d)
        F += (1 - d / (lags + 1.)) * (S + S.T)

    # transform into dataframe
    F = pd.DataFrame(F, index=factor_ret.columns,
                     columns=factor_ret.columns)
    vola = pd.Series(np.sqrt(np.diag(F)), index=factor_ret.columns)

    # return covariance
    if return_type == 'cov':
        return F

    # return correlation
    if return_type == 'corr':
        return F / np.outer(vola, vola)

    # return volatility
    if return_type == 'vola':
        return vola

    raise ValueError('unknown return_type `%s`'%return_type)


def adj_factor_cov_eigen(factor_cov, T_mc=252, N_mc=1000, alpha=1.4, n_skip=15):
    """adjust factor covariance by Eigenfactor Risk Adjustment

    Args:
        factor_cov (pandas.DataFrame): factor return covariance
        T_mc (int, optional): simulated data length for Monte Carlo simulation
        N_mc (int, optional): total number of Month Carlo simulation
        alpha (float, optional): scale constant for eigen volatilities
        n_skip (int, optional): skip first n eigvalus during polyfit

    Returns:
        pandas.DataFrame: adjusted factor covariance

    References:
        - Menchero, J., Orr, D. J., & Wang, J. (2011). \
          The Barra US equity model (USE4), methodology notes. \
          MSCI Barra., P41-P42
    """

    # Monte Carlo Simulation
    F0 = factor_cov.values
    s0, U0 = np.linalg.eigh(F0) # NOTE: F0 = U0 @ diag(s0) @ U0.T
    D0 = U0.T @ F0 @ U0
    V = []
    for _ in range(N_mc):
        bm = np.random.normal(scale=s0**0.5, size=(T_mc, len(s0))).T # [F, T]
        fm = U0 @ bm # [F, F] x [F, T] => [F, T]
        Fm = np.cov(fm) # [F, F]
        sm, Um = np.linalg.eigh(Fm)
        Dm = Um.T @ Fm @ Um
        Dm_hat = Um.T @ F0 @ Um
        V.append(np.diag(Dm_hat) / sm)
    v = np.sqrt(np.mean(V, axis=0))

    # Parabolic Fit
    # x = np.arange(len(v))
    # p = np.poly1d(np.polyfit(x[n_skip+1:], v[n_skip+1:], 2))
    # v = alpha*(p(x) - 1) + 1

    # Adjust
    D0_hat = np.diag(v**2) @ D0
    F0_hat = U0 @ D0_hat @ U0.T

    return pd.DataFrame(F0_hat, index=factor_cov.index,
                        columns=factor_cov.columns)


def adj_factor_cov_vra(factor_cov, bias_stats, tau=42):
    """adjust factor covariance by Volatility Regime Adjustment

    Args:
        factor_cov (pandas.DataFrame): factor return covariance
        bias_stats (list): history bias statistic B^2
        tau (int): exponential decay halflife

    Returns:
        factor_cov (pandas.DataFrame): adjusted factor covariance
        lamb (float): volatility adjustment multiplier

    References:
        - Menchero, J., Orr, D. J., & Wang, J. (2011). \
          The Barra US equity model (USE4), methodology notes. \
          MSCI Barra., P24-P25
    """

    # calc factor volatility multiplier
    lamb = 1.0
    if len(bias_stats) >= tau:
        w = get_exp_weights(tau, len(bias_stats))
        lamb = np.average(bias_stats, weights=w) # NOTE: \lambda^2

    # adjust covariance
    factor_cov *= lamb

    return factor_cov, np.sqrt(lamb)


def run(factor_ret, tau_corr=504, lags_corr=2,
        tau_vola=84, lags_vola=5, tau_vra=42,
        max_T=None, adj_eigen=True, adj_vra=True):


    min_T = max([tau_corr, tau_vola, tau_vra]) # ensure 50%
    if max_T is None:
        max_T = int(min_T * np.log(1 - 0.95) / np.log(0.5)) # ensure 95%

    bias_stats = []
    multipliers = []

    res = dict()
    iterator = tqdm(range(min_T, len(factor_ret)))

    for i in iterator:

        date = factor_ret.index[i]
        iterator.set_description(str(date)[:10])

        # calc covariance
        slc = slice(max(i - max_T, 0), i) # NOTE: i is not included
        corr = calc_factor_cov(factor_ret.iloc[slc], tau=tau_corr,
                               lags=lags_corr, return_type='corr')
        vola = calc_factor_cov(factor_ret.iloc[slc], tau=tau_vola,
                               lags=lags_vola, return_type='vola')
        cov = corr * np.outer(vola, vola)

        # adjust covariance
        if adj_eigen:
            cov = adj_factor_cov_eigen(cov)

        if adj_vra:
            cov, lamb = adj_factor_cov_vra(cov, bias_stats[-max_T:], tau_vra)
            multipliers.append(lamb)
        else:
            multipliers.append(1.0)

        # update bias
        B = np.mean(factor_ret.iloc[i]**2 / np.diag(cov)) # NOTE: B^2
        bias_stats.append(B)

        res[date] = cov

    factor_cov = pd.concat(res, axis=0)

    bias_stats = pd.Series(bias_stats, index=factor_ret.index[min_T:])
    multipliers = pd.Series(multipliers, index=factor_ret.index[min_T:])

    return factor_cov, bias_stats, multipliers


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', default='data')

    parser.add_argument('--lags_corr', type=int, default=0)
    parser.add_argument('--lags_vola', type=int, default=0)
    parser.add_argument('--tau_corr', type=int, default=240)
    parser.add_argument('--tau_vola', type=int, default=60)
    parser.add_argument('--tau_vra', type=int, default=20)
    parser.add_argument('--max_T', type=int, default=480)
    parser.add_argument('--adj_eigen', action='store_true')
    parser.add_argument('--adj_vra', action='store_true', default=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    factor_ret = pd.read_pickle(args.outdir + '/factor_ret.pkl').loc[pd.Timestamp('2015-01-01'):]

    factor_cov, bias_stats, multipliers = run(
        factor_ret, lags_corr=args.lags_corr,
        lags_vola=args.lags_vola, tau_corr=args.tau_corr,
        tau_vola=args.tau_vola, tau_vra=args.tau_vra,
        max_T=args.max_T, adj_eigen=args.adj_eigen,
        adj_vra=args.adj_vra)

    factor_cov.to_pickle(args.outdir + '/factor_cov.pkl')
    bias_stats.to_pickle(args.outdir + '/bias_stats.pkl')
    multipliers.to_pickle(args.outdir + '/multipliers.pkl')
